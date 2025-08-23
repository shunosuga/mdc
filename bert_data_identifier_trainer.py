#!/usr/bin/env python3
"""
BERT Data Identifier Trainer
MDCコーパスを用いてデータ識別子を検出するBERTモデルを訓練する
メモリ効率的な逐次学習によりディスクに保存せずに学習データを動的生成
"""

import logging
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PMCTextReader:
    """PMCテキストファイルの効率的な読み込み"""

    def __init__(self, pmc_base_dir: str, pmc_ids_file: str):
        self.pmc_base_dir = Path(pmc_base_dir)
        self.pmc_to_doi = self._load_pmc_doi_mapping(pmc_ids_file)
        self.text_files = self._find_text_files()
        logger.info(f"Found {len(self.text_files)} PMC text files")

    def _load_pmc_doi_mapping(self, pmc_ids_file: str) -> Dict[str, str]:
        """PMC ID -> DOIマッピングを読み込み"""
        try:
            df = pd.read_csv(pmc_ids_file)
            mapping = {}
            for _, row in df.iterrows():
                if pd.notna(row.get("DOI")):
                    pmcid = str(row["PMCID"]).replace("PMC", "")
                    doi = str(row["DOI"]).strip()
                    mapping[pmcid] = doi
            logger.info(f"Loaded {len(mapping)} PMC-DOI mappings")
            return mapping
        except Exception as e:
            logger.error(f"Error loading PMC-DOI mapping: {e}")
            return {}

    def _find_text_files(self) -> List[Path]:
        """PMCテキストファイルを探索"""
        text_files = []
        for subdir in self.pmc_base_dir.iterdir():
            if subdir.is_dir() and "PMC" in subdir.name:
                for nested_dir in subdir.iterdir():
                    if nested_dir.is_dir():
                        for txt_file in nested_dir.glob("PMC*.txt"):
                            text_files.append(txt_file)
        return text_files

    def get_paper_text(self, doi: str) -> Optional[str]:
        """DOIに対応する論文テキストを取得"""
        # DOI -> PMC ID逆引き
        pmcid = None
        for pmc, paper_doi in self.pmc_to_doi.items():
            if paper_doi == doi:
                pmcid = pmc
                break

        if not pmcid:
            return None

        # テキストファイルを探索
        for text_file in self.text_files:
            if text_file.stem.replace("PMC", "") == pmcid:
                try:
                    with open(text_file, "r", encoding="utf-8", errors="ignore") as f:
                        return f.read()
                except Exception as e:
                    logger.warning(f"Error reading {text_file}: {e}")
                    return None
        return None


class DataIdentifierPattern:
    """データ識別子のパターン認識とバリデーション"""

    # 一般的なデータ識別子パターン
    IDENTIFIER_PATTERNS = {
        "GEO": re.compile(r"^GS[EMP]\d+$", re.IGNORECASE),
        "SRA": re.compile(r"^SR[RXAEP]\d+$", re.IGNORECASE),
        "PDB": re.compile(r"^[0-9][A-Z0-9]{3}$", re.IGNORECASE),
        "GenBank": re.compile(r"^[A-Z]{1,2}\d+(\.\d+)?$"),
        "DOI": re.compile(r"^10\.\d+/.+$"),
        "ArrayExpress": re.compile(r"^E-[A-Z]+-\d+$", re.IGNORECASE),
        "dbGaP": re.compile(r"^phs\d+$", re.IGNORECASE),
        "Zenodo": re.compile(r"zenodo\.\d+", re.IGNORECASE),
        "Figshare": re.compile(r"figshare\.\d+", re.IGNORECASE),
    }

    @classmethod
    def is_likely_data_identifier(cls, text: str) -> bool:
        """文字列がデータ識別子らしいかを判定"""
        text = text.strip()
        if len(text) < 3:
            return False

        for pattern_type, pattern in cls.IDENTIFIER_PATTERNS.items():
            if pattern.match(text):
                return True
        return False

    @classmethod
    def get_identifier_type(cls, text: str) -> Optional[str]:
        """識別子のタイプを判定"""
        for pattern_type, pattern in cls.IDENTIFIER_PATTERNS.items():
            if pattern.match(text.strip()):
                return pattern_type
        return None


class TrainingDataGenerator:
    """MDCコーパスから学習データを動的生成"""

    def __init__(
        self,
        corpus_dir: str,
        pmc_reader: PMCTextReader,
        negative_sample_ratio: float = 0.3,
    ):
        self.corpus_dir = Path(corpus_dir)
        self.pmc_reader = pmc_reader
        self.negative_sample_ratio = negative_sample_ratio
        self.doi_to_datasets = self._load_corpus_data()

    def _load_corpus_data(self) -> Dict[str, List[str]]:
        """コーパスデータを読み込み、DOI->データセット識別子のマッピングを作成"""
        logger.info("Loading MDC corpus data...")
        doi_to_datasets = defaultdict(list)

        corpus_files = list(self.corpus_dir.glob("*.csv"))
        for file_path in tqdm(corpus_files, desc="Processing corpus files"):
            try:
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    if pd.notna(row.get("publication")) and pd.notna(
                        row.get("dataset")
                    ):
                        doi = row["publication"].replace("https://doi.org/", "")
                        dataset_id = str(row["dataset"]).strip()

                        if dataset_id and dataset_id != "nan":
                            # データ識別子らしいもののみ採用
                            if DataIdentifierPattern.is_likely_data_identifier(
                                dataset_id
                            ):
                                doi_to_datasets[doi].append(dataset_id)
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                continue

        logger.info(f"Loaded {len(doi_to_datasets)} DOIs with datasets")
        return doi_to_datasets

    def _split_into_sentences(self, text: str) -> List[str]:
        """テキストを文に分割"""
        # シンプルな文分割（改行とピリオドベース）
        sentences = []
        for paragraph in text.split("\n"):
            paragraph = paragraph.strip()
            if len(paragraph) > 20:  # 短すぎる行は除外
                # ピリオドで分割
                sent_candidates = re.split(r"[.!?]+\s+", paragraph)
                for sent in sent_candidates:
                    sent = sent.strip()
                    if len(sent) > 10:  # 短すぎる文は除外
                        sentences.append(sent)
        return sentences

    def _create_negative_examples(
        self, text: str, target_identifiers: List[str]
    ) -> List[Dict]:
        """ネガティブサンプル（データ識別子を含まない文）を生成"""
        sentences = self._split_into_sentences(text)
        negative_examples = []

        for sentence in sentences[:20]:  # 最初の20文から選択
            words = sentence.split()
            if len(words) < 5 or len(words) > 100:  # 長さフィルタ
                continue

            # target_identifiersが含まれていないかチェック
            contains_target = False
            for target_id in target_identifiers:
                if target_id.lower() in sentence.lower():
                    contains_target = True
                    break

            if not contains_target:
                # すべてのラベルを0（OTHER）に設定
                labels = [0] * len(words)
                negative_examples.append(
                    {"tokens": words, "labels": labels, "source": "negative"}
                )

        return negative_examples

    def generate_training_examples(self) -> Generator[Dict, None, None]:
        """学習データを逐次生成"""
        logger.info("Starting training data generation...")

        processed_count = 0
        positive_count = 0
        negative_count = 0

        for doi, dataset_ids in tqdm(
            self.doi_to_datasets.items(), desc="Generating examples"
        ):
            paper_text = self.pmc_reader.get_paper_text(doi)
            if not paper_text:
                continue

            # ポジティブサンプル生成
            sentences = self._split_into_sentences(paper_text)

            for dataset_id in dataset_ids:
                for sentence in sentences:
                    if dataset_id.lower() in sentence.lower():
                        words = sentence.split()
                        if len(words) < 5 or len(words) > 100:
                            continue

                        # ラベル生成：データ識別子を含む単語に1を付与
                        labels = []
                        for word in words:
                            if dataset_id.lower() in word.lower():
                                labels.append(1)  # DATA_ID
                            else:
                                labels.append(0)  # OTHER

                        yield {
                            "tokens": words,
                            "labels": labels,
                            "source": "positive",
                            "target_id": dataset_id,
                        }
                        positive_count += 1
                        break  # 同じ識別子は文ごとに1つまで

            # ネガティブサンプル生成
            if random.random() < self.negative_sample_ratio:
                negative_examples = self._create_negative_examples(
                    paper_text, dataset_ids
                )
                for example in negative_examples[:2]:  # 論文あたり最大2つのネガティブ例
                    yield example
                    negative_count += 1

            processed_count += 1

            # 進捗報告
            if processed_count % 100 == 0:
                logger.info(
                    f"Processed {processed_count} papers, "
                    f"generated {positive_count} positive, {negative_count} negative examples"
                )


class DataIdentifierDataset(Dataset):
    """PyTorchデータセット（バッチ処理用）"""

    def __init__(self, examples: List[Dict], tokenizer, max_length: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        tokens = example["tokens"]
        labels = example["labels"]

        # トークナイザーでエンコード
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # ラベルをトークンレベルに調整
        word_ids = encoding.word_ids()
        token_labels = [-100] * len(word_ids)  # -100は損失計算で無視される

        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx < len(labels):
                token_labels[token_idx] = labels[word_idx]

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(token_labels, dtype=torch.long),
        }


class BERTDataIdentifierTrainer:
    """BERTデータ識別子検出モデルの訓練"""

    def __init__(
        self, model_name: str = "allenai/scibert_scivocab_uncased", num_labels: int = 2
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # トークナイザーとモデル
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)

        self.label_names = ["OTHER", "DATA_ID"]

    def train(
        self,
        data_generator: TrainingDataGenerator,
        batch_size: int = 16,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        warmup_steps: int = 1000,
        max_examples_per_epoch: int = 50000,
    ):
        """モデル訓練"""

        # オプティマイザー設定
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        total_steps = (max_examples_per_epoch // batch_size) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        self.model.train()

        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

            # エポックごとに学習データを再生成
            examples = []
            for i, example in enumerate(data_generator.generate_training_examples()):
                examples.append(example)
                if len(examples) >= max_examples_per_epoch:
                    break

            logger.info(f"Generated {len(examples)} examples for epoch {epoch + 1}")

            # データセット作成
            dataset = DataIdentifierDataset(examples, self.tokenizer)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            epoch_loss = 0
            for batch_idx, batch in enumerate(
                tqdm(dataloader, desc=f"Epoch {epoch + 1}")
            ):
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()

                if batch_idx % 100 == 0:
                    logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

    def save_model(self, save_path: str):
        """モデルを保存"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")

    def predict(self, text: str) -> List[str]:
        """テキストからデータ識別子を予測"""
        self.model.eval()

        words = text.split()
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            max_length=128,
            padding=True,
            truncation=True,
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

        # 予測結果を単語レベルに戻す
        word_ids = encoding.word_ids()
        word_predictions = []

        for word_idx in range(len(words)):
            # 各単語に対応するトークンの予測を取得
            token_predictions = []
            for token_idx, w_id in enumerate(word_ids):
                if w_id == word_idx:
                    token_predictions.append(predictions[0][token_idx].item())

            # 単語レベルの予測（いずれかのトークンがDATA_IDならDATA_ID）
            if token_predictions and max(token_predictions) == 1:
                word_predictions.append(1)
            else:
                word_predictions.append(0)

        # データ識別子と予測された単語を抽出
        identified_data = []
        for word, prediction in zip(words, word_predictions):
            if prediction == 1:
                identified_data.append(word)

        return identified_data


def main():
    """メイン実行関数"""
    # 設定
    corpus_dir = "data/corpus/2025-08-15-data-citation-corpus-v4.1-csv"
    pmc_base_dir = "data/papers/pmc"
    pmc_ids_file = "data/papers/pmc/PMC-ids.csv"
    model_save_path = "models/bert_data_identifier"

    # パス存在確認
    if not Path(corpus_dir).exists():
        logger.error(f"Corpus directory not found: {corpus_dir}")
        return

    if not Path(pmc_base_dir).exists():
        logger.error(f"PMC directory not found: {pmc_base_dir}")
        return

    if not Path(pmc_ids_file).exists():
        logger.error(f"PMC IDs file not found: {pmc_ids_file}")
        return

    # 保存ディレクトリ作成
    Path(model_save_path).mkdir(parents=True, exist_ok=True)

    try:
        # コンポーネント初期化
        logger.info("Initializing components...")
        pmc_reader = PMCTextReader(pmc_base_dir, pmc_ids_file)
        data_generator = TrainingDataGenerator(corpus_dir, pmc_reader)
        trainer = BERTDataIdentifierTrainer()

        # 訓練実行
        logger.info("Starting training...")
        start_time = time.time()

        trainer.train(
            data_generator=data_generator,
            batch_size=16,
            num_epochs=3,
            learning_rate=2e-5,
            max_examples_per_epoch=100000,
        )

        # モデル保存
        trainer.save_model(model_save_path)

        end_time = time.time()
        logger.info(f"Training completed in {end_time - start_time:.2f} seconds")

        # テスト予測
        test_text = (
            "The data is available under GSE12345 and SRR67890 in the repository."
        )
        predictions = trainer.predict(test_text)
        logger.info(f"Test prediction: {predictions}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
