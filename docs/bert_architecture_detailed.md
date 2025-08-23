# BERT Data Identifier Trainer - 詳細アーキテクチャ設計書

`bert_data_identifier_trainer.py`の各クラスと処理内容の詳細分析

## システム概要

ModernBERT-baseを使用してトークンレベルでデータ識別子を検出する機械学習システム。MDCコーパスとPMC全文論文を使用して動的に学習データを生成し、メモリ効率的な逐次学習を実現。

## 主要クラス詳細

### 1. PMCTextReader クラス

**目的**: PMC全文論文の効率的な読み込みとDOIマッピング管理

#### 主要メソッド

- **`__init__(pmc_base_dir, pmc_ids_file)`**
  - PMCベースディレクトリとPMC ID-DOIマッピングファイルを初期化
  - テキストファイルの探索を実行

- **`_load_pmc_doi_mapping(pmc_ids_file)`**
  - PMC-ids.csvからPMC ID→DOIのマッピングを作成
  - NANデータの除外とデータクリーニング
  - 戻り値: `Dict[str, str]` (PMC ID → DOI)

- **`_find_text_files()`**
  - PMCディレクトリ階層を再帰的に探索
  - `PMC*.txt`パターンのファイルを収集
  - 戻り値: `List[Path]`

- **`get_paper_text(doi)`**
  - DOIから対応するPMC全文テキストを取得
  - DOI→PMC ID逆引き→テキストファイル読み込み
  - エラーハンドリング付きファイル読み込み

#### データフロー
```
DOI → PMC ID逆引き → テキストファイル特定 → UTF-8読み込み → 全文テキスト返却
```

### 2. DataIdentifierPattern クラス

**目的**: データ識別子の正規表現パターン認識とバリデーション

#### パターン定義
- **GEO**: `^GS[EMP]\d+$` (Gene Expression Omnibus)
- **SRA**: `^SR[RXAEP]\d+$` (Sequence Read Archive)
- **PDB**: `^[0-9][A-Z0-9]{3}$` (Protein Data Bank)
- **GenBank**: `^[A-Z]{1,2}\d+(\.\d+)?$`
- **DOI**: `^10\.\d+/.+$`
- **ArrayExpress**: `^E-[A-Z]+-\d+$`
- **dbGaP**: `^phs\d+$`
- **Zenodo**: `zenodo\.\d+`
- **Figshare**: `figshare\.\d+`

#### 主要メソッド

- **`is_likely_data_identifier(text)`**
  - 文字列がデータ識別子の可能性があるかを判定
  - 最小長チェック（3文字以上）
  - 全パターンに対する一致検証

- **`get_identifier_type(text)`**
  - 識別子の種類を特定
  - パターンマッチングによる分類
  - 戻り値: パターン名 or None

### 3. TrainingDataGenerator クラス

**目的**: MDCコーパスから動的に学習データを生成

#### 初期化パラメータ
- `corpus_dir`: MDCコーパスディレクトリ
- `pmc_reader`: PMCTextReaderインスタンス
- `negative_sample_ratio`: ネガティブサンプル比率（デフォルト0.3）

#### 主要メソッド

- **`_load_corpus_data()`**
  - MDCコーパスのCSVファイル群を読み込み
  - DOI→データセット識別子のマッピング作成
  - データ識別子バリデーション付き
  - 戻り値: `Dict[str, List[str]]`

- **`_split_into_sentences(text)`**
  - 論文全文を文単位に分割
  - 改行とピリオドベースの分割
  - 短すぎる文の除外（10文字未満）
  - 戻り値: `List[str]`

- **`_create_negative_examples(text, target_identifiers)`**
  - ネガティブサンプル（データ識別子を含まない文）を生成
  - 最初の20文から候補を選択
  - 長さフィルタ（5-100単語）
  - 目標識別子が含まれていないことを確認
  - 戻り値: `List[Dict]`（tokens, labels, source）

- **`generate_training_examples()`**
  - 学習データを逐次生成するGenerator
  - ポジティブ・ネガティブサンプルの動的生成
  - メモリ効率的な処理（ディスク保存なし）

#### データ生成フロー
```
MDCコーパス → DOI-データセット → PMC全文 → 文分割 → ラベル付け → 学習例生成
```

#### 学習例の構造
- **ポジティブサンプル**: データ識別子を含む文
  - `tokens`: 単語リスト
  - `labels`: [0,0,1,0,0] (1がデータ識別子)
  - `source`: "positive"
  - `target_id`: 対象識別子

- **ネガティブサンプル**: データ識別子を含まない文
  - `tokens`: 単語リスト
  - `labels`: [0,0,0,0,0] (全て0)
  - `source`: "negative"

### 4. DataIdentifierDataset クラス

**目的**: PyTorch互換のデータセットクラス

#### 主要機能
- トークナイザーによるエンコーディング
- 単語レベルからトークンレベルへのラベル調整
- パディング・トランケーション処理
- -100ラベルによる損失計算制御

#### `__getitem__`処理フロー
```
生の例 → トークナイザー → word_ids取得 → ラベル調整 → PyTorchテンソル
```

### 5. BERTDataIdentifierTrainer クラス

**目的**: BERTモデルの訓練と推論

#### モデル構成
- **ベースモデル**: `answerdotai/ModernBERT-base`
- **タスク**: Token Classification (2クラス: OTHER, DATA_ID)
- **デバイス**: CUDA/CPU自動選択

#### 訓練パラメータ
- `batch_size`: 16
- `num_epochs`: 3
- `learning_rate`: 2e-5
- `warmup_steps`: 1000
- `max_examples_per_epoch`: 50000

#### 主要メソッド

- **`train(data_generator, ...)`**
  - エポック毎の学習データ再生成
  - AdamWオプティマイザー + 線形warmupスケジューラー
  - バッチ処理とロス計算
  - プログレス表示とログ出力

- **`predict(text)`**
  - テキストからデータ識別子を予測
  - 単語分割→トークナイザー→モデル推論
  - トークンレベル予測を単語レベルに集約
  - 戻り値: データ識別子リスト

#### 予測処理フロー
```
入力テキスト → 単語分割 → トークナイザー → BERT → Softmax → Argmax → 単語集約
```

## システム全体フロー

### 訓練フェーズ
```
1. PMCTextReader初期化（論文-DOIマッピング）
2. TrainingDataGenerator初期化（MDCコーパス読み込み）
3. BERTDataIdentifierTrainer初期化（ModernBERT読み込み）
4. 各エポック:
   - 学習データ動的生成（最大100,000例）
   - DatasetとDataLoader作成
   - BERT訓練（AdamW + warmup）
5. モデル保存
```

### 推論フェーズ
```
1. 訓練済みモデル読み込み
2. 入力テキスト前処理
3. BERT推論
4. 後処理（トークン→単語集約）
5. データ識別子リスト出力
```

## 技術的特徴

### メモリ効率性
- 学習データをディスクに保存せず動的生成
- Generatorパターンによる逐次処理
- エポック毎のデータ再生成

### 学習データ品質
- 実際の論文-データセット対応関係を使用
- パターンベースの識別子バリデーション
- ポジティブ・ネガティブサンプルのバランス調整

### モデル性能
- ModernBERT使用による科学論文特化
- トークンレベル分類による高精度検出
- ワームアップ付き学習率スケジュール

## 拡張可能性

1. **新しいパターン追加**: DataIdentifierPatternクラスへの正規表現追加
2. **異なるBERTモデル**: model_name変更による他のBERTモデル利用
3. **アンサンブル**: 複数モデルの予測結果統合
4. **後処理強化**: コンテキスト情報を活用した精度向上