#!/usr/bin/env python3
"""
データ識別子検証スクリプト
MDCコーパスに含まれるデータ識別子が実際の論文テキストに含まれているかを検証する
"""

import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def load_corpus_data(corpus_file):
    """corpus_consolidated.jsonからDOIとデータセット識別子のマッピングを作成"""
    print("=== Loading corpus data ===")

    doi_to_datasets = defaultdict(list)
    all_datasets = set()

    print(f"Loading corpus data from {corpus_file}")
    try:
        with open(corpus_file, "r", encoding="utf-8") as f:
            corpus_data = json.load(f)

        print(f"Loaded {len(corpus_data)} records from corpus file")

        for record in tqdm(corpus_data, desc="Processing corpus records"):
            publication = record.get("publication", "").strip()
            datasets = record.get("datasets", [])

            if publication and datasets:
                # DOI正規化 - 10.xxxxのプレフィックスのみを抽出
                doi = publication.lower()  # 大文字小文字を区別しない

                # プロトコルプレフィックスを削除
                doi = doi.replace("https://doi.org/", "").replace(
                    "http://dx.doi.org/", ""
                )
                doi = doi.replace("https://", "").replace("http://", "")

                # 10.で始まるDOIパターンのみを抽出
                if "10." in doi:
                    # 10.から始まる部分を取得
                    doi_start = doi.find("10.")
                    if doi_start != -1:
                        doi = doi[doi_start:]
                        # スペースや他の区切り文字で終わる可能性があるので、適切に切り取り
                        doi = doi.split()[0]  # 最初の単語のみ取得

                for dataset_id in datasets:
                    if dataset_id and isinstance(dataset_id, str):
                        dataset_id = dataset_id.strip()
                        if dataset_id and dataset_id != "nan":
                            doi_to_datasets[doi].append(dataset_id)
                            all_datasets.add(dataset_id)

    except Exception as e:
        print(f"Error loading corpus data: {e}")
        raise

    print(
        f"Loaded {len(doi_to_datasets)} DOIs with {len(all_datasets)} unique datasets"
    )
    return doi_to_datasets, all_datasets


def load_pmc_doi_mapping(pmc_ids_file):
    """PMC IDとDOIのマッピングを読み込み"""
    print("=== Loading PMC-DOI mapping ===")

    try:
        df = pd.read_csv(pmc_ids_file)
        pmc_to_doi = {}

        for _, row in tqdm(
            df.iterrows(), total=len(df), desc="Processing PMC mappings"
        ):
            if pd.notna(row.get("DOI")):
                pmcid = str(row["PMCID"]).replace("PMC", "")
                doi = str(row["DOI"]).strip()
                pmc_to_doi[pmcid] = doi

        print(f"Loaded {len(pmc_to_doi)} PMC-DOI mappings")
        return pmc_to_doi

    except Exception as e:
        print(f"Error loading PMC-DOI mapping: {e}")
        return {}


def find_pmc_text_files(pmc_base_dir):
    """PMCテキストファイルを探索"""
    print("=== Finding PMC text files ===")

    text_files = []
    base_path = Path(pmc_base_dir) / "txt"  # txtサブディレクトリを追加

    if not base_path.exists():
        print(f"Warning: {base_path} does not exist")
        return text_files

    # 展開済みディレクトリを探索
    for subdir in base_path.iterdir():
        if subdir.is_dir() and "PMC" in subdir.name:
            for txt_file in subdir.glob("PMC*.txt"):
                text_files.append(txt_file)

    print(f"Found {len(text_files)} PMC text files")
    return text_files


def create_search_patterns(dataset_id):
    """データセット識別子の検索パターンを作成"""
    patterns = []

    # DOIまたはURLの場合の特別処理
    if dataset_id.startswith(("http://", "https://")) or "10." in dataset_id:
        # URLからプロトコルを削除
        clean_id = dataset_id
        if clean_id.startswith(("http://", "https://")):
            clean_id = clean_id.replace("https://", "").replace("http://", "")

        # 10.で始まるDOIパターンのみを抽出
        if "10." in clean_id:
            doi_start = clean_id.find("10.")
            if doi_start != -1:
                clean_id = clean_id[doi_start:]
                # スペースや他の区切り文字で終わる可能性があるので、適切に切り取り
                clean_id = clean_id.split()[0]

        # 大文字小文字を区別しない検索パターンを追加
        patterns.append(re.escape(clean_id.lower()))
        patterns.append(re.escape(clean_id.upper()))
        patterns.append(re.escape(clean_id))
    else:
        # 通常のデータセット識別子の処理
        # 基本パターン（そのまま）
        patterns.append(re.escape(dataset_id))

        # 大文字小文字を無視
        patterns.append(re.escape(dataset_id.upper()))
        patterns.append(re.escape(dataset_id.lower()))

    # スペースや区切り文字を含む可能性
    spaced_id = re.sub(r"([A-Za-z])(\d)", r"\1 \2", dataset_id)
    if spaced_id != dataset_id:
        patterns.append(re.escape(spaced_id))

    # ドットやハイフンの前後でのバリエーション
    if "." in dataset_id or "-" in dataset_id:
        alt_id = dataset_id.replace(".", " ").replace("-", " ")
        patterns.append(re.escape(alt_id))

    return patterns


def search_identifiers_in_text(text_content, dataset_ids):
    """テキスト内でデータ識別子を検索"""
    found_identifiers = []

    # テキストを小文字に変換して検索しやすくする
    text_lower = text_content.lower()

    for dataset_id in dataset_ids:
        patterns = create_search_patterns(dataset_id)

        for pattern in patterns:
            # 単語境界を含む検索パターン
            # NOTE: この設定をすると、単語境界がない場合はヒットしない
            regex_pattern = r"\b" + pattern + r"\b"

            try:
                if re.search(regex_pattern, text_content, re.IGNORECASE):
                    found_identifiers.append(
                        {"identifier": dataset_id, "pattern": pattern, "found": True}
                    )
                    break  # 一つ見つかったら次の識別子へ
            except re.error:
                # 正規表現エラーの場合は単純な文字列検索
                if dataset_id.lower() in text_lower:
                    found_identifiers.append(
                        {
                            "identifier": dataset_id,
                            "pattern": "simple_match",
                            "found": True,
                        }
                    )
                    break

    return found_identifiers


def verify_identifiers_in_papers(sample_percentage=5.0):
    """メイン検証処理"""
    print("=== Data Identifier Verification in PMC Papers ===")
    print(f"Sampling {sample_percentage}% of available files")

    # パス設定
    corpus_file = "data/corpus/corpus_consolidated.json"
    pmc_ids_file = "data/pmc/PMC-ids.csv"
    pmc_base_dir = "data/pmc"

    # データ読み込み
    doi_to_datasets, all_datasets = load_corpus_data(corpus_file)
    pmc_to_doi = load_pmc_doi_mapping(pmc_ids_file)
    all_text_files = find_pmc_text_files(pmc_base_dir)

    # サンプリング
    if sample_percentage < 100:
        sample_size = int(len(all_text_files) * sample_percentage / 100)
        text_files = random.sample(
            all_text_files, min(sample_size, len(all_text_files))
        )
        print(
            f"Sampled {len(text_files)} files out of {len(all_text_files)} total files"
        )
    else:
        text_files = all_text_files
        print(f"Processing all {len(text_files)} files")

    # 結果収集用
    verification_results = []
    summary_stats = {
        "total_papers_processed": 0,
        "papers_with_corpus_data": 0,
        "papers_with_found_identifiers": 0,
        "total_identifiers_expected": 0,
        "total_identifiers_found": 0,
        "identifier_types_found": Counter(),
        "processing_errors": 0,
    }

    print("\n=== Starting verification process ===")

    # 進捗バーでテキストファイルを処理
    for text_file in tqdm(text_files, desc="Verifying papers"):
        try:
            # PMC IDを抽出
            pmcid = text_file.stem.replace("PMC", "")

            # DOIを取得
            doi = pmc_to_doi.get(pmcid)
            if not doi:
                continue

            # コーパスデータを確認
            expected_datasets = doi_to_datasets.get(doi, [])
            if not expected_datasets:
                continue

            summary_stats["papers_with_corpus_data"] += 1
            summary_stats["total_identifiers_expected"] += len(expected_datasets)

            # テキストファイルを読み込み
            try:
                with open(text_file, "r", encoding="utf-8", errors="ignore") as f:
                    text_content = f.read()
            except Exception as e:
                summary_stats["processing_errors"] += 1
                continue

            # 識別子を検索
            found_identifiers = search_identifiers_in_text(
                text_content, expected_datasets
            )

            # 結果を記録
            result = {
                "pmcid": f"PMC{pmcid}",
                "doi": doi,
                "expected_datasets": expected_datasets,
                "found_identifiers": found_identifiers,
                "found_count": len(found_identifiers),
                "expected_count": len(expected_datasets),
                "coverage_ratio": len(found_identifiers) / len(expected_datasets)
                if expected_datasets
                else 0,
            }

            verification_results.append(result)

            # 統計更新
            if found_identifiers:
                summary_stats["papers_with_found_identifiers"] += 1
                summary_stats["total_identifiers_found"] += len(found_identifiers)

                for found in found_identifiers:
                    # 識別子の種類を分析
                    identifier = found["identifier"]
                    if identifier.startswith(("GSE", "GSM")):
                        summary_stats["identifier_types_found"]["GEO"] += 1
                    elif identifier.startswith(("SRR", "SRA", "SRX")):
                        summary_stats["identifier_types_found"]["SRA"] += 1
                    elif identifier.startswith(("PDB", "pdb")):
                        summary_stats["identifier_types_found"]["PDB"] += 1
                    elif re.match(r"^[A-Z]{2}\d+", identifier):
                        summary_stats["identifier_types_found"]["GenBank"] += 1
                    else:
                        summary_stats["identifier_types_found"]["Other"] += 1

        except Exception as e:
            summary_stats["processing_errors"] += 1
            continue

        summary_stats["total_papers_processed"] += 1

        # 中間結果表示（100件ごと）
        if summary_stats["total_papers_processed"] % 100 == 0:
            current_found_rate = (
                summary_stats["papers_with_found_identifiers"]
                / summary_stats["papers_with_corpus_data"]
                * 100
                if summary_stats["papers_with_corpus_data"] > 0
                else 0
            )
            tqdm.write(
                f"Progress: {current_found_rate:.1f}% papers have found identifiers"
            )

    # 結果の保存と表示
    save_results(verification_results, summary_stats)
    display_summary(summary_stats)


def save_results(verification_results, summary_stats):
    """結果をファイルに保存"""
    print("\n=== Saving results ===")

    # 詳細結果をJSONで保存
    with open("identifier_verification_results.json", "w", encoding="utf-8") as f:
        json.dump(verification_results, f, indent=2, ensure_ascii=False)

    # サマリーをJSONで保存
    with open("identifier_verification_summary.json", "w", encoding="utf-8") as f:
        # Counterオブジェクトを辞書に変換
        summary_to_save = summary_stats.copy()
        summary_to_save["identifier_types_found"] = dict(
            summary_stats["identifier_types_found"]
        )
        json.dump(summary_to_save, f, indent=2, ensure_ascii=False)

    # 見つかった識別子の詳細CSVを作成
    found_details = []
    for result in verification_results:
        if result["found_identifiers"]:
            for found in result["found_identifiers"]:
                found_details.append(
                    {
                        "pmcid": result["pmcid"],
                        "doi": result["doi"],
                        "identifier": found["identifier"],
                        "pattern": found["pattern"],
                    }
                )

    if found_details:
        df_found = pd.DataFrame(found_details)
        df_found.to_csv("found_identifiers_details.csv", index=False)

    print("Results saved to:")
    print("- identifier_verification_results.json")
    print("- identifier_verification_summary.json")
    print("- found_identifiers_details.csv")


def display_summary(summary_stats):
    """結果サマリーを表示"""
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)

    print(f"Total papers processed: {summary_stats['total_papers_processed']:,}")
    print(f"Papers with corpus data: {summary_stats['papers_with_corpus_data']:,}")
    print(
        f"Papers with found identifiers: {summary_stats['papers_with_found_identifiers']:,}"
    )
    print(f"Processing errors: {summary_stats['processing_errors']:,}")

    if summary_stats["papers_with_corpus_data"] > 0:
        coverage_rate = (
            summary_stats["papers_with_found_identifiers"]
            / summary_stats["papers_with_corpus_data"]
            * 100
        )
        print(f"\nPaper-level coverage: {coverage_rate:.2f}%")

    if summary_stats["total_identifiers_expected"] > 0:
        identifier_rate = (
            summary_stats["total_identifiers_found"]
            / summary_stats["total_identifiers_expected"]
            * 100
        )
        print(f"Identifier-level coverage: {identifier_rate:.2f}%")

    print(
        f"\nTotal expected identifiers: {summary_stats['total_identifiers_expected']:,}"
    )
    print(f"Total found identifiers: {summary_stats['total_identifiers_found']:,}")

    if summary_stats["identifier_types_found"]:
        print("\nIdentifier types found:")
        for id_type, count in summary_stats["identifier_types_found"].most_common():
            print(f"  {id_type}: {count:,}")


if __name__ == "__main__":
    start_time = time.time()
    # デフォルトで5%のサンプリング、コマンドライン引数で変更可能
    import sys

    sample_percentage = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    verify_identifiers_in_papers(sample_percentage)
    end_time = time.time()
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")
