#!/usr/bin/env python3
"""
PMC text files と corpus CSV データの対応分析

PMCテキストファイルのうち、corpus CSVに含まれているDOIがあるファイルと
そうでないファイルを特定し、統計を出力する。
"""

from pathlib import Path

import polars as pl


def load_pmc_ids_mapping() -> pl.DataFrame:
    """PMC-ids.csvからPMCID -> DOIのマッピングを作成"""
    print("Loading PMC-ids.csv...")
    pmc_ids_path = Path("data/papers/pmc/PMC-ids.csv")

    # 大きなファイルなので、必要な列のみ読み込み
    try:
        df = pl.read_csv(
            pmc_ids_path,
            columns=["PMCID", "DOI"],
            truncate_ragged_lines=True,  # 不正な行を切り詰め
        )
        # 空のDOIを除外
        df = df.filter(pl.col("DOI").is_not_null() & (pl.col("DOI") != ""))

        print(f"PMC-ids.csv loaded: {len(df)} records with valid DOI")
        return df
    except Exception as e:
        print(f"Error loading PMC-ids.csv: {e}")
        return pl.DataFrame({"PMCID": [], "DOI": []})


def load_corpus_dois() -> set[str]:
    """corpus CSVファイルからDOIのセットを作成"""
    print("Loading corpus CSV files...")
    corpus_dir = Path("data/corpus/2025-08-15-data-citation-corpus-v4.1-csv")
    corpus_files = list(corpus_dir.glob("*.csv"))

    all_dois = set()

    for csv_file in corpus_files:
        print(f"Processing {csv_file.name}...")
        try:
            df = pl.read_csv(csv_file, columns=["publication"])
            # DOI URLからDOIを抽出 (https://doi.org/を除去)
            dois = df["publication"].str.replace("https://doi.org/", "").to_list()
            all_dois.update(dois)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

    # 空文字列やNoneを除外
    all_dois = {doi for doi in all_dois if doi and doi != ""}
    print(f"Total unique DOIs in corpus: {len(all_dois)}")
    return all_dois


def find_pmc_text_files() -> list[tuple[str, Path]]:
    """PMCテキストファイルのリストを取得"""
    print("Finding PMC text files...")
    pmc_dir = Path("data/papers/pmc")

    # PMC*.txtファイルを再帰的に検索
    txt_files: list[Path] = []
    for pattern in ["oa_comm_txt.PMC*/*.txt", "oa_comm_txt.PMC*/PMC*/*.txt"]:
        txt_files.extend(pmc_dir.glob(pattern))

    # ファイル名からPMCIDを抽出
    pmc_files: list[tuple[str, Path]] = []
    for txt_file in txt_files:
        filename = txt_file.name
        if filename.startswith("PMC") and filename.endswith(".txt"):
            pmcid = filename[:-4]  # .txtを除去
            pmc_files.append((pmcid, txt_file))

    print(f"Found {len(pmc_files)} PMC text files")
    return pmc_files


def analyze_coverage(
    pmc_files: list[tuple[str, Path]],
    pmc_doi_mapping: pl.DataFrame,
    corpus_dois: set[str],
) -> tuple[list[tuple[str, str, Path]], list[tuple[str, str | None, Path]]]:
    """PMCテキストファイルとcorpus DOIの対応を分析"""
    print("Analyzing coverage...")

    # PMCIDをキーとした辞書に変換
    pmc_doi_dict: dict[str, str] = {
        row["PMCID"]: row["DOI"] for row in pmc_doi_mapping.iter_rows(named=True)
    }

    files_with_corpus_match: list[tuple[str, str, Path]] = []
    files_without_corpus_match: list[tuple[str, str | None, Path]] = []

    for pmcid, txt_file in pmc_files:
        doi = pmc_doi_dict.get(pmcid)

        if doi and doi in corpus_dois:
            files_with_corpus_match.append((pmcid, doi, txt_file))
        else:
            files_without_corpus_match.append((pmcid, doi, txt_file))

    return files_with_corpus_match, files_without_corpus_match


def main() -> None:
    print("=== PMC Text Files vs Corpus DOI Analysis ===\n")

    # データを読み込み
    pmc_doi_mapping = load_pmc_ids_mapping()
    corpus_dois = load_corpus_dois()
    pmc_files = find_pmc_text_files()

    # 分析実行
    files_with_match, files_without_match = analyze_coverage(
        pmc_files, pmc_doi_mapping, corpus_dois
    )

    # 統計出力
    total_files = len(pmc_files)
    matched_files = len(files_with_match)
    unmatched_files = len(files_without_match)

    print("\n=== Results ===")
    print(f"Total PMC text files: {total_files:,}")
    print(
        f"Files with corpus DOI match: {matched_files:,} ({matched_files / total_files * 100:.2f}%)"
    )
    print(
        f"Files without corpus DOI match: {unmatched_files:,} ({unmatched_files / total_files * 100:.2f}%)"
    )

    # サンプル出力
    print("\n=== Sample matched files ===")
    for i, (pmcid, doi, txt_file) in enumerate(files_with_match[:5]):
        print(f"{pmcid} -> {doi}")
        if i >= 4:
            break

    print("\n=== Sample unmatched files ===")
    for i, (pmcid, doi_maybe, txt_file) in enumerate(files_without_match[:5]):
        doi_info: str = doi_maybe if doi_maybe else "No DOI"
        print(f"{pmcid} -> {doi_info}")
        if i >= 4:
            break

    # 結果をファイルに保存
    print("\nSaving results...")

    # マッチしたファイルのリスト
    with open("pmc_corpus_matched.txt", "w") as f:
        f.write("PMCID\tDOI\tFile_Path\n")
        for pmcid, doi, txt_file in files_with_match:
            f.write(f"{pmcid}\t{doi}\t{txt_file}\n")

    # マッチしなかったファイルのリスト
    with open("pmc_corpus_unmatched.txt", "w") as f:
        f.write("PMCID\tDOI\tFile_Path\n")
        for pmcid, doi_maybe, txt_file in files_without_match:
            doi_str: str = doi_maybe if doi_maybe else "None"
            f.write(f"{pmcid}\t{doi_str}\t{txt_file}\n")

    print("Results saved to pmc_corpus_matched.txt and pmc_corpus_unmatched.txt")


if __name__ == "__main__":
    main()
