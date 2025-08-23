#!/usr/bin/env python3
"""
PMC論文ファイル削除スクリプト

corpus_consolidated.jsonに含まれていない論文をpmc/txt/から削除する。
PMCidとDOIの対応はPMC-ids.csvから取得する。

使用方法:
    python cleanup_unused_pmc_files.py [--dry-run] [--backup]

    --dry-run: 実際の削除は行わず、削除対象ファイルを表示するのみ
    --backup: 削除前にバックアップディレクトリを作成
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Set

import pandas as pd
from tqdm import tqdm

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_corpus_consolidated(corpus_file: Path) -> Set[str]:
    """
    corpus_consolidated.jsonから使用されているDOIのセットを取得

    Args:
        corpus_file: corpus_consolidated.jsonのパス

    Returns:
        Set[str]: 使用されているDOIのセット
    """
    logger.info(f"Loading corpus consolidated data from {corpus_file}")

    try:
        with open(corpus_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        used_dois = set()
        for record in data:
            publication = record.get("publication", "").strip()
            if publication:
                # DOI形式の正規化（必要に応じて）
                if publication.startswith("10."):
                    used_dois.add(publication)
                elif publication.startswith("https://doi.org/"):
                    used_dois.add(publication.replace("https://doi.org/", ""))
                elif publication.startswith("http://dx.doi.org/"):
                    used_dois.add(publication.replace("http://dx.doi.org/", ""))
                else:
                    used_dois.add(publication)

        logger.info(f"Found {len(used_dois)} unique DOIs in corpus consolidated data")
        return used_dois

    except Exception as e:
        logger.error(f"Error loading corpus consolidated data: {e}")
        raise


def load_pmc_doi_mapping(pmc_ids_file: Path) -> Dict[str, str]:
    """
    PMC-ids.csvからPMCID -> DOIのマッピングを作成

    Args:
        pmc_ids_file: PMC-ids.csvのパス

    Returns:
        Dict[str, str]: PMCID -> DOIのマッピング
    """
    logger.info(f"Loading PMC-DOI mapping from {pmc_ids_file}")

    try:
        # CSVファイルを読み込む
        df = pd.read_csv(pmc_ids_file, dtype=str, low_memory=False)

        # DOIが存在するレコードのみを対象とする
        df = df.dropna(subset=["DOI", "PMCID"])
        df = df[df["DOI"].str.strip() != ""]
        df = df[df["PMCID"].str.strip() != ""]

        # PMCIDからPMCプレフィックスを削除（PMC123456 -> 123456）
        df["PMC_ID_CLEAN"] = df["PMCID"].str.replace("PMC", "", regex=False)

        # マッピング辞書を作成
        pmc_to_doi = dict(zip(df["PMC_ID_CLEAN"], df["DOI"]))

        logger.info(f"Created PMC-DOI mapping with {len(pmc_to_doi)} entries")
        return pmc_to_doi

    except Exception as e:
        logger.error(f"Error loading PMC-DOI mapping: {e}")
        raise


def find_pmc_files(pmc_txt_dir: Path) -> Dict[str, Path]:
    """
    pmc/txt/ディレクトリ内のすべてのPMCファイルを検索

    Args:
        pmc_txt_dir: pmc/txt/ディレクトリのパス

    Returns:
        Dict[str, Path]: PMC ID -> ファイルパスのマッピング
    """
    logger.info(f"Scanning PMC files in {pmc_txt_dir}")

    pmc_files = {}

    if not pmc_txt_dir.exists():
        logger.error(f"PMC directory not found: {pmc_txt_dir}")
        return pmc_files

    # PMC000xxxxxx形式のディレクトリを検索
    for pmc_dir in pmc_txt_dir.glob("PMC*"):
        if not pmc_dir.is_dir():
            continue

        # 各ディレクトリ内のPMC*.txtファイルを検索
        for pmc_file in pmc_dir.glob("PMC*.txt"):
            # ファイル名からPMC IDを抽出（PMC123456.txt -> 123456）
            pmc_id = pmc_file.stem.replace("PMC", "")
            pmc_files[pmc_id] = pmc_file

    logger.info(f"Found {len(pmc_files)} PMC files")
    return pmc_files


def create_backup(pmc_txt_dir: Path, backup_dir: Path) -> None:
    """
    PMCファイルのバックアップを作成

    Args:
        pmc_txt_dir: 元のPMCディレクトリ
        backup_dir: バックアップディレクトリ
    """
    logger.info(f"Creating backup from {pmc_txt_dir} to {backup_dir}")

    if backup_dir.exists():
        logger.warning(f"Backup directory already exists: {backup_dir}")
        response = input("Backup directory exists. Continue? (y/N): ")
        if response.lower() != "y":
            logger.info("Backup cancelled")
            return
        shutil.rmtree(backup_dir)

    shutil.copytree(pmc_txt_dir, backup_dir)
    logger.info(f"Backup completed: {backup_dir}")


def cleanup_unused_files(
    used_dois: Set[str],
    pmc_to_doi: Dict[str, str],
    pmc_files: Dict[str, Path],
    dry_run: bool = False,
) -> None:
    """
    使用されていないPMCファイルを削除

    Args:
        used_dois: 使用されているDOIのセット
        pmc_to_doi: PMCID -> DOIのマッピング
        pmc_files: PMC ID -> ファイルパスのマッピング
        dry_run: True の場合は実際の削除は行わない
    """
    logger.info("Starting cleanup process")

    files_to_delete = []
    files_to_keep = []
    missing_doi_mapping = []

    for pmc_id, file_path in tqdm(pmc_files.items(), desc="Processing PMC files"):
        # PMC IDに対応するDOIを取得
        doi = pmc_to_doi.get(pmc_id)

        if doi is None:
            # DOIマッピングが見つからない場合
            missing_doi_mapping.append((pmc_id, file_path))
            continue

        # DOIがcorpus_consolidatedに含まれているかチェック
        if doi in used_dois:
            files_to_keep.append((pmc_id, file_path, doi))
        else:
            files_to_delete.append((pmc_id, file_path, doi))

    # 結果の表示
    logger.info("Analysis complete:")
    logger.info(f"  Files to keep: {len(files_to_keep)}")
    logger.info(f"  Files to delete: {len(files_to_delete)}")
    logger.info(f"  Files with missing DOI mapping: {len(missing_doi_mapping)}")

    if missing_doi_mapping:
        logger.warning("Files with missing DOI mapping:")
        for pmc_id, file_path in missing_doi_mapping[:10]:  # 最初の10件のみ表示
            logger.warning(f"  {pmc_id}: {file_path}")
        if len(missing_doi_mapping) > 10:
            logger.warning(f"  ... and {len(missing_doi_mapping) - 10} more")

    if dry_run:
        logger.info("DRY RUN MODE - No files will be deleted")
        if files_to_delete:
            logger.info("Files that would be deleted:")
            for pmc_id, file_path, doi in files_to_delete[:10]:  # 最初の10件のみ表示
                logger.info(f"  {pmc_id} ({doi}): {file_path}")
            if len(files_to_delete) > 10:
                logger.info(f"  ... and {len(files_to_delete) - 10} more")
    else:
        # 実際の削除を実行
        if files_to_delete:
            logger.info(f"Deleting {len(files_to_delete)} unused files...")

            deleted_count = 0
            for pmc_id, file_path, doi in tqdm(files_to_delete, desc="Deleting files"):
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {e}")

            logger.info(f"Successfully deleted {deleted_count} files")

            # 空のディレクトリをクリーンアップ
            cleanup_empty_directories(Path("data/pmc/txt"))
        else:
            logger.info("No files to delete")


def cleanup_empty_directories(base_dir: Path) -> None:
    """
    空のディレクトリを削除

    Args:
        base_dir: ベースディレクトリ
    """
    logger.info("Cleaning up empty directories")

    for pmc_dir in base_dir.glob("PMC*"):
        if pmc_dir.is_dir():
            try:
                # ディレクトリが空の場合は削除
                pmc_dir.rmdir()
                logger.info(f"Removed empty directory: {pmc_dir}")
            except OSError:
                # ディレクトリが空でない場合は無視
                pass


def main():
    parser = argparse.ArgumentParser(description="Clean up unused PMC files")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--backup", action="store_true", help="Create backup before deletion"
    )
    parser.add_argument(
        "--corpus-file",
        default="data/corpus/corpus_consolidated.json",
        help="Path to corpus consolidated JSON file",
    )
    parser.add_argument(
        "--pmc-ids-file",
        default="data/pmc/PMC-ids.csv",
        help="Path to PMC-ids CSV file",
    )
    parser.add_argument(
        "--pmc-txt-dir", default="data/pmc/txt", help="Path to PMC text files directory"
    )
    parser.add_argument(
        "--backup-dir", default="data/pmc/txt_backup", help="Path for backup directory"
    )

    args = parser.parse_args()

    # パスの設定
    corpus_file = Path(args.corpus_file)
    pmc_ids_file = Path(args.pmc_ids_file)
    pmc_txt_dir = Path(args.pmc_txt_dir)
    backup_dir = Path(args.backup_dir)

    # ファイルの存在確認
    if not corpus_file.exists():
        logger.error(f"Corpus consolidated file not found: {corpus_file}")
        return 1

    if not pmc_ids_file.exists():
        logger.error(f"PMC-ids file not found: {pmc_ids_file}")
        return 1

    if not pmc_txt_dir.exists():
        logger.error(f"PMC text directory not found: {pmc_txt_dir}")
        return 1

    try:
        # バックアップの作成（必要な場合）
        if args.backup and not args.dry_run:
            create_backup(pmc_txt_dir, backup_dir)

        # データの読み込み
        used_dois = load_corpus_consolidated(corpus_file)
        pmc_to_doi = load_pmc_doi_mapping(pmc_ids_file)
        pmc_files = find_pmc_files(pmc_txt_dir)

        # クリーンアップの実行
        cleanup_unused_files(used_dois, pmc_to_doi, pmc_files, args.dry_run)

        logger.info("Cleanup completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
