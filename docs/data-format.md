# Data Citation Corpus データフォーマット仕様

## 概要
Make Data Count プロジェクトのData Citation Corpusデータセット（v4.1）の詳細仕様について説明します。

## データセット構成
- **ファイル名パターン**: `2025-08-15-data-citation-corpus-{番号}-v4.1.csv`
- **分割ファイル数**: 11個（01～11）
- **形式**: CSV（カンマ区切り）
- **エンコーディング**: UTF-8

## カラム構成

### 基本識別情報
| カラム名 | 型 | 必須 | 説明 | 例 |
|---------|---|-----|-----|---|
| `id` | UUID | ✓ | データ引用の一意識別子 | `00000025-982e-40b8-bdea-29f2eb3742a6` |
| `created` | ISO 8601 DateTime | ✓ | レコード作成日時 | `2025-07-25T22:45:39.844+00:00` |
| `updated` | ISO 8601 DateTime | ✓ | レコード更新日時 | `2025-07-25T22:45:39.844+00:00` |

### データリポジトリ情報
| カラム名 | 型 | 必須 | 説明 | 例 |
|---------|---|-----|-----|---|
| `repository` | String | - | データが格納されているリポジトリ名 | `The Protein Data Bank`、`European Nucleotide Archive` |
| `dataset` | String | - | データセット識別子またはアクセッション番号 | `2CHS`、`JX090164`、`SRR7236385` |

### 出版物情報
| カラム名 | 型 | 必須 | 説明 | 例 |
|---------|---|-----|-----|---|
| `publisher` | String | - | 出版社名 | `Proceedings of the National Academy of Sciences`、`Public Library of Science (PLoS)` |
| `journal` | String | - | 雑誌名 | `Proceedings of the National Academy of Sciences`、`PLOS ONE` |
| `title` | String | - | 論文タイトル（現在は空欄） | （空文字列） |
| `publication` | DOI URL | - | 論文のDOI URL | `https://doi.org/10.1073/pnas.102679799` |
| `publishedDate` | ISO 8601 DateTime | - | 論文出版日 | `2002-05-07T00:00:00+00:00` |

### メタデータ
| カラム名 | 型 | 必須 | 説明 | 例 |
|---------|---|-----|-----|---|
| `source` | String | - | データソース | `eupmc`、`czi`、`datacite` |
| `subjects` | String (semicolon-separated) | - | 研究分野（セミコロン区切り） | `chemical sciences; basic medicine; physical sciences; biological sciences` |

### 所属・資金情報
| カラム名 | 型 | 必須 | 説明 | 例 |
|---------|---|-----|-----|---|
| `affiliations` | String | - | 所属機関名 | `National Institute for Environmental Studies (NIES)` |
| `affiliationsROR` | String | - | 所属機関のROR ID | `None None` |
| `funders` | String | - | 資金提供機関名 | （多くは空欄） |
| `fundersROR` | String | - | 資金提供機関のROR ID | （多くは空欄） |

## データ例

### Protein Data Bank の例
```csv
00000025-982e-40b8-bdea-29f2eb3742a6,2025-07-25T22:45:39.844+00:00,2025-07-25T22:45:39.844+00:00,The Protein Data Bank,Proceedings of the National Academy of Sciences,Proceedings of the National Academy of Sciences,,2CHS,https://doi.org/10.1073/pnas.102679799,2002-05-07T00:00:00+00:00,eupmc,chemical sciences; basic medicine; physical sciences; biological sciences,,,,
```

### European Nucleotide Archive の例
```csv
0000007b-294c-4568-9ad7-2212d1e28bb4,2025-07-24T22:12:43.372+00:00,2025-07-24T22:12:43.372+00:00,European Nucleotide Archive,MDPI AG,International Journal of Molecular Sciences,,JX090164,https://doi.org/10.3390/ijms130912046,2012-09-21T00:00:00+00:00,eupmc,basic medicine; biological sciences,,,,
```

### Cambridge Crystallographic Data Centre の例
```csv
000006e4-ee4d-4240-a518-7a7e43bd9ed7,2023-06-07T20:40:28.762+00:00,2023-06-07T20:40:28.762+00:00,Cambridge Crystallographic Data Centre,American Chemical Society (ACS),Inorganic Chemistry,CCDC 983926: Experimental Crystal Structure Determination,https://doi.org/10.5517/cc120vkh,https://doi.org/10.1021/ic502805h,2015-01-23T00:00:00+00:00,datacite,,,,,
```

## データソース分類

### `source` フィールドの値
- **`eupmc`**: Europe PMC由来のデータ
- **`czi`**: Chan Zuckerberg Initiative由来のデータ  
- **`datacite`**: DataCite由来のデータ

## 主要なリポジトリ

### 生物学系
- **The Protein Data Bank**: タンパク質構造データ（PDB ID）
- **European Nucleotide Archive**: DNA/RNA配列データ（アクセッション番号）
- **UniProt**: タンパク質配列・機能データ（UniProt ID）
- **Gene Expression Omnibus (GEO)**: 遺伝子発現データ（GSE番号）
- **Ensembl**: ゲノムデータ（ENST番号など）
- **NCBI Reference Sequence Database**: 参照配列（NM_番号など）
- **dbSNP Reference SNP**: SNPデータ（rs番号）

### 化学系
- **Cambridge Crystallographic Data Centre**: 結晶構造データ（CCDC番号）

### その他
- **Earth System Grid Federation**: 気候データ

## データ利用時の注意点

1. **空欄の扱い**: 多くのフィールドが空文字列の場合があります
2. **日付形式**: ISO 8601形式（UTC）
3. **セミコロン区切り**: `subjects`フィールドは複数値をセミコロンで区切り
4. **識別子の多様性**: `dataset`フィールドには様々な形式の識別子が含まれます
5. **URL形式**: DOIはhttps://doi.org/形式で統一

## PMC-ids.csv ファイル仕様

### 概要
PMC（PubMed Central）の論文IDとメタデータを含むマスターファイル

### ファイル情報
- **ファイル名**: `PMC-ids.csv`
- **場所**: `data/papers/pmc/PMC-ids.csv`
- **サイズ**: 約1GB
- **総レコード数**: 11,247,558行（ヘッダー含む）
- **エンコーディング**: UTF-8

### カラム構成（12カラム）

| 位置 | カラム名 | 型 | 説明 | 例 |
|-----|---------|---|-----|---|
| 1 | `Journal Title` | String | 雑誌タイトル | `Breast Cancer Res`, `Proc Natl Acad Sci U S A` |
| 2 | `ISSN` | String | 印刷版ISSN | `1465-5411`, `0027-8424` |
| 3 | `eISSN` | String | 電子版ISSN | `1465-542X`, `1091-6490` |
| 4 | `Year` | Integer | 出版年 | `2000`, `2025` |
| 5 | `Volume` | String | 巻数 | `3`, `98`, `2025` |
| 6 | `Issue` | String | 号数 | `1`, `2`, 空文字列 |
| 7 | `Page` | String | ページ番号 | `55`, `26`, `6662321` |
| 8 | `DOI` | String | Digital Object Identifier | `10.1186/bcr271`, `10.1073/pnas.011393898` |
| 9 | `PMCID` | String | PubMed Central ID | `PMC13900`, `PMC12364609` |
| 10 | `PMID` | String | PubMed ID | `11250746`, `40751508` |
| 11 | `Manuscript Id` | String | 原稿ID（多くは空） | `NIHMS1965845`, 空文字列 |
| 12 | `Release Date` | String | リリース状態 | `live` |

### PMCID形式
- **パターン**: `PMC` + 数字
- **範囲**: PMC13900 ～ PMC12364609
- **桁数**: 5～8桁の数字部分

### データ品質
- **標準フィールド数**: 12（全体の99.99%）
- **異常レコード**: 
  - 13フィールド: 1,145件
  - 14フィールド: 279件
  - 15フィールド以上: 30件

### 対応するテキストファイル
PMCIDに対応するテキストファイルが以下のディレクトリに格納：
```
data/papers/pmc/oa_comm_txt.PMC00Xxxxxxx.baseline.2025-06-17/PMC00Xxxxxxx/
```

### データ例
```csv
Journal Title,ISSN,eISSN,Year,Volume,Issue,Page,DOI,PMCID,PMID,Manuscript Id,Release Date
Breast Cancer Res,1465-5411,1465-542X,2000,3,1,55,10.1186/bcr271,PMC13900,11250746,,live
Proc Natl Acad Sci U S A,0027-8424,1091-6490,2000,98,1,26,10.1073/pnas.011393898,PMC14538,11134511,,live
```

## ファイル統計
- 総ファイル数: 11個
- 推定総レコード数: 数万件以上
- 各ファイルサイズ: 大容量（具体的なサイズは要確認）