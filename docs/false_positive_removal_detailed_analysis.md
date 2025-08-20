# 偽陽性除去の詳細コードレベル分析

## 概要

Make Data Countコンペティションにおいて、両解法（0.563と0.591）で実装されている偽陽性除去メカニズムを、コードレベルで詳細に分析します。偽陽性除去は、正確でないデータセット引用を特定し、除外することでF1スコアの精度部分を向上させる重要な処理です。

## 偽陽性除去の多層構造

両解法では、以下の**4つの段階**で偽陽性除去を実行しています：

### 1. 基本的な除外パターン（`tidy_extraction`内）

#### 1.1 悪意のあるプレフィックスの除外
```python
bad_ids = [f'{DOI_LINK}{e}' for e in ['10.5061/dryad', '10.5281/zenodo', '10.6073/pasta']]
.filter(~pl.col('dataset_id').is_in(bad_ids))
```

**目的**: 部分的なDOIパターンを除外
- `https://doi.org/10.5061/dryad` （Dryadのルートドメイン）
- `https://doi.org/10.5281/zenodo` （Zenodoのルートドメイン）  
- `https://doi.org/10.6073/pasta` （PASTAのルートドメイン）

これらは完全なデータセットIDではなく、リポジトリのトップレベルを指すため除外されます。

#### 1.2 Figshareキーワード除外
```python
.filter(~pl.col('dataset_id').str.contains('figshare', literal=True))
```

**目的**: Figshareに関連する不完全なIDパターンを除外
- 文字列として`figshare`を含むが、適切なDOI形式でないIDを除外

#### 1.3 自己言及除外（循環参照防止）
```python
.filter(~pl.col('article_id').str.replace('_','/').str.contains(pl.col('dataset_id').str.split(DOI_LINK).list.last().str.escape_regex()))
.filter(~pl.col('dataset_id').str.contains(pl.col('article_id').str.replace('_','/').str.escape_regex()))
```

**詳細メカニズム**:
1. `article_id`の`_`を`/`に変換（例: `10.1002_abc123` → `10.1002/abc123`）
2. `dataset_id`からDOIプレフィックスを除去
3. 相互に含まれる場合は除外

**例**:
- `article_id`: `10.1002_chem.202001412` 
- `dataset_id`: `https://doi.org/10.1002/chem.202001412`
- → 除外（論文が自分自身を参照）

#### 1.4 短いDOI ID除外
```python
.filter(
    pl.when(is_doi_link('dataset_id') &
            (pl.col('dataset_id').str.split('/').list.last().str.len_chars() < 5))
     .then(False)
     .otherwise(True)
)
```

**目的**: 非常に短いDOI IDを除外
- DOIの最後の部分が5文字未満の場合は除外
- 例: `https://doi.org/10.1234/ab` → 除外

### 2. 正規表現パターンレベルでの除外

#### 2.1 パターン自体による偽陽性削減（0.563→0.591での変更）

**削除されたパターン（偽陽性削減）**:
```python
# 0.563で含まれていたが0.591で削除
"PRJEB\d+"  # European Nucleotide Archive (ENA) プロジェクト
"SR[PRX]"   # SRRパターンを含む形式
```

**残されたパターン（0.591）**:
```python
"SR[PX]"    # SRPとSRXのみ（SRRを除外）
```

**理由**: 
- `PRJEB`は文献参照として分類されることが多い
- `SRR`（Sequence Read Archive）は個別リードファイルで、データセット全体を表さない

### 3. LLMによる文脈ベース除外（`llm_validate.py`）

#### 3.1 DOI分類システム
```python
SYS_PROMPT_CLASSIFY_DOI = """
1. Priority Rules (highest → lowest)
1.1 Always classify as A (Data) if:
DOI prefix matches a known data repository:
- Dryad: 10.5061
- Zenodo: 10.5281
- Figshare: 10.6084
[...]

2. Classify as B (Literature) if:
DOI prefix belongs to a publisher (e.g., 10.1038, 10.1007, 10.1126, 10.1016, ...)
[...]
"""
```

#### 3.2 文脈キーワード検出
LLMが以下のキーワードを検出した場合、データリポジトリとして分類：
- `dataset`, `data set`
- `data repository`, `data archive`, `data portal`
- `deposited in`, `uploaded to`, `archived at`
- `available at`, `stored on`, `hosted by`
- `accessible via`, `retrieved from`, `provided by`
- `supplementary dataset`, `supporting dataset`
- `experimental data`, `raw data`
- `public repository`

#### 3.3 logits制約による確実性向上
```python
mclp = MultipleChoiceLogitsProcessor(tokenizer, choices=["A", "B"])
outputs = llm.generate(prompts, vllm.SamplingParams(..., logits_processors=[mclp], ...))
```

**効果**: LLMの出力を`A`（データ）または`B`（文献）に強制的に制限し、曖昧な出力を防止

### 4. ポストフィルタリング（`post_filter.py`）

#### 4.1 出版社プレフィックス除外
```python
PAPER_PREFIXES = [
    "10.5061","10.5281","10.17632","10.1594","10.15468","10.17882","10.7937","10.7910","10.6073",
    "10.3886","10.3334","10.4121","10.5066","10.5067","10.18150","10.25377","10.25387","10.23642","10.24381","10.22033"
]

def is_paper_prefix(col: str = "dataset_id") -> pl.Expr:
    expr = pl.lit(False)
    for p in PAPER_PREFIXES:
        expr = expr | pl.col(col).str.starts_with(f"{DOI_LINK}{p}")
    return expr
```

**注意**: この`PAPER_PREFIXES`リストには**矛盾**があります：
- `10.5061`（Dryad）と`10.5281`（Zenodo）は実際にはデータリポジトリ
- しかし、ここでは「論文プレフィックス」として扱われている
- これは、文脈なしの裸のDOIを除外する戦略的判断

#### 4.2 文脈ベース救済メカニズム
```python
CONTEXT_RE = r"(?i)\b(data(?:set)?|repository|archive|deposited|available|supplementary|raw(?:\s+data)?|uploaded|hosted|stored|accession)\b"

keep_mask = (
    (~is_paper_prefix("dataset_id"))  # not a known paper prefix
    | doi_rows["window"].fill_null("").str.contains(CONTEXT_RE)
)
```

**動作原理**:
1. `PAPER_PREFIXES`に含まれるDOIでも、文脈に適切なキーワードがあれば保持
2. 例：`10.5281/zenodo.1234567`が文脈に「data uploaded to」を含む場合は保持

#### 4.3 アクセッション保護
```python
doi_rows = sub.filter(is_doi_link("dataset_id")).join(win, on=["article_id", "dataset_id"], how="left")
acc_rows = sub.filter(~is_doi_link("dataset_id"))
```

**重要**: アクセッション（非DOI）は`acc_rows`として分離され、ポストフィルタリングを受けない
- `CHEMBL123`, `GSE456`, `PXD789`などは保護される

## 偽陽性除去の効果測定

### 精度向上のメカニズム

1. **段階的フィルタリング**: 4段階のフィルタリングにより、段階的に偽陽性を削減
2. **文脈考慮**: LLMと正規表現の組み合わせで、文脈を考慮した判定
3. **保守的アプローチ**: 疑わしいものは除外し、確実なもののみ残す

### 両解法での共通戦略

1. **多層防御**: 複数の異なる手法を組み合わせ
2. **段階的除外**: 粗い除外から細かい除外へ
3. **文脈重視**: 単純なパターンマッチングだけでなく、周辺文脈を考慮
4. **アクセッション保護**: 信頼性の高いアクセッションIDは保護

## コードレベルでの実装詳細

### Polarフレームワークの活用
```python
df = (
    df.unique(['article_id', 'dataset_id'])
      .filter(~pl.col('article_id').str.replace('_','/').str.contains(...))
      .filter(~pl.col('dataset_id').str.contains(...))
      .filter(~pl.col('dataset_id').str.contains('figshare', literal=True))
      .filter(~pl.col('dataset_id').is_in(bad_ids))
      .filter(pl.when(...).then(False).otherwise(True))
)
```

**特徴**:
- チェーンメソッドによる段階的フィルタリング
- ベクトル化された文字列操作
- 条件分岐による複雑なロジック

### エラーハンドリング
```python
def get_context_window(text: str, substring: str, window: int = 100) -> str:
    idx = text.find(substring)
    if idx == -1:
        raise ValueError
    start = max(idx - window, 0)
    end = min(idx + len(substring) + window, len(text))
    return text[start:end]
```

**堅牢性**: 文脈ウィンドウ構築時の境界条件を適切に処理

## まとめ

偽陽性除去は、両解法で高度に洗練された多層システムとして実装されています。特に：

1. **階層的アプローチ**: 粗い除外→細かい除外→文脈判定→最終フィルタ
2. **矛盾した戦略**: データリポジトリプレフィックスを「論文」として扱い、文脈で救済
3. **保守的判定**: 疑わしいものは除外し、確実なもののみ残す
4. **実装の堅牢性**: Polarフレームワークによる効率的なベクトル化処理

この多層除去システムが、0.563と0.591の両解法で高いF1スコアを実現する基盤となっています。