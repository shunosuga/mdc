import re
import nltk
from transformers import AutoTokenizer
from typing import List, Dict, Tuple
import logging

# NLTK のsentence tokenizer をダウンロード（初回のみ）
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class PaperChunker:
    def __init__(self, model_name: str = "google/modern-bert-base", max_tokens: int = 1024):
        """
        Args:
            model_name: 使用するBERTモデル名
            max_tokens: チャンクあたりの最大トークン数
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens
        
        # セクション名の正規化マッピング
        self.section_mapping = {
            'abstract': 'ABSTRACT',
            'introduction': 'INTRODUCTION', 
            'background': 'INTRODUCTION',
            'methods': 'METHODS',
            'methodology': 'METHODS',
            'materials and methods': 'METHODS',
            'experimental': 'METHODS',
            'results': 'RESULTS',
            'findings': 'RESULTS',
            'discussion': 'DISCUSSION',
            'conclusion': 'CONCLUSION',
            'conclusions': 'CONCLUSION',
            'references': 'REFERENCES',
            'bibliography': 'REFERENCES',
            'acknowledgments': 'ACKNOWLEDGMENTS',
            'acknowledgements': 'ACKNOWLEDGMENTS',
            'funding': 'FUNDING',
            'data availability': 'DATA_AVAILABILITY',
            'supplementary': 'SUPPLEMENTARY',
            'appendix': 'APPENDIX'
        }
        
    def detect_sections(self, text: str) -> List[Tuple[int, str, str]]:
        """
        テキストからセクション境界を検出
        
        Returns:
            List of (start_position, section_name, normalized_section)
        """
        sections = []
        
        # セクションヘッダーのパターン
        patterns = [
            r'^#+\s*(.+?)$',  # Markdown形式 (## Introduction)
            r'^(\d+\.?\s*.+?)$',  # 番号付き (1. Introduction)
            r'^([A-Z][A-Z\s&-]+)$',  # 大文字のみ (INTRODUCTION)
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)$',  # タイトルケース (Introduction)
        ]
        
        lines = text.split('\n')
        current_pos = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                current_pos += len(lines[i]) + 1
                continue
                
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    section_text = match.group(1).strip()
                    section_key = self._normalize_section_name(section_text)
                    
                    if section_key:
                        sections.append((current_pos, section_text, section_key))
                        break
            
            current_pos += len(lines[i]) + 1
            
        return sections
    
    def _normalize_section_name(self, section_text: str) -> str:
        """セクション名を正規化"""
        # 番号や記号を除去
        clean_text = re.sub(r'^\d+\.?\s*', '', section_text)
        clean_text = re.sub(r'[^\w\s]', '', clean_text)
        clean_text = clean_text.lower().strip()
        
        # 部分マッチで検索
        for key, normalized in self.section_mapping.items():
            if key in clean_text or clean_text in key:
                return normalized
                
        # 明らかなセクションっぽいものは汎用名で
        if len(clean_text.split()) <= 3 and clean_text.isupper():
            return clean_text.upper().replace(' ', '_')
            
        return None
    
    def split_into_sentences(self, text: str) -> List[str]:
        """文単位に分割"""
        try:
            sentences = nltk.sent_tokenize(text)
        except:
            # フォールバック: 簡単な文分割
            sentences = re.split(r'[.!?]+\s+', text)
            
        return [s.strip() for s in sentences if s.strip()]
    
    def count_tokens(self, text: str) -> int:
        """トークン数をカウント"""
        return len(self.tokenizer.encode(text, add_special_tokens=True))
    
    def chunk_paper(self, text: str) -> List[Dict]:
        """
        論文をチャンク化
        
        Returns:
            List of dictionaries with keys: 'text', 'section', 'chunk_id', 'token_count'
        """
        # セクション検出
        sections = self.detect_sections(text)
        
        chunks = []
        current_section = "UNKNOWN"
        chunk_id = 0
        
        if not sections:
            # セクションが検出されない場合の処理
            return self._chunk_without_sections(text)
        
        # セクションごとに処理
        for i, (section_start, original_name, normalized_name) in enumerate(sections):
            current_section = normalized_name
            
            # セクションの終了位置を決定
            if i + 1 < len(sections):
                section_end = sections[i + 1][0]
            else:
                section_end = len(text)
            
            section_text = text[section_start:section_end].strip()
            
            # セクション内をチャンク化
            section_chunks = self._chunk_section(
                section_text, current_section, chunk_id
            )
            
            chunks.extend(section_chunks)
            chunk_id += len(section_chunks)
        
        return chunks
    
    def _chunk_section(self, section_text: str, section_name: str, start_chunk_id: int) -> List[Dict]:
        """セクション内をチャンク化"""
        sentences = self.split_into_sentences(section_text)
        chunks = []
        current_chunk = f"[{section_name}] "
        section_token_overhead = self.count_tokens(f"[{section_name}] ")
        
        chunk_id = start_chunk_id
        
        for sentence in sentences:
            # 文を追加した場合のトークン数を計算
            test_chunk = current_chunk + sentence + " "
            token_count = self.count_tokens(test_chunk)
            
            if token_count <= self.max_tokens:
                # まだ余裕がある場合
                current_chunk = test_chunk
            else:
                # 制限を超える場合、現在のチャンクを保存
                if len(current_chunk.strip()) > len(f"[{section_name}]"):
                    chunks.append({
                        'text': current_chunk.strip(),
                        'section': section_name,
                        'chunk_id': f"chunk_{chunk_id:04d}",
                        'token_count': self.count_tokens(current_chunk.strip())
                    })
                    chunk_id += 1
                
                # 新しいチャンクを開始
                current_chunk = f"[{section_name}] {sentence} "
                
                # 単一文でも制限を超える場合の処理
                if self.count_tokens(current_chunk) > self.max_tokens:
                    # 文を強制的に分割
                    word_chunks = self._split_long_sentence(sentence, section_name)
                    for word_chunk in word_chunks:
                        chunks.append({
                            'text': f"[{section_name}] {word_chunk}",
                            'section': section_name,
                            'chunk_id': f"chunk_{chunk_id:04d}",
                            'token_count': self.count_tokens(f"[{section_name}] {word_chunk}")
                        })
                        chunk_id += 1
                    current_chunk = f"[{section_name}] "
        
        # 最後のチャンクを保存
        if len(current_chunk.strip()) > len(f"[{section_name}]"):
            chunks.append({
                'text': current_chunk.strip(),
                'section': section_name,
                'chunk_id': f"chunk_{chunk_id:04d}",
                'token_count': self.count_tokens(current_chunk.strip())
            })
        
        return chunks
    
    def _split_long_sentence(self, sentence: str, section_name: str) -> List[str]:
        """長すぎる文を単語レベルで分割"""
        words = sentence.split()
        chunks = []
        current_chunk = ""
        section_prefix = f"[{section_name}] "
        
        for word in words:
            test_chunk = current_chunk + " " + word if current_chunk else word
            test_text = section_prefix + test_chunk
            
            if self.count_tokens(test_text) <= self.max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = word
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _chunk_without_sections(self, text: str) -> List[Dict]:
        """セクション情報がない場合の処理"""
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = "[UNKNOWN] "
        chunk_id = 0
        
        for sentence in sentences:
            test_chunk = current_chunk + sentence + " "
            
            if self.count_tokens(test_chunk) <= self.max_tokens:
                current_chunk = test_chunk
            else:
                if len(current_chunk.strip()) > len("[UNKNOWN]"):
                    chunks.append({
                        'text': current_chunk.strip(),
                        'section': 'UNKNOWN',
                        'chunk_id': f"chunk_{chunk_id:04d}",
                        'token_count': self.count_tokens(current_chunk.strip())
                    })
                    chunk_id += 1
                
                current_chunk = f"[UNKNOWN] {sentence} "
        
        if len(current_chunk.strip()) > len("[UNKNOWN]"):
            chunks.append({
                'text': current_chunk.strip(),
                'section': 'UNKNOWN',
                'chunk_id': f"chunk_{chunk_id:04d}",
                'token_count': self.count_tokens(current_chunk.strip())
            })
        
        return chunks

# 使用例
def main():
    # チャンカーを初期化
    chunker = PaperChunker(max_tokens=1024)
    
    # サンプルテキスト
    sample_text = """
    Abstract
    
    This study investigates the effectiveness of deep learning models in medical diagnosis.
    We analyzed 10,000 patient records and achieved 95% accuracy.
    
    1. Introduction
    
    Medical diagnosis has been revolutionized by artificial intelligence.
    Previous studies have shown promising results in various domains.
    However, there are still challenges in clinical implementation.
    
    2. Methods
    
    We collected data from three hospitals over two years.
    The dataset includes patient demographics, symptoms, and diagnoses.
    Figure 1 shows the data collection process.
    
    3. Results
    
    Our model achieved state-of-the-art performance.
    Table 1 presents the detailed results across different conditions.
    The precision was 0.94 and recall was 0.96.
    """
    
    # チャンク化実行
    chunks = chunker.chunk_paper(sample_text)
    
    # 結果表示
    for chunk in chunks:
        print(f"ID: {chunk['chunk_id']}")
        print(f"Section: {chunk['section']}")
        print(f"Tokens: {chunk['token_count']}")
        print(f"Text: {chunk['text'][:100]}...")
        print("-" * 50)

if __name__ == "__main__":
    main()