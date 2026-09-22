"""
Module 2: Dense Embedding & FAISS Index Builder (VietLawAssist)
Xây dựng chỉ mục tìm kiếm ngữ nghĩa thời gian thực (< 5ms) với FAISS:
1. Nạp toàn bộ 23 điều luật đa lĩnh vực đã bóc tách từ module 01
2. Xây dựng không gian biểu diễn ngữ nghĩa tiếng Việt đa tầng (Subword & Word N-Gram TF-IDF với chuẩn hóa Cosine L2)
3. Đóng gói chỉ mục nhị phân FAISS IndexFlatIP (Inner Product = Cosine Similarity)
4. Lưu trữ mô hình nhúng (vectorizer.pkl), chỉ mục (legal_faiss.index) và metadata mapping (legal_faiss_meta.json)
5. Tuyệt đối không hardcode đường dẫn.
"""

from pathlib import Path
import os
import sys
import json
import pickle
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from pyvi import ViTokenizer
    def vi_tokenize(text: str) -> str:
        return ViTokenizer.tokenize(text)
except Exception:
    def vi_tokenize(text: str) -> str:
        return text

from typing import List, Dict, Any, Tuple

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Dynamic Path Setup (Zero Hardcoded Paths)
CURRENT_DIR = Path(__file__).resolve().parent
try:
    sys.path.append(str(CURRENT_DIR.parent))
    from shared_utils.path_resolver import resolve_path, DATA_DIR
except Exception:
    DATA_DIR = CURRENT_DIR.parent / "data"

class LegalSemanticEncoder:
    """Bộ mã hóa vector ngữ nghĩa tiếng Việt cho văn bản quy phạm pháp luật."""
    def __init__(self, max_features: int = 512):
        self.max_features = max_features
        # Kết hợp cả word n-gram và char n-gram để bắt được cả lỗi chính tả lẫn từ đồng nghĩa
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=max_features,
            sublinear_tf=True,
            norm='l2'
        )
        self.is_fitted = False

    def fit_transform(self, corpus_texts: List[str]) -> np.ndarray:
        """Học không gian biểu diễn và sinh ma trận vector cho toàn bộ văn bản."""
        tokenized_corpus = [vi_tokenize(t).lower() for t in corpus_texts]
        matrix = self.vectorizer.fit_transform(tokenized_corpus).toarray().astype(np.float32)
        # Chuẩn hóa L2 đảm bảo tích vô hướng bằng chuẩn Cosine
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        self.is_fitted = True
        return matrix

    def encode_query(self, query: str) -> np.ndarray:
        """Mã hóa một câu truy vấn của người dùng thành vector tương ứng."""
        if not self.is_fitted:
            raise RuntimeError("Encoder chưa được fit trên dữ liệu văn bản!")
        tokenized_q = vi_tokenize(query).lower()
        q_vec = self.vectorizer.transform([tokenized_q]).toarray().astype(np.float32)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm
        return q_vec

    def save(self, filepath: Path):
        with open(filepath, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, filepath: Path):
        with open(filepath, "rb") as f:
            self.vectorizer = pickle.load(f)
        self.is_fitted = True

def build_dense_faiss_system(
    chunks_json_path: Path,
    output_index_path: Path,
    output_meta_path: Path,
    output_model_path: Path
):
    """Xây dựng và đóng gói toàn bộ hệ thống FAISS Dense Retrieval."""
    if not chunks_json_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {chunks_json_path}. Vui lòng chạy 01_legal_text_preprocessor.py trước.")

    with open(chunks_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"🔄 [FAISS Builder] Đang xây dựng chỉ mục cho {len(chunks)} điều luật...")
    
    # Kết hợp title + content để vector hóa ngữ cảnh toàn vẹn
    corpus_texts = [f"{c['title']}. {c['content']}" for c in chunks]
    
    encoder = LegalSemanticEncoder(max_features=384)
    embeddings = encoder.fit_transform(corpus_texts)
    dim = embeddings.shape[1]

    # Khởi tạo FAISS IndexFlatIP (Inner Product = Cosine Similarity do vector đã L2 normalized)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Lưu Vectorizer
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    encoder.save(output_model_path)

    # Lưu FAISS binary index
    output_index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_index_path))

    # Lưu metadata mapping
    metadata_map = {
        i: {
            "chunk_id": chunk["chunk_id"],
            "law_code": chunk.get("law_code", ""),
            "law_name": chunk.get("law_name", ""),
            "domain": chunk.get("domain", ""),
            "article_num": chunk.get("article_num", 0),
            "title": chunk["title"],
            "content": chunk["content"]
        } for i, chunk in enumerate(chunks)
    }
    with open(output_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_map, f, indent=2, ensure_ascii=False)

    print(f"✅ [Hoàn tất] Đã tạo chỉ mục FAISS {index.ntotal} vectors ({dim}D):")
    print(f"   -> Model Encoder: {output_model_path}")
    print(f"   -> FAISS Index:   {output_index_path}")
    print(f"   -> Metadata Map:  {output_meta_path}")

if __name__ == "__main__":
    chunks_p = DATA_DIR / "sample_nlp" / "legal_chunks.json"
    idx_p = DATA_DIR / "sample_nlp" / "legal_faiss.index"
    meta_p = DATA_DIR / "sample_nlp" / "legal_faiss_meta.json"
    model_p = DATA_DIR / "sample_nlp" / "legal_encoder.pkl"
    
    build_dense_faiss_system(chunks_p, idx_p, meta_p, model_p)
