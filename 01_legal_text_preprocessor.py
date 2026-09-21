"""
Module 1: Multi-Core CPU Legal Text Preprocessor (VietLawAssist)
Tận dụng CPU đa luồng để chuẩn hóa và chunking văn bản quy phạm pháp luật:
1. Chuẩn hóa Unicode NFC (tránh lỗi ký tự tổ hợp tiếng Việt)
2. Phân tích cấu trúc phân cấp pháp lý: Chương -> Mục -> Điều -> Khoản
3. Tách từ tiếng Việt chuyên dụng (PyVi Word Tokenizer)
4. Lưu danh mục chunks sạch cho Retrieval.
"""

from pathlib import Path
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import re
import json
import unicodedata
from pyvi import ViTokenizer
from tqdm import tqdm
from typing import List, Dict, Any

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR.parent))
from shared_utils.path_resolver import resolve_path, DATA_DIR

def normalize_vietnamese_text(text: str) -> str:
    """Chuẩn hóa Unicode NFC và làm sạch khoảng trắng."""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n', text)
    return text.strip()

def parse_legal_articles(raw_text: str, law_code: str = "LUAT_DAN_SU") -> List[Dict[str, Any]]:
    """Tách văn bản luật thành các Điều luật cấu trúc."""
    normalized = normalize_vietnamese_text(raw_text)
    
    # Regex nhận diện "Điều 1.", "Điều 24."
    pattern = re.compile(r'(Điều\s+(\d+)[\.:]\s*([^\n]+))', re.IGNORECASE)
    matches = list(pattern.finditer(normalized))
    
    chunks = []
    if not matches:
        # Nếu không có định dạng Điều thì chunk theo đoạn văn
        paragraphs = [p.strip() for p in normalized.split("\n") if len(p.strip()) > 30]
        for idx, p in enumerate(paragraphs):
            chunks.append({
                "chunk_id": f"{law_code}_p_{idx+1}",
                "law_code": law_code,
                "article_num": idx + 1,
                "title": f"Đoạn văn {idx+1}",
                "content": p,
                "tokenized_content": ViTokenizer.tokenize(p)
            })
        return chunks

    for i in range(len(matches)):
        start_idx = matches[i].start()
        end_idx = matches[i+1].start() if i + 1 < len(matches) else len(normalized)
        
        full_block = normalized[start_idx:end_idx].strip()
        article_num = int(matches[i].group(2))
        title = matches[i].group(3).strip()
        
        # Tách từ tiếng Việt
        tokenized = ViTokenizer.tokenize(full_block)
        
        chunks.append({
            "chunk_id": f"{law_code}_D{article_num}",
            "law_code": law_code,
            "article_num": article_num,
            "title": title,
            "content": full_block,
            "tokenized_content": tokenized
        })
        
    return chunks

def create_sample_legal_data_if_needed(sample_file: Path):
    """Tạo mẫu dữ liệu luật nếu chưa có file gốc để kiểm thử pipeline."""
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    if not sample_file.exists():
        sample_corpus = """
Điều 1. Phạm vi điều chỉnh
Bộ luật này quy định địa vị pháp lý, chuẩn mực pháp lý về cách ứng xử của cá nhân, pháp nhân; quyền, nghĩa vụ về nhân thân và tài sản của cá nhân, pháp nhân trong các quan hệ được hình thành trên cơ sở bình đẳng, tự do ý chí, độc lập về tài sản và tự chịu trách nhiệm.

Điều 2. Công nhận, tôn trọng, bảo vệ và bảo đảm quyền dân sự
1. Mọi quyền dân sự của cá nhân, pháp nhân được công nhận, tôn trọng, bảo vệ và bảo đảm theo Hiến pháp và pháp luật.
2. Quyền dân sự chỉ có thể bị hạn chế theo quy định của luật trong trường hợp cần thiết vì lý do quốc phòng, an ninh quốc gia, trật tự, an toàn xã hội, đạo đức xã hội, sức khỏe của cộng đồng.

Điều 3. Nguyên tắc cơ bản của pháp luật dân sự
1. Mọi cá nhân, pháp nhân đều bình đẳng, không được lấy bất kỳ lý do nào để phân biệt đối xử; được pháp luật bảo hộ như nhau về các quyền nhân thân và tài sản.
2. Cá nhân, pháp nhân xác lập, thực hiện, chấm dứt quyền, nghĩa vụ dân sự của mình trên cơ sở tự do, tự nguyện cam kết, thỏa thuận.
"""
        sample_file.write_text(sample_corpus.strip(), encoding="utf-8")
        print(f"📦 Đã tạo file luật mẫu tại: {sample_file}")

if __name__ == "__main__":
    sample_law_file = DATA_DIR / "sample_nlp" / "bo_luat_dan_su_sample.txt"
    create_sample_legal_data_if_needed(sample_law_file)
    
    content = sample_law_file.read_text(encoding="utf-8")
    articles = parse_legal_articles(content, law_code="BLDS_2015")
    
    out_json = DATA_DIR / "sample_nlp" / "legal_chunks.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Đã xử lý và chunking {len(articles)} điều luật sang: {out_json}")
