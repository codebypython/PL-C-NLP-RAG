"""
Module 5: High-Performance Production FastAPI Server (VietLawAssist)
Cung cấp RESTful API và Server-Sent Events (SSE) cho hệ thống trợ lý pháp luật:
1. POST /api/v1/search: Truy xuất Hybrid RRF siêu tốc (<15ms)
2. POST /api/v1/chat: Tư vấn pháp lý có cấu trúc kèm báo cáo chống ảo giác
3. GET  /api/v1/chat/stream: Stream token theo thời gian thực (SSE)
4. GET  /api/v1/health: Giám sát tài nguyên hệ thống (RAM, VRAM GPU, Cache Hit Rate)
5. Static Mounting: Tự động phục vụ Web App giao diện luật sư tại root URL (/).
6. Tuyệt đối không hardcode đường dẫn.
"""

from pathlib import Path
import os
import sys
import json
import time
from typing import Optional, List, Dict, Any

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
    from shared_utils.gpu_health_monitor import get_gpu_memory_stats
except Exception:
    DATA_DIR = CURRENT_DIR.parent / "data"
    def get_gpu_memory_stats():
        return {"cuda_available": False, "device_name": "N/A"}

import importlib
engine_mod = importlib.import_module("04_production_rag_engine")
ProductionLegalRAGEngine = engine_mod.ProductionLegalRAGEngine

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field

# Khởi tạo App & Core Engine
app = FastAPI(
    title="VietLawAssist Production API",
    description="Enterprise Legal AI Assistant with Hybrid RRF & Anti-Hallucination Guardrails",
    version="2.0.0"
)

# CORS Policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core RAG Engine Singleton
rag_engine = ProductionLegalRAGEngine()
SERVER_START_TIME = time.time()

# Request/Response Schemas
class SearchRequest(BaseModel):
    query: str = Field(..., example="Bồi thường tai nạn giao thông xe máy")
    domain: Optional[str] = Field(None, example="dan_su")
    top_k: int = Field(3, ge=1, le=10)

class ChatRequest(BaseModel):
    query: str = Field(..., example="Người lao động bị đuổi việc trái luật được đền bù gì?")
    domain: Optional[str] = Field(None, example="lao_dong")
    top_k: int = Field(2, ge=1, le=5)

# API Endpoints
@app.get("/api/v1/health")
def health_check():
    """Kiểm tra sức khỏe hệ thống và vi độ trễ hoạt động."""
    gpu_info = get_gpu_memory_stats()
    retriever = rag_engine.retriever
    cache_total = retriever.cache_hits + retriever.cache_misses
    cache_hit_rate = round((retriever.cache_hits / cache_total) * 100, 1) if cache_total > 0 else 0.0

    return {
        "status": "HEALTHY",
        "service": "VietLawAssist Production RAG",
        "uptime_seconds": round(time.time() - SERVER_START_TIME, 1),
        "indexed_articles": len(retriever.chunks),
        "domains_supported": ["dan_su", "lao_dong", "doanh_nghiep", "hinh_su"],
        "cache_stats": {
            "capacity": retriever.cache_capacity,
            "cached_entries": len(retriever._cache),
            "hits": retriever.cache_hits,
            "misses": retriever.cache_misses,
            "hit_rate_pct": cache_hit_rate
        },
        "gpu_telemetry": gpu_info
    }

@app.post("/api/v1/search")
def search_legal_documents(req: SearchRequest):
    """Truy xuất điều luật bằng thuật toán Hybrid RRF (BM25 + FAISS)."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Câu truy vấn không được để trống.")
    
    result = rag_engine.retriever.search_hybrid_rrf(
        query=req.query,
        top_k=req.top_k,
        domain=req.domain
    )
    return result

@app.post("/api/v1/chat")
def chat_legal_advice(req: ChatRequest):
    """Tư vấn pháp lý có cấu trúc kèm chứng thực chống ảo giác."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")
    
    result = rag_engine.generate_response(
        user_query=req.query,
        domain=req.domain,
        top_k=req.top_k
    )
    return result

@app.get("/api/v1/chat/stream")
async def chat_stream_sse(
    query: str = Query(..., description="Câu hỏi pháp lý từ người dùng"),
    domain: Optional[str] = Query(None, description="Lĩnh vực pháp lý"),
    top_k: int = Query(2, ge=1, le=5)
):
    """Stream token phản hồi theo thời gian thực chuẩn SSE."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")

    def event_generator():
        for chunk in rag_engine.stream_response(user_query=query, domain=domain, top_k=top_k):
            yield {
                "event": chunk["event"],
                "data": json.dumps(chunk["data"], ensure_ascii=False)
            }

    return EventSourceResponse(event_generator())

# Phục vụ Static Web App
WEB_APP_DIR = CURRENT_DIR / "web_app"
if WEB_APP_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_APP_DIR)), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(str(WEB_APP_DIR / "index.html"))

if __name__ == "__main__":
    import uvicorn
    print("🚀 [VietLawAssist API] Đang khởi chạy máy chủ tại http://127.0.0.1:8000 ...")
    uvicorn.run("05_api_service:app", host="127.0.0.1", port=8000, reload=False)
