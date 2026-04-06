# config.py
import os
from pathlib import Path

# 路径配置
BASE_DIR = Path(__file__).parent
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
UPLOAD_DIR = BASE_DIR / "uploads"

# 模型配置
EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"
LLM_MODEL = "Qwen/Qwen-1_8B-Chat"

# 文档处理配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# 检索配置
TOP_K_RESULTS = 3
SIMILARITY_THRESHOLD = 0.7

# 创建必要目录
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)