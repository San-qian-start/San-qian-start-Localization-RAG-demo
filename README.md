# 📚 内部文档智能问答助手

基于 **Streamlit + LangChain + Chroma + Ollama** 构建的企业级内部文档问答系统。支持上传 PDF、Word、TXT 文档，自动构建向量数据库，并通过本地大模型（如 Qwen2.5:3b）实现基于文档内容的智能问答。

---

## ✨ 功能特点

- ✅ 支持多种文档格式：`.pdf`, `.docx`, `.txt`
- ✅ 中文语义检索（使用 `bge-large-zh-v1.5` 嵌入模型）
- ✅ 本地化部署，无需联网，保护数据安全
- ✅ 基于 **Ollama** 运行大模型（默认 `qwen2.5:3b`）
- ✅ 自动构建与增量更新向量数据库（Chroma）
- ✅ 支持文档来源溯源，显示引用来源
- ✅ 会话历史记录与管理
- ✅ 数据库重建、清空文档等高级管理功能

---

## 🧱 技术栈

| 组件 | 技术 |
|------|------|
| 前端 & 交互 | Streamlit |
| 文档加载 | LangChain (`PyPDFLoader`, `Docx2txtLoader`, `TextLoader`) |
| 文本分割 | `RecursiveCharacterTextSplitter` |
| 嵌入模型 | `BAAI/bge-large-zh-v1.5`（本地加载） |
| 向量数据库 | Chroma（本地持久化） |
| 大语言模型 | Ollama + Qwen2.5:3b |
| 开发语言 | Python 3.8+ |

---

## 📦 安装与配置
- 自主安装词嵌入模型、大语言。将代码中模型位置替换为本地模型
- 终端输入：streamlit run app.py 即可运行


### 1. 克隆或下载项目

```bash
git clone <your-repo-url>
cd <project-folder>
