import os

os.environ["STREAMLIT_TELEMETRY_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 防止tokenizer警告

import streamlit as st
import tempfile
from typing import List, Dict, Optional
import hashlib
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# LangChain 相关导入 - 使用新的导入方式避免警告
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# 设置页面配置
st.set_page_config(
    page_title="内部文档问答助手",
    page_icon="📚",
    layout="wide"
)

# 初始化session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'history' not in st.session_state:
    st.session_state.history = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()


class DocumentProcessor:
    """文档处理器"""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "．", "，", "、", " ", ""]
        )

        # 您的本地模型路径
        local_model_path = "E:\\my_ai_model\\model_dir\\BAAI\\bge-large-zh-v1___5"

        try:
            # 检查本地模型路径是否存在
            if os.path.exists(local_model_path):
                st.success(f"✅ 找到本地模型: bge-large-zh-v1")

                # 加载本地模型 - 移除冲突参数
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=local_model_path,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={
                        'normalize_embeddings': True
                    }
                )

                # 测试模型
                try:
                    test_embedding = self.embeddings.embed_query("测试")
                    st.info(f"✅ 模型加载成功，向量维度: {len(test_embedding)}")
                except Exception as test_e:
                    st.warning(f"模型测试有小问题，但可能仍可使用: {str(test_e)}")

            else:
                st.error(f"❌ 本地模型路径不存在: {local_model_path}")
                st.info("尝试使用备用模型...")
                from langchain_community.embeddings import FakeEmbeddings
                self.embeddings = FakeEmbeddings(size=1024)
                st.warning("⚠️ 使用假嵌入模式（功能受限）")

        except Exception as e:
            st.error(f"❌ 嵌入模型加载失败: {str(e)}")
            try:
                from langchain_community.embeddings import FakeEmbeddings
                self.embeddings = FakeEmbeddings(size=1024)
                st.warning("⚠️ 使用假嵌入模式（功能受限）")
            except:
                st.stop()

    def _get_file_hash(self, file):
        """计算文件哈希值"""
        file.seek(0)
        file_hash = hashlib.md5(file.getvalue()).hexdigest()
        file.seek(0)
        return file_hash

    def process_uploaded_file(self, uploaded_file):
        """处理上传的文件"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # 根据文件类型选择加载器
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()

            if file_extension == '.pdf':
                loader = PyPDFLoader(tmp_file_path)
            elif file_extension == '.docx':
                loader = Docx2txtLoader(tmp_file_path)
            elif file_extension == '.txt':
                loader = TextLoader(tmp_file_path, encoding='utf-8')
            else:
                from langchain_community.document_loaders import UnstructuredFileLoader
                loader = UnstructuredFileLoader(tmp_file_path)

            # 加载文档
            documents = loader.load()

            # 添加元数据
            for doc in documents:
                doc.metadata['source'] = uploaded_file.name
                doc.metadata['upload_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                doc.metadata['file_hash'] = self._get_file_hash(uploaded_file)

            # 分割文档
            splits = self.text_splitter.split_documents(documents)

            return splits

        except Exception as e:
            st.error(f"处理文件 {uploaded_file.name} 时出错: {str(e)}")
            return []  # 返回空列表而不是抛出异常
        finally:
            # 清理临时文件
            try:
                os.unlink(tmp_file_path)
            except:
                pass

    def create_vectorstore(self, documents, persist_directory="./chroma_db", force_recreate=False):
        """创建或更新向量数据库

        Args:
            documents: 文档列表
            persist_directory: 数据库存储目录
            force_recreate: 是否强制重建（清空旧数据）
        """
        import os
        import shutil
        import time
        from langchain_community.vectorstores import Chroma

        # 如果文档为空
        if not documents:
            st.error("没有文档可处理")
            return None

        # 如果需要强制重建，删除旧数据库
        if force_recreate and os.path.exists(persist_directory):
            try:
                # 先清除 vectorstore 引用
                if hasattr(st.session_state, 'vectorstore') and st.session_state.vectorstore:
                    st.session_state.vectorstore = None
                    time.sleep(0.5)

                # 删除整个数据库目录
                shutil.rmtree(persist_directory)
                st.info(f"🗑️ 已清空旧数据库")
                time.sleep(0.5)
            except Exception as e:
                st.warning(f"无法删除旧数据库: {str(e)[:100]}")
                # 使用新目录
                import uuid
                persist_directory = f"./chroma_db_{uuid.uuid4().hex[:8]}"
                st.info(f"使用新目录: {persist_directory}")

        # 检查数据库是否已存在（非强制重建模式）
        elif os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0:
            try:
                # 加载现有数据库
                st.info(f"📂 加载现有数据库...")
                vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )

                # 获取当前文档数量
                try:
                    current_count = vectorstore._collection.count()
                    st.info(f"当前数据库包含 {current_count} 个文档片段")
                except:
                    current_count = 0

                # 追加新文档
                st.info(f"➕ 正在追加 {len(documents)} 个新文档...")
                vectorstore.add_documents(documents)
                vectorstore.persist()

                # 获取新总数
                try:
                    new_count = vectorstore._collection.count()
                    st.success(f"✅ 数据库已更新！从 {current_count} 增加到 {new_count} 个片段")
                except:
                    st.success(f"✅ 数据库已更新！新增 {len(documents)} 个片段")

                return vectorstore

            except Exception as e:
                st.warning(f"加载现有数据库失败: {str(e)[:100]}")
                st.info("将创建新数据库...")
                # 继续创建新数据库

        # 创建新数据库
        try:
            st.info(f"📝 创建新数据库，包含 {len(documents)} 个文档片段...")
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_directory
            )
            vectorstore.persist()
            st.success(f"✅ 创建成功！数据库包含 {len(documents)} 个文档片段")
            return vectorstore
        except Exception as e:
            st.error(f"创建向量数据库失败: {str(e)}")
            return None

class QASystem:
    """问答系统 - 使用真实LLM"""

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = self._init_real_llm()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

    def _init_real_llm(self):
        """初始化真实的LLM模型"""
        # 修改1：使用正确的导入路径
        from langchain_ollama import ChatOllama

        try:
            # 修改2：使用 ChatOllama 而不是 Ollama
            llm = ChatOllama(
                model="qwen2.5:3b",  # 您下载的模型名称
                temperature=0.7,
                base_url="http://localhost:11434"  # Ollama 默认地址
            )
            # 测试连接
            test_response = llm.invoke("测试")
            st.success("✅ 使用 Ollama 模型 (qwen2.5:3b)")
            return llm
        except Exception as e:
            st.error(f"Ollama 连接失败: {str(e)}")
            st.info("请确保 Ollama 服务已启动 (运行: ollama serve)")
            # 如果 Ollama 不可用，使用备用方案
            return self._init_backup_llm()

    def _init_backup_llm(self):
        """备用LLM方案"""
        from langchain_community.llms import FakeListLLM

        responses = [
            "文档中未找到相关信息，请检查文档内容。",
            "抱歉，当前无法获取准确回答，请确保 Ollama 服务已启动。",
            "请先启动 Ollama 服务: ollama serve"
        ]
        st.warning("⚠️ 使用模拟模式，请启动 Ollama 服务以获得真实回答")
        return FakeListLLM(responses=responses)

    def get_answer(self, question):
        """获取基于文档的真实回答"""
        # 1. 检索相关文档
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        docs = retriever.invoke(question)  # 修改：使用 invoke 而不是 get_relevant_documents

        # 如果没有找到相关文档
        if not docs:
            return "未找到相关文档内容", []

        # 2. 构建提示词
        context = "\n\n".join([f"[来源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}" for doc in docs])
        sources = list(set([doc.metadata.get("source", "未知") for doc in docs]))

        prompt = f"""请基于以下文档内容回答问题。如果文档中没有相关信息，请说明"文档中未找到相关信息"。

文档内容：
{context}

问题：{question}

要求：
1. 只基于上面提供的文档内容回答
2. 回答时要注明信息来源（文档名称）
3. 回答要简洁准确

回答："""

        # 3. 生成回答
        try:
            # 修改：ChatOllama 使用 invoke 方法
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            answer = f"生成回答时出错: {str(e)}"

        return answer, docs


def main():
    """主函数"""
    st.title("📚 内部文档智能问答助手")

    # 初始化 session state
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'all_documents' not in st.session_state:
        st.session_state.all_documents = []  # 存储所有文档片段
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'history' not in st.session_state:
        st.session_state.history = []

    # 侧边栏 - 文档上传和历史记录
    with st.sidebar:
        st.header("📤 文档上传")

        uploaded_files = st.file_uploader(
            "选择文档 (支持PDF, Word, TXT)",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True
        )

        # 处理文档按钮
        # 处理文档按钮部分
        if uploaded_files and st.button("处理文档"):
            with st.spinner("正在处理文档..."):
                processor = DocumentProcessor()
                new_splits = []

                # 显示处理进度
                status_text = st.empty()
                progress_bar = st.progress(0)

                # 处理每个文件
                for i, file in enumerate(uploaded_files):
                    if file.name not in st.session_state.processed_files:
                        try:
                            status_text.text(f"正在处理: {file.name}")
                            splits = processor.process_uploaded_file(file)
                            if splits:
                                new_splits.extend(splits)
                                st.session_state.processed_files.add(file.name)
                                st.success(f"✅ {file.name} 处理成功 ({len(splits)}个片段)")
                        except Exception as e:
                            st.error(f"❌ {file.name} 处理失败: {str(e)}")
                    progress_bar.progress((i + 1) / len(uploaded_files))

                status_text.empty()

                # 累积新文档到总文档集
                if new_splits:
                    # 检查是否需要强制重建（如果 all_documents 为空但有新文档）
                    force_recreate = len(st.session_state.all_documents) == 0

                    # 更新总文档列表
                    st.session_state.all_documents.extend(new_splits)
                    st.info(f"📊 当前共有 {len(st.session_state.all_documents)} 个文档片段")

                    # 创建或更新数据库
                    with st.spinner(f"正在更新向量数据库..."):
                        # 如果 all_documents 和新文档数量相同，说明是清空后的首次添加
                        if len(st.session_state.all_documents) == len(new_splits):
                            st.info("检测到新建数据库模式")
                            st.session_state.vectorstore = processor.create_vectorstore(
                                st.session_state.all_documents,
                                force_recreate=True  # 强制重建
                            )
                        else:
                            st.session_state.vectorstore = processor.create_vectorstore(
                                new_splits,
                                force_recreate=False  # 追加模式
                            )

                        if st.session_state.vectorstore:
                            st.success(f"✨ 数据库已更新")
                        else:
                            st.error("创建向量数据库失败")
                else:
                    st.warning("没有新文档被处理")
        st.divider()

        # 文档统计
        st.header("📊 文档统计")

        # 显示统计信息
        if st.session_state.all_documents:
            # 尝试获取数据库中的实际数量
            try:
                if st.session_state.vectorstore:
                    count = st.session_state.vectorstore._collection.count()
                    st.metric("文档片段数", count, delta=None)
                else:
                    st.metric("文档片段数", len(st.session_state.all_documents))
            except:
                st.metric("文档片段数", len(st.session_state.all_documents))

            st.metric("已处理文件", len(st.session_state.processed_files))

            # 显示已处理文件列表
            if st.session_state.processed_files:
                st.subheader("📄 已处理文件")
                for file in list(st.session_state.processed_files)[-5:]:
                    st.write(f"- {file}")
                if len(st.session_state.processed_files) > 5:
                    st.write(f"... 还有 {len(st.session_state.processed_files) - 5} 个文件")
        else:
            st.info("暂无文档数据")

        # 管理按钮
        st.divider()
        st.subheader("🗂️ 文档管理")
        # 在文档管理部分添加
        st.subheader("🔧 高级操作")

        if st.button("🔄 重建数据库", use_container_width=True):
            with st.spinner("正在重建数据库..."):
                if st.session_state.all_documents:
                    processor = DocumentProcessor()
                    st.session_state.vectorstore = processor.create_vectorstore(
                        st.session_state.all_documents,
                        force_recreate=True
                    )
                    if st.session_state.vectorstore:
                        st.success(f"✅ 数据库重建成功！包含 {len(st.session_state.all_documents)} 个文档片段")
                    else:
                        st.error("数据库重建失败")
                else:
                    st.warning("没有文档可重建")
            st.rerun()
        # 清空所有文档按钮
        # 清空所有文档按钮
        # 清空所有文档按钮
        if st.button("🗑️ 清空所有文档", use_container_width=True):
            import shutil
            import time

            with st.spinner("正在清空所有文档..."):
                # 1. 清除 vectorstore 引用，释放连接
                if st.session_state.vectorstore:
                    st.session_state.vectorstore = None
                    time.sleep(0.5)

                # 2. 清空 session state 中的文档列表
                st.session_state.processed_files = set()
                st.session_state.all_documents = []

                # 3. 删除数据库文件夹
                db_path = "./chroma_db"
                if os.path.exists(db_path):
                    try:
                        shutil.rmtree(db_path)
                        st.success("✅ 数据库文件已删除")
                    except Exception as e:
                        st.warning(f"无法删除数据库文件: {str(e)[:100]}")
                        st.info("尝试使用新目录创建数据库")
                        # 清空 session_state 中的路径记录
                        st.session_state.db_path = None

                # 4. 可选：删除所有临时文件
                temp_files = [f for f in os.listdir(".") if f.startswith("chroma_db_")]
                for temp_file in temp_files:
                    try:
                        shutil.rmtree(temp_file)
                    except:
                        pass

                st.success("✅ 已清空所有文档和数据库")
                time.sleep(1)
                st.rerun()

        # 历史记录
        st.divider()
        st.header("📋 对话历史")
        if st.session_state.history:
            for i, (q, a) in enumerate(st.session_state.history[-10:]):
                with st.expander(f"Q: {q[:30]}..."):
                    st.write(f"A: {a[:100]}...")
        else:
            st.info("暂无历史记录")

        # 清空历史按钮
        if st.button("清空历史", use_container_width=True):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()

    # 主界面 - 聊天区域
    chat_col, stats_col = st.columns([3, 1])

    with chat_col:
        st.header("💬 问答对话")

        # 显示聊天消息
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("📚 信息来源"):
                        for source in message["sources"]:
                            st.write(f"- {source}")

    with stats_col:
        st.header("📈 当前状态")

        # 显示数据库信息
        if st.session_state.vectorstore:
            try:
                count = st.session_state.vectorstore._collection.count()
                st.metric("📄 文档片段", count)
                st.metric("📁 已处理文件", len(st.session_state.processed_files))
            except:
                st.metric("📄 文档片段", len(st.session_state.all_documents))
                st.metric("📁 已处理文件", len(st.session_state.processed_files))
        else:
            st.info("📭 暂无文档数据")
            st.caption("请先在左侧上传文档")

    # 聊天输入框
    st.divider()
    if prompt := st.chat_input("请输入您的问题..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 立即显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)

        # 生成回答
        if st.session_state.vectorstore:
            with st.chat_message("assistant"):
                with st.spinner("正在检索文档并生成回答..."):
                    try:
                        # 初始化问答系统
                        qa_system = QASystem(st.session_state.vectorstore)

                        # 获取回答（使用 get_answer 方法）
                        answer, docs = qa_system.get_answer(prompt)

                        # 显示回答
                        st.markdown(answer)

                        # 提取信息来源
                        sources = list(set([doc.metadata.get("source", "未知文档") for doc in docs]))
                        if sources:
                            with st.expander("📚 信息来源"):
                                for source in sources:
                                    st.write(f"- {source}")

                        # 保存到session
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })

                        # 保存到历史记录
                        st.session_state.history.append((prompt, answer))

                    except Exception as e:
                        error_msg = f"抱歉，处理您的问题时出现错误：{str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })
        else:
            with st.chat_message("assistant"):
                st.warning("请先上传并处理文档")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "请先上传并处理文档"
                })

        st.rerun()
if __name__ == "__main__":
    main()

