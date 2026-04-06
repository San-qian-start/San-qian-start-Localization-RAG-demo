import os
from langchain.embeddings import HuggingFaceEmbeddings

# 设置模型路径
model_path = "E:\\my_ai_model\\model_dir\\BAAI\\bge-large-zh-v1___5"

print(f"检查路径: {model_path}")
print(f"路径是否存在: {os.path.exists(model_path)}")

if os.path.exists(model_path):
    print("路径中的文件:")
    for file in os.listdir(model_path)[:10]:  # 显示前10个文件
        print(f"  - {file}")

    try:
        # 尝试加载模型
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 测试编码
        test_text = "这是一个测试句子"
        embedding = embeddings.embed_query(test_text)
        print(f"✅ 模型加载成功")
        print(f"   向量维度: {len(embedding)}")

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
else:
    print("❌ 路径不存在，请检查：")
    print("1. 路径是否正确")
    print("2. 是否有权限访问")
    print("3. 是否使用了正确的分隔符（\\ 或 /）")