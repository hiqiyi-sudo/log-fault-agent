"""rag 包: 日志故障知识库加载与混合检索(RAG 检索模块)。

对外暴露:
    get_vectorstore : 获取/构建 Chroma 持久化向量库(进程内单例)
    hybrid_retrieve : 混合检索(向量相似度 + 关键词重合度)
    load_knowledge  : 加载故障知识库 JSON
"""
