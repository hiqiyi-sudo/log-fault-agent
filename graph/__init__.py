"""graph 包: LangGraph 诊断工作流。

流程: classify_intent(意图识别) -> retrieve(RAG检索) -> diagnose(GLM诊断) -> record(SQLite归档)
"""
