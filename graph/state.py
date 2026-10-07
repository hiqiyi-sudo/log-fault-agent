"""LangGraph 状态定义: 整个诊断流程共享的数据结构。"""
from typing import List, Optional, TypedDict


class DiagnosisState(TypedDict):
    """日志故障诊断 Agent 的状态对象。

    字段说明:
        user_query:     用户输入的日志 / 故障描述
        intent:         意图识别结果(greeting / log_analysis / general_qa / fallback)
        retrieved_docs: RAG 检索到的故障知识片段(作为 LLM 的检索增强上下文)
        final_answer:   最终诊断结论
        history_id:     SQLite 归档记录 ID
        error:          流程中出现的错误信息(可选)
    """

    user_query: str
    intent: str
    retrieved_docs: List[str]
    final_answer: str
    history_id: Optional[int]
    error: Optional[str]
