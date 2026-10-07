"""LangGraph 工作流图: 编排日志故障诊断 Agent 的节点与路由。

拓扑:
    START -> classify_intent
             |--(log_analysis / general_qa / fallback)--> retrieve -> diagnose -> record -> END
             |--(greeting)---------------------------------------> diagnose -> record -> END
"""
from langgraph.graph import END, START, StateGraph

from graph.nodes import classify_intent, diagnose, record, retrieve
from graph.state import DiagnosisState


def route_after_intent(state: DiagnosisState) -> str:
    """意图路由: 问候类输入跳过 RAG 检索, 直接进入诊断节点。"""
    return "direct" if state.get("intent") == "greeting" else "rag"


def build_agent():
    """构建并编译诊断工作流图。"""
    builder = StateGraph(DiagnosisState)

    builder.add_node("classify_intent", classify_intent)
    builder.add_node("retrieve", retrieve)
    builder.add_node("diagnose", diagnose)
    builder.add_node("record", record)

    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {"rag": "retrieve", "direct": "diagnose"},
    )
    builder.add_edge("retrieve", "diagnose")
    builder.add_edge("diagnose", "record")
    builder.add_edge("record", END)

    return builder.compile()


# 全局单例: main.py 直接 import 使用
agent = build_agent()
