"""LangGraph 节点实现。

流程节点:
    classify_intent : 意图识别(LLM 优先, 规则兜底)
    retrieve        : RAG 混合检索故障知识
    diagnose        : 基于检索上下文生成诊断结论(GLM 或演示模式)
    record          : 诊断结果归档 SQLite

同时包含: 智谱 GLM 客户端封装、SQLite 初始化与历史查询。
"""
import json
import os
import sqlite3
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv

from graph.state import DiagnosisState
from rag.data_loader import hybrid_retrieve

# ---------- 路径与配置 ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "").strip()
ZHIPU_MODEL = os.getenv("ZHIPU_MODEL", "glm-4-flash").strip()

INTENT_GREETING = "greeting"
INTENT_LOG = "log_analysis"
INTENT_QA = "general_qa"
INTENT_FALLBACK = "fallback"


def _resolve_db_path() -> str:
    raw = os.getenv("SQLITE_DB_PATH", "db/diagnose_history.db").strip()
    return raw if os.path.isabs(raw) else os.path.join(PROJECT_ROOT, raw)


DB_PATH = _resolve_db_path()


# ---------- SQLite ----------
def init_db() -> None:
    """初始化 SQLite: 建目录 + 建表(幂等)。"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS diagnose_history (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                query          TEXT NOT NULL,
                intent         TEXT,
                answer         TEXT,
                retrieved_docs TEXT,
                created_at     TEXT
            )
            """
        )


def fetch_history(limit: int = 20) -> List[dict]:
    """查询诊断历史, 按时间倒序, retrieved_docs 还原为列表。"""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, query, intent, answer, retrieved_docs, created_at "
            "FROM diagnose_history ORDER BY id DESC LIMIT ?",
            (max(1, min(int(limit), 200)),),
        ).fetchall()
        return [
            {**dict(r), "retrieved_docs": json.loads(r["retrieved_docs"] or "[]")}
            for r in rows
        ]


# ---------- 智谱 GLM 封装 ----------
class GLMClient:
    """智谱 GLM 对话客户端。

    未配置 ZHIPU_API_KEY 时 enabled=False, 上层节点走「演示模式」规则回答,
    保证项目无 Key 也能完整跑通 RAG + Agent 链路。
    """

    def __init__(self) -> None:
        self.api_key = ZHIPU_API_KEY
        self.model = ZHIPU_MODEL
        self.enabled = bool(self.api_key)
        self._client = None
        if self.enabled:
            try:
                from zhipuai import ZhipuAI

                self._client = ZhipuAI(api_key=self.api_key)
            except Exception as exc:  # SDK 缺失或初始化失败 -> 退化为演示模式
                self.enabled = False
                print(f"[warn] GLM 初始化失败, 进入演示模式: {exc}")

    def chat(self, prompt: str, system: str = "", temperature: float = 0.3) -> str:
        """同步对话, 返回模型回复文本。"""
        if not self._client:
            raise RuntimeError("ZHIPU_API_KEY 未配置或 GLM 客户端不可用")
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


llm = GLMClient()


# ---------- 规则兜底 ----------
GREETING_WORDS = ("你好", "您好", "hi", "hello", "在吗", "早上好", "下午好", "晚上好", "help", "帮助")
LOG_KEYWORDS = (
    "日志", "报错", "错误", "异常", "故障", "排查", "崩溃", "crash", "error", "exception",
    "oom", "内存", "cpu", "磁盘", "死锁", "慢查询", "502", "504", "拒绝连接", "超时", "timeout",
)


def rule_intent(query: str) -> str:
    """规则意图识别(LLM 不可用或调用失败时的兜底)。"""
    q = query.lower()
    if len(query) <= 20 and any(w in q for w in GREETING_WORDS):
        return INTENT_GREETING
    if any(k in q for k in LOG_KEYWORDS):
        return INTENT_LOG
    return INTENT_FALLBACK


# ---------- 节点: 意图识别 ----------
def classify_intent(state: DiagnosisState) -> DiagnosisState:
    """意图识别: 优先 LLM 分类, 失败/演示模式下用规则兜底。"""
    query = (state.get("user_query") or "").strip()
    intent = INTENT_FALLBACK

    if query:
        if llm.enabled:
            prompt = (
                "判断以下用户输入的意图, 只输出一个英文标签, 不要输出任何解释。\n"
                "可选标签: \n"
                "- greeting: 问候、打招呼、询问在不在\n"
                "- log_analysis: 提供了日志/报错/系统异常信息, 希望排查故障\n"
                "- general_qa: 普通问题咨询\n"
                f"用户输入: {query}\n"
                "意图标签:"
            )
            try:
                label = llm.chat(prompt, temperature=0.1).strip().lower()
                if "log" in label or "analysis" in label:
                    intent = INTENT_LOG
                elif "greeting" in label or "hello" in label:
                    intent = INTENT_GREETING
                elif "general" in label or "qa" in label:
                    intent = INTENT_QA
                else:
                    intent = rule_intent(query)
            except Exception:
                intent = rule_intent(query)
        else:
            intent = rule_intent(query)

    return {**state, "intent": intent}


# ---------- 节点: RAG 检索 ----------
def retrieve(state: DiagnosisState) -> DiagnosisState:
    """RAG 检索: 仅对日志故障类输入执行知识库混合检索。"""
    if state.get("intent") != INTENT_LOG:
        return {**state, "retrieved_docs": []}

    query = state.get("user_query") or ""
    try:
        docs = hybrid_retrieve(query, top_k=3)
        return {**state, "retrieved_docs": [d.page_content for d in docs]}
    except Exception as exc:
        return {**state, "retrieved_docs": [], "error": f"知识库检索失败: {exc}"}


# ---------- 节点: 诊断 ----------
def demo_answer(query: str, docs: List[str]) -> str:
    """演示模式回答: 直接汇总知识库检索结果, 保证无 API Key 也能看到完整输出结构。"""
    head = f"【演示模式 · 未配置 ZHIPU_API_KEY, 以下为知识库检索到的相关内容】\n待诊断日志/故障: {query}\n"
    if not docs:
        return head + "\n未检索到匹配的故障模式, 可在 rag/data/fault_knowledge.json 中补充, 或配置智谱 API Key 获得完整诊断。"
    parts = [head]
    for i, text in enumerate(docs):
        parts.append(f"—— 知识片段 {i + 1} ——\n{text}")
    return "\n\n".join(parts)


def diagnose(state: DiagnosisState) -> DiagnosisState:
    """生成诊断结论: GLM 基于 RAG 上下文推理; 演示模式返回知识库原文。"""
    query = (state.get("user_query") or "").strip()
    intent = state.get("intent")

    if intent == INTENT_GREETING:
        answer = (
            "你好, 我是日志故障诊断助手。把系统日志、报错信息或故障现象发给我, "
            "我会基于故障知识库帮你定位根因并给出排查方案。"
        )
        return {**state, "final_answer": answer}

    docs = state.get("retrieved_docs") or []
    if intent == INTENT_LOG and docs:
        context = "\n\n".join(f"—— 知识片段 {i + 1} ——\n{text}" for i, text in enumerate(docs))
    else:
        context = "(知识库未检索到直接相关的故障模式)"

    if llm.enabled:
        system_prompt = (
            "你是一名资深的系统运维与日志故障诊断专家。请严格基于【检索到的相关故障知识】定位根因并给出可执行方案; "
            "若知识库无法覆盖, 可结合通用运维经验回答, 但必须明确标注「此为通用经验, 未经知识库验证」。"
            "回答结构固定: 1. 故障类型判断  2. 根因分析  3. 处理方案与排查命令。语言简洁专业。"
        )
        user_prompt = (
            f"【检索到的相关故障知识】\n{context}\n\n"
            f"【用户提供的日志/故障信息】\n{query}\n"
        )
        try:
            answer = llm.chat(user_prompt, system=system_prompt, temperature=0.3)
        except Exception as exc:
            answer = demo_answer(query, docs) + f"\n\n(注: GLM 调用失败, 已回退演示模式, 原因: {exc})"
    else:
        answer = demo_answer(query, docs)

    return {**state, "final_answer": answer}


# ---------- 节点: SQLite 归档 ----------
def record(state: DiagnosisState) -> DiagnosisState:
    """将本次诊断结果归档到 SQLite 历史表。"""
    try:
        init_db()
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute(
                "INSERT INTO diagnose_history(query, intent, answer, retrieved_docs, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    state.get("user_query", ""),
                    state.get("intent", ""),
                    state.get("final_answer", ""),
                    json.dumps(state.get("retrieved_docs") or [], ensure_ascii=False),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            history_id: Optional[int] = cur.lastrowid
        return {**state, "history_id": history_id}
    except Exception as exc:
        return {**state, "error": f"SQLite 归档失败: {exc}"}
