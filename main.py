"""FastAPI 服务入口: 日志故障诊断 Agent 后端。

接口:
    GET  /api/history        查询历史诊断记录
    POST /api/diagnose       提交日志/故障, Agent 完成 意图识别 -> RAG检索 -> GLM诊断 -> SQLite归档
"""
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from graph.agent_graph import agent  # noqa: E402  需在 .env 加载后导入
from graph.nodes import fetch_history, init_db  # noqa: E402
from rag.data_loader import get_vectorstore  # noqa: E402


@asynccontextmanager
async def lifespan(_: FastAPI):
    """服务启动时: 初始化 SQLite 表 + 预热构建/加载 Chroma 向量库。"""
    init_db()
    get_vectorstore()
    yield


app = FastAPI(
    title="Log Fault Diagnosis Agent",
    description="FastAPI + LangGraph + RAG(Chroma) + SQLite + 智谱GLM 的日志故障诊断后端",
    version="1.0.0",
    lifespan=lifespan,
)


class DiagnoseRequest(BaseModel):
    log_text: str = Field(
        ...,
        description="日志片段、报错信息或故障现象描述",
        min_length=1,
        max_length=4000,
    )


class DiagnoseResponse(BaseModel):
    intent: str
    answer: str
    history_id: Optional[int]
    retrieved_count: int
    error: Optional[str]


@app.get("/", tags=["基础"])
def health() -> dict:
    return {
        "service": "log-fault-agent",
        "status": "ok",
        "message": "日志故障诊断 Agent 服务运行中",
    }


@app.post("/api/diagnose", response_model=DiagnoseResponse, tags=["诊断"])
def diagnose(req: DiagnoseRequest) -> DiagnoseResponse:
    """核心接口: 提交日志/故障描述, 返回 Agent 诊断结论。"""
    try:
        result = agent.invoke({"user_query": req.log_text.strip()})
        return DiagnoseResponse(
            intent=result.get("intent", ""),
            answer=result.get("final_answer", ""),
            history_id=result.get("history_id"),
            retrieved_count=len(result.get("retrieved_docs") or []),
            error=result.get("error"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent 执行失败: {exc}")


@app.get("/api/history", tags=["诊断"])
def history(limit: int = Query(20, ge=1, le=200)) -> List[dict]:
    """查询历史诊断记录(按时间倒序)。"""
    return fetch_history(limit)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
