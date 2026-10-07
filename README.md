# 日志故障诊断 Agent（Log Fault Diagnosis Agent）

基于 **FastAPI + LangGraph + RAG + Chroma + 智谱 GLM** 的智能日志故障诊断系统：输入一段系统日志或故障描述，自动识别意图 → 从故障知识库中检索相关历史故障样本 → 由大模型生成根因分析与排查方案 → 归档为工单记录。

## 项目简介

本项目是一个可完整运行的日志故障诊断 RAG Agent，核心链路：

```
用户输入日志/故障 → 意图识别 → RAG 混合检索 → LLM 生成诊断 → SQLite 工单归档
```

- **数据源**：Kaggle HDFS 日志数据集（真实分布式存储系统运行日志，含 57.5 万个数据块及 Normal/Anomaly 标注），用于构建故障知识库
- **检索**：向量相似度 + 关键词重合度混合排序，兼顾语义与字面召回
- **Agent**：LangGraph 编排「意图识别 → 检索 → 诊断 → 归档」四个节点，可扩展
- **持久化**：每次诊断自动归档 SQLite，支持历史查询与工单追溯

## 技术栈

| 模块 | 技术 | 说明 |
|---|---|---|
| Web 框架 | FastAPI + Uvicorn | REST 接口 |
| Agent 编排 | LangGraph | 有向图编排多节点流程 |
| 向量检索 | ChromaDB（原生 SDK, PersistentClient） | 持久化向量库，cosine 距离 |
| Embedding | 智谱 embedding-3（2048 维） | 文档与查询向量化；无额度/失败时自动降级 |
| LLM | 智谱 GLM（glm-4-flash） | 意图分类与诊断生成 |
| 持久化 | SQLite | 诊断工单历史表 |
| 配置 | python-dotenv + .env | API Key 与路径配置 |

> 说明：`chromadb` 版本锁定 `>=1.5.0,<2.0.0`——0.5.0~1.0.0 在 Windows/Python 3.12 下依赖 `chroma-hnswlib` C++ 扩展（官方 wheel 仅到 cp39），1.5.x 已移除该依赖、纯 wheel 可安装。

## 系统架构

```
                    ┌───────────────────────────────┐
                    │        FastAPI (main.py)       │
                    │  GET / · POST /api/diagnose    │
                    │           GET /api/history     │
                    └──────────────┬────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────┐
                    │      LangGraph Agent (graph/)  │
                    │   classify_intent → retrieve   │
                    │        → diagnose → record     │
                    └───────┬─────────────┬─────────┘
                            │             │
                            ▼             ▼
              ┌───────────────────┐  ┌──────────────┐
              │ RAG 混合检索        │  │ SQLite 工单   │
              │ rag/data_loader.py│  │ diagnose_    │
              └─────────┬─────────┘  │ history.db   │
                        │            └──────────────┘
                        ▼
              ┌───────────────────┐
              │ Chroma 向量库       │
              │ rag/data/chroma    │
              │ 600 条 HDFS 样本    │
              │ (2048 维)          │
              └───────────────────┘
```

**数据流说明**：

1. **意图识别**：LLM 优先判断输入是日志故障、问候还是普通问答，失败时规则兜底
2. **RAG 检索**：仅对日志故障类输入执行混合检索，召回 top-3 相关知识片段
3. **诊断生成**：GLM 基于检索上下文按固定结构（故障类型/根因/排查命令）生成结论；知识库未覆盖时明确标注「通用经验，未经知识库验证」
4. **工单归档**：诊断结果写入 SQLite，`/api/history` 可查询

## 数据集来源

**Kaggle HDFS_v1 LogHub Dataset Archive**：[https://www.kaggle.com/datasets/tamaniwilliams/hdfs-v1-loghub-dataset-archive](https://www.kaggle.com/datasets/tamaniwilliams/hdfs-v1-loghub-dataset-archive)

- HDFS（Hadoop Distributed File System）运行日志，由伊利诺伊大学 LogHub/LogPAI 项目发布，原始数据约 **1117 万条日志消息**
- 日志按 **block_id**（`blk_` 标识）分组为数据块轨迹，`anomaly_label.csv` 为每个 block 提供 **Normal / Anomaly** 标注
- 本项目使用：`HDFS.log`（原始日志）+ `anomaly_label.csv`（标签），放置在 `rag/data/` 目录

> 数据集文件较大（日志约 1.5GB），已通过 `.gitignore` 排除，不入版本库。

## 环境准备

```bash
# 1. 安装 Python 3.12（需包含 pip）
# 2. 创建虚拟环境（Windows）
cd E:\pythonProject1\log-fault-agent
py -3.12 -m venv .venv

# 3. 激活并安装依赖
.venv\Scripts\activate
pip install -r requirements.txt

# 4. 配置 .env（复制 .env.example 并填写）
#    ZHIPU_API_KEY=你的智谱开放平台APIKey
```

`.env` 关键配置项：

```ini
ZHIPU_API_KEY=your_key_here
ZHIPU_MODEL=glm-4-flash
CHROMA_PERSIST_DIR=rag/data/chroma
SQLITE_DB_PATH=db/diagnose_history.db
```

> `.env` 含密钥，已被 `.gitignore` 排除，切勿提交到 GitHub。

## 项目启动步骤

```bash
# 1.（可选）基于 HDFS 数据集构建向量知识库（重建会清空旧数据）
.venv\Scripts\python.exe rag\data_loader.py
# 成功输出: "向量库已使用Kaggle HDFS数据集构建完毕"

# 2. 启动服务
.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000

# 3. 访问
#    接口文档: http://127.0.0.1:8000/docs
#    健康检查: http://127.0.0.1:8000/
```

> 服务启动时自动加载向量库；若向量库为空但检测到 HDFS 数据文件，会提示先执行构建命令；若完全没有 HDFS 数据，则自动灌入内置故障知识（`rag/data/fault_knowledge.json`，10 条常见故障）保证演示可用。

## 接口说明

### 1. `GET /` 健康检查

```bash
curl http://127.0.0.1:8000/
```

### 2. `POST /api/diagnose` 日志故障诊断

请求体（JSON）：

```json
{
  "log_text": "081109 203518 143 INFO dfs.DataNode: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010 但随后出现 java.io.IOException 读取失败 网络连接超时"
}
```

响应示例：

```json
{
  "intent": "log_analysis",
  "answer": "1. 故障类型判断\n   ...(GLM 生成的诊断结论)...",
  "history_id": 5,
  "retrieved_count": 3,
  "error": null
}
```

curl 调用（Windows 命令行，`--data-binary @file` 读取 JSON 文件避免转义问题）：

```bash
curl -X POST http://127.0.0.1:8000/api/diagnose --data-binary "@_req_test.json" -H "Content-Type: application/json"
```

### 3. `GET /api/history` 查询诊断工单历史

```bash
curl http://127.0.0.1:8000/api/history
```

## 项目亮点

### 1. 混合检索（Hybrid Retrieval）

向量相似度与关键词重合度**加权融合**排序：向量权重 30% + 关键词权重 70%，两路分数在候选集内分别做 min-max 归一化后合并，消除量纲差异导致的排序失真。关键词打分为英数 token 双向匹配 + 中文 3/4 字窗口，对 `blk_xxx`、`java.io.IOException` 等日志特征召回准确。

### 2. Embedding 自动降级

智谱 Embedding 调用失败（余额不足 / 网络异常 / 单条超长）时自动降级：

- **查询侧**：`hybrid_retrieve` 自动切换为纯关键词检索，服务不中断
- **构建侧**：批量失败自动逐条重试；单条超接口字符上限自动截断（4096 → 3000 字符）重试

### 3. LangGraph Agent 编排

以有向图方式编排「意图识别 → RAG 检索 → 诊断生成 → 工单归档」四节点，节点间通过状态对象传递数据；意图识别 LLM 优先、规则兜底；诊断阶段知识库未覆盖时标注「通用经验」而非强行编造，保证回答可信。

### 4. SQLite 工单持久化

每次诊断自动写入 `db/diagnose_history.db`，记录原始查询、意图、诊断结论与召回的知识片段；`/api/history` 支持按时间倒序查询，形成可追溯的故障工单台账，便于复盘与积累。

## 目录结构

```
log-fault-agent/
├── main.py                 # FastAPI 入口
├── requirements.txt        # 依赖清单
├── .env                    # 密钥与配置（不入库）
├── .gitignore
├── README.md
├── rag/
│   ├── data_loader.py      # HDFS 解析/分块/标签合并/Embedding/Chroma/混合检索
│   └── data/               # HDFS.log、anomaly_label.csv、chroma/（均不入库）
├── graph/
│   ├── state.py            # LangGraph 状态定义
│   ├── nodes.py            # 意图识别/检索/诊断/归档节点
│   └── agent_graph.py      # Agent 图构建
└── db/
    └── diagnose_history.db # SQLite 工单库（不入库）
```


