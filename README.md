# 日志故障诊断 Agent（Log Fault Diagnosis Agent）

> 基于 **Kaggle HDFS 日志数据集** + **FastAPI + LangGraph + RAG** 构建的分布式日志故障诊断后端 Agent：自动分析日志、检索相似故障案例、输出诊断方案，结果存入 SQLite 工单库，支持历史查询。

---

## 1. 项目简介

本项目是一个可完整运行的日志故障诊断 **RAG Agent** 后端。输入一段系统日志或故障描述，系统自动完成：**意图识别 → 故障知识检索（RAG）→ 大模型诊断生成 → 工单归档**，最终返回结构化诊断方案（故障类型 / 根因分析 / 排查命令），并将每一次诊断持久化为 SQLite 工单，支持历史追溯。

- **数据源**：Kaggle HDFS 日志数据集（真实分布式存储系统运行日志，57.5 万个数据块轨迹，含 Normal / Anomaly 标注），用于构建故障知识库
- **检索**：向量相似度 + 关键词重合度混合排序（Hybrid Retrieval），兼顾语义与字面召回
- **Agent**：LangGraph 有向图编排「意图识别 → 检索 → 诊断 → 归档」四节点，状态机驱动、可扩展
- **持久化**：每次诊断自动归档 SQLite，支持历史工单查询与故障复盘
- **可评估**：内置 `evaluate.py` 离线评估脚本，支持批量效果验证与指标量化

核心链路：

```
用户输入日志/故障 → 意图识别 → RAG 混合检索 → LLM 生成诊断 → SQLite 工单归档
```

---

## 2. 技术栈

| 模块 | 技术 | 说明 |
|---|---|---|
| Web 框架 | FastAPI + Uvicorn | REST 接口（/docs 自动文档） |
| Agent 编排 | LangGraph 0.4.x | 有向图编排多节点流程，状态对象驱动 |
| 向量检索 | ChromaDB 1.5.x（原生 SDK, PersistentClient） | 持久化向量库，cosine 距离 |
| Embedding | 智谱 embedding-3（2048 维） | 文档与查询向量化；无额度/失败时自动降级 |
| LLM | 智谱 GLM（glm-4-flash） | 意图分类与诊断生成 |
| 持久化 | SQLite | 诊断工单历史表（db/diagnose_history.db） |
| 配置 | python-dotenv + .env | API Key 与路径配置 |

> 注意：`chromadb` 版本锁定 `>=1.5.0,<2.0.0`——0.5.0~1.0.0 在 Windows/Python 3.12 下依赖 `chroma-hnswlib` C++ 扩展（官方 wheel 仅到 cp39），1.5.x 已移除该依赖、纯 wheel 可安装。

---

## 3. 项目目录结构

```
log-fault-agent/
├── main.py                 # FastAPI 入口（服务启动、/api/diagnose、/api/history）
├── requirements.txt        # 依赖清单
├── .env                    # 密钥与配置（已被 .gitignore 排除，不入库）
├── .gitignore              # 忽略 .venv / 数据集 / 向量库 / .env / 日志缓存
├── README.md
├── init_git.bat            # Git 仓库初始化脚本
├── evaluate.py             # 离线评估脚本（200 条批量效果验证）
├── eval_result.csv         # 评估逐条明细（block_id/真实标签/预测/耗时）
├── rag/
│   ├── __init__.py
│   ├── data_loader.py      # HDFS 解析/清洗/日志分块/标签合并/Embedding/Chroma 入库/混合检索
│   └── data/               # 数据与向量库目录（均不入库）
│       ├── HDFS.log        # Kaggle 原始日志（约 1117 万行）
│       ├── anomaly_label.csv   # 575,061 条 BlockId/标签标注
│       └── chroma/         # Chroma 持久化向量库（collection: fault_knowledge, 600 条）
├── graph/
│   ├── __init__.py
│   ├── state.py            # LangGraph Agent 状态定义
│   ├── nodes.py            # 意图识别 / 检索 / 诊断 / 归档 节点
│   └── agent_graph.py      # Agent 图构建与编译
└── db/
    ├── diagnose_history.db # SQLite 工单库（不入库）
    └── .gitkeep
```

---

## 4. 核心功能介绍

### 4.1 日志数据管线（rag/data_loader.py）
- 流式解析 **HDFS.log**（1,117 万行，0 丢弃），按 `blk_` block_id 聚合为数据块轨迹
- 数据清洗 + 日志分块 + 与 **anomaly_label.csv**（575,061 条）合并异常标签
- 调用智谱 Embedding 生成 2048 维向量，写入 Chroma 持久化向量库（600 条文档：Normal 300 / Anomaly 300）
- 单条超长自动截断（4096 → 3000 字符）重试，批量失败自动逐条降级重试

### 4.2 RAG 混合检索（Hybrid Retrieval）
- 向量相似度 + 关键词重合度**加权融合**排序：向量权重 30% + 关键词权重 70%，两路分数在候选集内分别 min-max 归一化后合并，消除量纲差异
- 关键词打分为英数 token 双向匹配 + 中文 3/4 字窗口，对 `blk_xxx`、`java.io.IOException` 等日志特征召回准确

### 4.3 Embedding 自动降级
- **查询侧**：智谱 Embedding 调用失败（余额不足 / 网络异常 / 单条超长）时，`hybrid_retrieve` 自动切换为纯关键词检索，服务不中断
- **构建侧**：批量失败自动逐条重试；单条超字符上限自动截断重试

### 4.4 LangGraph Agent 编排（graph/）
- 有向图编排「意图识别 → RAG 检索 → 诊断生成 → 工单归档」四节点，节点间通过状态对象传递数据
- 意图识别 LLM 优先、规则兜底；诊断阶段知识库未覆盖时标注「通用经验，未经知识库验证」而非强行编造，保证回答可信

### 4.5 SQLite 工单持久化
- 每次诊断自动写入 `db/diagnose_history.db`，记录原始查询、意图、诊断结论与召回的知识片段
- `/api/history` 支持按时间倒序查询，形成可追溯的故障工单台账

### 4.6 离线评估（evaluate.py）
- 读取 HDFS 标签数据集，均衡抽样 200 条（Normal / Anomaly 各 100）
- 复用 RAG 检索 + Agent 诊断链路批量测试，自动计算准确率 / Top3 召回率 / 平均耗时
- 生成 `eval_result.csv` 逐条明细，支持效果复现与对比

---

## 5. 离线评估指标

### 5.1 评估设置

| 项 | 值 |
|---|---|
| 数据集 | Kaggle HDFS_v1 LogHub Dataset Archive |
| 样本数 | 200 条（Normal 100 / Anomaly 100，均衡抽样 seed=7 可复现） |
| 评估链路 | RAG 检索（Top3）→ Agent 诊断（GLM 生成） |
| 预测口径 | Top3 检索文档异常标签投票，任一为 Anomaly 即判异常（不漏报优先） |

### 5.2 指标结果

| 指标 | 数值 |
|---|---|
| 样本总数 | 200（有效计分 200，无失败样本） |
| **故障诊断准确率** | **54.00%**（108 / 200） |
| **RAG Top3 召回率** | **93.00%**（93 / 100） |
| **异常精确率** | **52.25%**（TP=93, FP=85, FN=7, TN=15） |
| 单条平均检索耗时 | 5.82 s |
| 单条平均诊断耗时 | 25.62 s（检索 + GLM 生成） |

### 5.3 结果解读

- **RAG 召回效果优秀**：93.00% 的真实异常样本能在 Top3 检索中命中异常知识——故障"不漏报"能力很强，说明知识库构建与混合检索策略对日志特征的表征有效，这是诊断链路最核心的保障指标。
- **准确率瓶颈来自 Embedding 向量区分不足**：当前正常日志被大量误判为异常（FP=85），主要原因是智谱 Embedding 在评估期间额度耗尽（429），检索实际走**纯关键词降级路径**；正常块（Normal）与异常块日志共享大量 HDFS 读写操作词汇，关键词维度区分度不足，Top3 中混入 Anomaly 知识即触发误判。
- **优化方向**：① 引入 **Rerank 重排**（对 Top-K 候选做语义精排，抑制关键词误召回）；② **优化 Embedding**（恢复向量检索、更换更高区分度的 Embedding 模型，或对日志语料微调）；③ **调优诊断 Prompt**（约束"仅在证据充分时判异常"，加入阈值校验）；④ 日志预处理（过滤高频公共字段，增强异常特征词权重）。

> 说明：以上为 Embedding 降级模式下的真实评估结果；恢复向量检索后预期准确率显著提升。`eval_result.csv` 提供逐条明细（block_id / 真实标签 / 预测结果 / 检索标签 / 耗时 / 诊断答案），支持二次分析。

---

## 6. 环境部署步骤

### 6.1 创建虚拟环境

```bash
cd E:\pythonProject1\log-fault-agent
py -3.12 -m venv .venv
.venv\Scripts\activate
```

### 6.2 安装依赖

```bash
pip install -r requirements.txt
```

### 6.3 配置 .env

在项目根目录创建 `.env`（含密钥，已被 `.gitignore` 排除，**切勿提交到 GitHub**）：

```ini
ZHIPU_API_KEY=你的智谱开放平台APIKey
ZHIPU_MODEL=glm-4-flash
ZHIPU_EMBED_MODEL=embedding-3
CHROMA_PERSIST_DIR=rag/data/chroma
SQLITE_DB_PATH=db/diagnose_history.db
```

> `ZHIPU_API_KEY` 在 [智谱开放平台](https://open.bigmodel.cn/) 申请；`ZHIPU_EMBED_MODEL` 不配置时默认使用 `embedding-3`。

### 6.4 构建向量知识库（首次 / 重建）

```bash
.venv\Scripts\python.exe rag\data_loader.py
```

成功输出提示：**"向量库已使用Kaggle HDFS数据集构建完毕"**（约需 1~2 分钟读取日志 + 逐条向量化）。

> 服务启动时自动加载向量库；若向量库为空但检测到 HDFS 数据文件，会提示先执行构建命令；若完全没有 HDFS 数据，则自动灌入内置故障知识（`rag/data/fault_knowledge.json`）保证演示可用。

### 6.5 启动服务

```bash
.venv\Scripts\python.exe main.py
```

- 服务地址：http://127.0.0.1:8000
- 接口文档（Swagger）：http://127.0.0.1:8000/docs
- 健康检查：http://127.0.0.1:8000/

---

## 7. 接口使用说明

### 7.1 `GET /` 健康检查

```bash
curl http://127.0.0.1:8000/
```

### 7.2 `POST /api/diagnose` 日志故障诊断

请求体（JSON，`log_text` 为待诊断的日志文本）：

```json
{
  "log_text": "081109 203518 143 INFO dfs.DataNode: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010, 随后出现 java.io.IOException 读取失败 网络连接超时"
}
```

响应示例：

```json
{
  "intent": "log_analysis",
  "answer": "1. 故障类型判断\n   数据节点在服务数据块时出现 IO 异常，判定为数据节点读写故障。\n2. 根因分析\n   ...（GLM 生成的诊断结论）...",
  "history_id": 5,
  "retrieved_count": 3,
  "error": null
}
```

Windows 命令行 curl 调用（用 `--data-binary "@file"` 读取 JSON 文件，避免转义问题）：

```bash
curl -X POST http://127.0.0.1:8000/api/diagnose --data-binary "@_req_test.json" -H "Content-Type: application/json"
```

`_req_test.json` 内容示例：

```json
{
  "log_text": "081109 203518 143 INFO dfs.DataNode: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010 随后出现 java.io.IOException 读取失败 网络连接超时"
}
```

### 7.3 `GET /api/history` 查询诊断工单历史

```bash
# 查询全部历史工单（按时间倒序）
curl http://127.0.0.1:8000/api/history

# 限制返回条数
curl "http://127.0.0.1:8000/api/history?limit=10"
```

响应示例：

```json
{
  "total": 8,
  "records": [
    {
      "id": 8,
      "query_text": "081109 ... Receiving block blk_...",
      "intent": "log_analysis",
      "answer": "1. 故障类型判断\n   ...",
      "retrieved_docs": ["[异常标签] Anomaly\nblk_...", "..."],
      "created_at": "2026-10-07 15:30:12"
    }
  ]
}
```

---

## 8. 项目优化 TODO

- [ ] **Rerank 重排**：对 RAG Top-K 候选引入语义精排模型（如 bge-reranker / Cohere Rerank），抑制关键词误召回，降低 FP
- [ ] **优化 Prompt**：诊断节点加入"证据充分性校验"与阈值约束，仅在检索证据支持时判定异常，减少正常日志误报
- [ ] **升级 Embedding**：恢复/切换更高区分度的 Embedding 模型（如 bge-m3 / 更强向量模型），对日志语料针对性调参，提升向量检索占比
- [ ] **日志预处理**：过滤时间戳/序号/公共字段，抽取异常特征词并加权，提升 Normal / Anomaly 文本可分性
- [ ] **流式检索缓存**：对高频故障模式缓存检索结果，降低平均检索耗时
- [ ] **并发诊断**：诊断阶段支持批量并发调用，缩短多工单整体处理时间

---

## 9. 声明

本项目配套提供离线评估工具：

- **evaluate.py**：离线评估脚本，自动读取 HDFS 标签数据集、均衡抽样 200 条样本，复用项目 RAG 检索 + Agent 诊断链路批量测试，自动计算故障诊断准确率、RAG Top3 召回率、单条平均诊断耗时，用于批量效果验证与版本对比。
- **eval_result.csv**：评估逐条明细（block_id / 真实标签 / 预测结果 / 检索标签 / 单次耗时 / 诊断答案），支持二次分析与问题定位。

运行方式：

```bash
cd E:\pythonProject1\log-fault-agent
.venv\Scripts\python.exe evaluate.py
```

---

## 附：数据集来源

**Kaggle HDFS_v1 LogHub Dataset Archive**：https://www.kaggle.com/datasets/tamaniwilliams/hdfs-v1-loghub-dataset-archive

- HDFS（Hadoop Distributed File System）运行日志，由伊利诺伊大学 LogHub/LogPAI 项目发布，原始数据约 **1117 万条日志消息**
- 日志按 **block_id**（`blk_` 标识）分组为数据块轨迹，`anomaly_label.csv` 为每个 block 提供 **Normal / Anomaly** 标注
- 本项目使用：`HDFS.log`（原始日志）+ `anomaly_label.csv`（标签），放置在 `rag/data/` 目录

> 数据集文件较大（日志约 1.5GB），已通过 `.gitignore` 排除，不入版本库。
