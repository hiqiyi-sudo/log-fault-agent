"""日志故障知识库加载与向量索引构建 (RAG 检索模块).

v2 核心变化:
    - 数据源: 由内置模拟故障知识 升级为 Kaggle HDFS 日志数据集
        * rag/data/HDFS.log           HDFS 原始运行日志(约 1100 万行)
        * rag/data/anomaly_label.csv  block_id 级异常标签(Normal/Anomaly)
    - 流程: 日志解析清洗 -> 按 block_id 分块 -> 合并异常标签 -> 智谱 Embedding 向量化 -> Chroma 持久化
    - 检索: 向量相似度(30%) + 关键词重合度(70%) 混合排序

接口兼容(main.py / graph 模块无需改动):
    get_vectorstore()   FastAPI 启动预热: 确保集合可用
    hybrid_retrieve()   LangGraph 节点检索: 返回 List[Document]
    load_knowledge()    加载知识(降级时返回内置故障知识)

构建方式:
    python rag/data_loader.py   -> 基于 HDFS 数据集重建向量库
"""
import json
import os
import random
import re
import sys
import time
from collections import Counter
from typing import List

import chromadb
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings

# ---------- 路径与配置 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

DATA_DIR = os.path.join(BASE_DIR, "data")
KNOWLEDGE_FILE = os.path.join(DATA_DIR, "fault_knowledge.json")
HDFS_LOG_FILE = os.path.join(DATA_DIR, "HDFS.log")
HDFS_LABEL_FILE = os.path.join(DATA_DIR, "anomaly_label.csv")
COLLECTION_NAME = "fault_knowledge"
EMBEDDING_SIZE = 768  # FakeEmbeddings 维度(仅无 API Key 时的降级路径)

# 智谱 Embedding 配置(构建与查询共用同一模型, 保证向量空间一致)
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "").strip()
ZHIPU_EMBED_MODEL = os.getenv("ZHIPU_EMBED_MODEL", "embedding-3")

# HDFS 构建参数
MAX_LOG_LINES_PER_BLOCK = 150   # 单个 block 文档保留的最大日志行数(控制 token 长度)
MAX_BLOCKS_PER_LABEL = 300      # 每个标签(正常/异常)抽样上限, 控制构建规模与 API 成本
RANDOM_SEED = 42                # 抽样固定种子, 保证可复现
EMBED_BATCH_SIZE = 32           # 智谱 embedding 批量大小


def _resolve_chroma_dir() -> str:
    raw = os.getenv("CHROMA_PERSIST_DIR", "rag/data/chroma").strip()
    return raw if os.path.isabs(raw) else os.path.join(PROJECT_ROOT, raw)


CHROMA_DIR = _resolve_chroma_dir()

# ---------- 内置故障知识库(仅作为 HDFS 数据缺失时的降级数据源) ----------
BUILTIN_KNOWLEDGE: List[dict] = [
    {
        "id": "kb_001",
        "fault_type": "CPU 使用率过高",
        "symptom_keywords": ["cpu", "load", "高负载", "使用率", "top", "进程", "死循环"],
        "diagnosis": "CPU 使用率持续飙高通常由死循环、高并发请求、GC 频繁、异常重试或挖矿/恶意进程导致。先用 top 定位占用最高的进程, 再结合线程栈或火焰图定位热点代码。",
        "solution": "1. top 按 CPU 排序定位进程; 2. Java 应用执行 jstack 抓线程栈、Python 应用用 py-spy dump; 3. 检查是否高并发流量或定时任务叠加; 4. 优化热点代码、增加限流或扩容。",
        "suggested_commands": [
            "top -c",
            "vmstat 1 5",
            "ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%cpu | head",
            "jstack <pid> | grep -A 30 'java.lang.Thread.State'",
            "py-spy dump --pid <pid>",
        ],
    },
    {
        "id": "kb_002",
        "fault_type": "内存溢出 OOM",
        "symptom_keywords": ["oom", "outofmemory", "内存溢出", "内存不足", "java heap", "memoryerror", "heap space"],
        "diagnosis": "OOM 常见于堆内存不足、大对象/集合无释放、连接/缓存未回收、元空间或线程栈溢出。Java 进程 OOM 后常伴随频繁 GC、进程重启或 core dump。",
        "solution": "1. 增加 JVM 堆内存并开启 -XX:+HeapDumpOnOutOfMemoryError 导出 dump; 2. 用 MAT/VisualVM 分析堆快照定位大对象; 3. 排查未关闭的连接、流、全局缓存; 4. 设置合理的队列与线程数, 防止积压。",
        "suggested_commands": [
            "jmap -heap <pid>",
            "jmap -dump:format=b,file=heap.hprof <pid>",
            "jstat -gcutil <pid> 1000",
            "free -h",
            "ps -eo pid,rss,cmd --sort=-rss | head",
        ],
    },
    {
        "id": "kb_003",
        "fault_type": "磁盘空间不足",
        "symptom_keywords": ["磁盘", "空间不足", "no space left", "disk full", "inode", "日志增长"],
        "diagnosis": "磁盘写满通常由日志/临时文件无限增长、数据库 WAL 膨胀、Docker 镜像残留或大文件上传导致。写满后服务会出现 No space left on device 类报错并中断。",
        "solution": "1. df -h 与 df -i 定位挂载点; 2. du 逐层找出大目录并清理过期日志/临时文件; 3. 对日志配置滚动策略(如 logrotate、按大小切分); 4. 为 /var/log 等目录设置容量告警。",
        "suggested_commands": [
            "df -h",
            "df -i",
            "du -sh /* 2>/dev/null | sort -rh | head",
            "find /var/log -type f -size +100M -exec ls -lh {} \\;",
            "lsof -n | grep deleted",
        ],
    },
    {
        "id": "kb_004",
        "fault_type": "数据库连接池耗尽",
        "symptom_keywords": ["连接池", "connection pool", "too many connections", "连接数", "wait timeout", "无法获取连接"],
        "diagnosis": "连接池耗尽通常由慢 SQL 长时间占用连接、连接泄漏(未归还)、并发突增超过池上限或数据库 max_connections 配置过小导致。表现为接口批量超时与 Connection is not available 报错。",
        "solution": "1. 查看数据库当前连接数与来源: SHOW PROCESSLIST; 2. 排查慢 SQL 并加索引/改写; 3. 检查应用连接池配置(最大连接数、空闲回收)与事务内 try-finally 归还; 4. 适当扩大连接池并压测验证。",
        "suggested_commands": [
            "SHOW PROCESSLIST;",
            "SHOW VARIABLES LIKE 'max_connections';",
            "SELECT COUNT(*) FROM information_schema.processlist WHERE user='app';",
            "EXPLAIN <慢SQL>;",
        ],
    },
    {
        "id": "kb_005",
        "fault_type": "数据库死锁 / 锁等待超时",
        "symptom_keywords": ["死锁", "deadlock", "lock wait", "锁等待", "lock timeout", "事务冲突"],
        "diagnosis": "死锁由多个事务以不同顺序持有并申请相同资源造成; 锁等待超时通常由长事务、大范围更新或未加索引导致行锁放大。数据库会回滚牺牲者事务并记录 Deadlock found 日志。",
        "solution": "1. 从错误日志提取死锁涉及的表与 SQL; 2. 统一事务内多表操作顺序; 3. 缩小事务范围、避免长事务与全表更新; 4. 对高频更新字段加索引减少锁冲突, 必要时调整 innodb_lock_wait_timeout。",
        "suggested_commands": [
            "SHOW ENGINE INNODB STATUS\\G",
            "SELECT * FROM information_schema.innodb_trx;",
            "SELECT * FROM information_schema.innodb_lock_waits;",
            "EXPLAIN <问题SQL>;",
        ],
    },
    {
        "id": "kb_006",
        "fault_type": "端口被占用 / 服务启动失败",
        "symptom_keywords": ["端口", "port", "address already in use", "启动失败", "bind", "无法启动"],
        "diagnosis": "服务启动失败最常见原因是端口被其他进程占用、配置文件错误或依赖服务未就绪。Address already in use 表明旧实例未退出或端口被其他程序抢占。",
        "solution": "1. 查看端口占用进程并确认是否为旧实例; 2. 确认服务配置的端口/依赖地址; 3. 重启前先优雅停止旧进程再启动; 4. 检查启动日志中的配置解析与依赖连接错误。",
        "suggested_commands": [
            "netstat -ano | findstr :8000",
            "tasklist | findstr <pid>",
            "ss -lntp | grep 8000",
            "lsof -i :8000",
        ],
    },
    {
        "id": "kb_007",
        "fault_type": "网络连接拒绝 / 超时",
        "symptom_keywords": ["connection refused", "连接拒绝", "网络超时", "timeout", "unreachable", "connect timed out"],
        "diagnosis": "Connection refused 说明目标端口未监听(服务未启动或监听地址不符); 超时说明目标可达但响应慢或被防火墙/安全组拦截。常见于服务迁移、负载均衡后端失联与跨网段访问。",
        "solution": "1. 确认目标服务进程与监听地址(是否只监听 127.0.0.1); 2. 用 telnet/curl 从发起方验证连通性; 3. 检查防火墙、安全组与网络路由; 4. 排查 DNS 解析与超时参数(connect/read timeout)。",
        "suggested_commands": [
            "curl -v telnet://<host>:<port>",
            "telnet <host> <port>",
            "ping -n 4 <host>",
            "tracert <host>",
            "netstat -ano | findstr LISTENING",
        ],
    },
    {
        "id": "kb_008",
        "fault_type": "Nginx 502 / 504",
        "symptom_keywords": ["502", "504", "bad gateway", "gateway timeout", "upstream", "nginx"],
        "diagnosis": "502 Bad Gateway 通常由后端服务崩溃/未启动或端口不通导致; 504 Gateway Timeout 由后端处理超时或上游连接队列积压导致。常见于后端重启、内存溢出 OOM、慢接口拖垮 upstream。",
        "solution": "1. 检查后端服务进程与健康检查接口; 2. 查看后端应用日志定位崩溃原因; 3. 检查 Nginx error.log 中 upstream 相关报错; 4. 调整 proxy_read_timeout、增加 upstream 实例或做超时降级。",
        "suggested_commands": [
            "tail -f /var/log/nginx/error.log",
            "curl -I http://127.0.0.1:<backend_port>/health",
            "systemctl status <backend-service>",
            "nginx -t",
        ],
    },
    {
        "id": "kb_009",
        "fault_type": "线程池任务拒绝",
        "symptom_keywords": ["rejectedexecution", "线程池", "任务拒绝", "队列", "饱和", "executor"],
        "diagnosis": "线程池拒绝任务说明核心线程与队列均已饱和: 常见于突发流量、任务执行耗时过长或线程池参数(核心数/队列长度/拒绝策略)配置不合理。拒绝策略默认抛 RejectedExecutionException。",
        "solution": "1. 确认线程池参数与当前任务积压量; 2. 定位任务耗时: 是否依赖外部服务/数据库慢查询; 3. 根据流量特性调大核心线程数或队列容量; 4. 设置合理的拒绝策略(如 CallerRunsPolicy)与告警监控。",
        "suggested_commands": [
            "jstack <pid> | grep -A 20 'RejectedExecutionException'",
            "jstat -gcutil <pid> 1000",
            "thread dump 分析线程池状态",
        ],
    },
    {
        "id": "kb_010",
        "fault_type": "空指针异常 NPE",
        "symptom_keywords": ["nullpointer", "npe", "空指针", "null", "空对象"],
        "diagnosis": "NPE 由代码对 null 对象调用方法/访问字段引发: 常见于外部接口返回 null、数据库查询无结果、缓存未命中后未判空、Map 取值无默认值等场景。日志会给出具体类名与行号。",
        "solution": "1. 依据堆栈中的类名与行号定位代码; 2. 检查数据来源: 外部接口/DB 查询/缓存是否可能返回 null; 3. 使用 Optional、判空或默认值兜底; 4. 对关键路径补充参数校验与降级策略。",
        "suggested_commands": [
            "grep -n 'NullPointerException' app.log",
            "jstack <pid> | head -100",
            "结合日志时间点回查上游接口返回值",
        ],
    },
]


# ---------- 内置知识加载(降级用) ----------
def ensure_knowledge_file() -> str:
    """首次运行时把内置知识落盘为 JSON。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(KNOWLEDGE_FILE):
        with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as fh:
            json.dump(BUILTIN_KNOWLEDGE, fh, ensure_ascii=False, indent=2)
    return KNOWLEDGE_FILE


def load_knowledge() -> List[dict]:
    """加载故障知识库(优先读取 JSON 文件)。"""
    ensure_knowledge_file()
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else BUILTIN_KNOWLEDGE


def build_doc_text(item: dict) -> str:
    """将知识条目结构化为检索文本。"""
    return (
        f"故障类型: {item['fault_type']}\n"
        f"症状关键词: {'、'.join(item['symptom_keywords'])}\n"
        f"诊断分析: {item['diagnosis']}\n"
        f"解决方案: {item['solution']}\n"
        f"排查命令: {'、'.join(item['suggested_commands'])}"
    )


# ---------- HDFS 数据集解析 ----------
# HDFS.log 行格式: "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_... src: ..."
_HDFS_LINE_RE = re.compile(
    r"^(?P<date>\d{6})\s+(?P<time>\d{6})\s+(?:\d+\s+)?"
    r"(?P<level>INFO|WARN|ERROR|FATAL|DEBUG|TRACE)\s+"
    r"(?P<component>[^:]+?):\s?(?P<message>.*)$"
)
_BLK_RE = re.compile(r"blk_(-?\d+)")


def _hdfs_files_exist() -> bool:
    return os.path.exists(HDFS_LOG_FILE) and os.path.exists(HDFS_LABEL_FILE)


def load_hdfs_logs() -> dict:
    """流式读取并清洗 HDFS.log, 按 block_id 聚合日志行。

    返回: {block_id: {"lines": [...], "levels": Counter, "start": ts, "end": ts}}
    清洗规则: 丢弃无法解析的行、无 blk_ 标识的行; 每个 block 最多保留 MAX_LOG_LINES_PER_BLOCK 行文本。
    """
    if not os.path.exists(HDFS_LOG_FILE):
        return {}
    blocks: dict = {}
    total = parsed = dropped = 0
    print(f"[HDFS] 开始读取 {HDFS_LOG_FILE} (请耐心等待, 约 1100 万行)...")
    with open(HDFS_LOG_FILE, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            total += 1
            m = _HDFS_LINE_RE.match(line)
            if not m:
                dropped += 1
                continue
            blk = _BLK_RE.search(line)
            if not blk:
                dropped += 1
                continue
            block_id = f"blk_{blk.group(1)}"
            ts = f"{m.group('date')} {m.group('time')}"
            level = m.group("level")
            message = f"{m.group('component')}: {m.group('message').strip()}"

            item = blocks.setdefault(block_id, {"lines": [], "levels": Counter(), "start": ts, "end": ts})
            if len(item["lines"]) < MAX_LOG_LINES_PER_BLOCK:
                item["lines"].append(message)
            item["levels"][level] += 1
            item["end"] = ts
            parsed += 1

    print(f"[HDFS] 日志行: {total:,} | 有效解析: {parsed:,} | 丢弃: {dropped:,} | 独立 block: {len(blocks):,}")
    return blocks


def load_hdfs_labels() -> dict:
    """读取 anomaly_label.csv -> {block_id: label}, label 归一化为 Normal/Anomaly。"""
    if not os.path.exists(HDFS_LABEL_FILE):
        return {}
    labels: dict = {}
    with open(HDFS_LABEL_FILE, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i == 0:  # 表头
                continue
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            bid = parts[0].strip()
            lab = parts[1].strip()
            labels[bid] = "Anomaly" if lab.lower() == "anomaly" else "Normal"
    print(f"[HDFS] 标签条目: {len(labels):,}")
    return labels


def build_hdfs_documents(blocks: dict, labels: dict, max_per_label: int = MAX_BLOCKS_PER_LABEL) -> List[Document]:
    """按标签均匀抽样构造检索文档(每个 block 一条, 含标签与日志摘要)。

    文档结构对检索友好: block_id / 异常标签 / 日志级别统计 / 时间范围 / 日志内容片段,
    查询方(用户日志)与正常/异常样本做相似度匹配。
    """
    normal_ids = [bid for bid, lab in labels.items() if lab == "Normal" and bid in blocks]
    anomaly_ids = [bid for bid, lab in labels.items() if lab == "Anomaly" and bid in blocks]

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(normal_ids)
    rng.shuffle(anomaly_ids)

    picked = normal_ids[:max_per_label] + anomaly_ids[:max_per_label]
    docs: List[Document] = []
    for bid in picked:
        item = blocks[bid]
        label = labels[bid]
        level_summary = ", ".join(f"{k}={v}" for k, v in sorted(item["levels"].items()))
        text = (
            f"[block_id] {bid}\n"
            f"[异常标签] {label}\n"
            f"[日志级别统计] {level_summary}\n"
            f"[日志时间范围] {item['start']} ~ {item['end']}\n"
            f"[日志条数] {sum(item['levels'].values())}\n"
            f"[日志内容]\n" + "\n".join(item["lines"])
        )
        docs.append(Document(page_content=text, metadata={"block_id": bid, "label": label}))

    print(f"[HDFS] 抽样文档: 正常 {len(normal_ids[:max_per_label])} 条 / 异常 {len(anomaly_ids[:max_per_label])} 条")
    return docs


# ---------- 智谱 Embedding 封装 ----------
class ZhipuEmbeddings:
    """智谱 Embedding 接口封装(embedding-3, 构建与查询共用)。"""

    def __init__(self, api_key: str, model: str = ZHIPU_EMBED_MODEL):
        self.model = model
        self._client = None
        if api_key:
            from zhipuai import ZhipuAI

            self._client = ZhipuAI(api_key=api_key)

    @property
    def available(self) -> bool:
        return self._client is not None

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量向量化(按 EMBED_BATCH_SIZE 分批); 批量失败自动退化为逐条截断重试。"""
        vectors: List[List[float]] = []
        total = len(texts)
        for start in range(0, total, EMBED_BATCH_SIZE):
            batch = texts[start:start + EMBED_BATCH_SIZE]
            try:
                vectors.extend(self._embed_batch(batch))
            except Exception:
                # 批量失败(单条超长或接口限制) -> 逐条截断重试
                for t in batch:
                    vectors.append(self._embed_one(t))
            print(f"[Embedding] 进度 {min(start + EMBED_BATCH_SIZE, total)}/{total} 条")
        return vectors

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        ordered = sorted(resp.data, key=lambda x: x.index)
        return [d.embedding for d in ordered]

    def _embed_one(self, text: str) -> List[float]:
        """单条向量化; 文本超过接口上限导致失败时逐级截断重试(4096 -> 3000 字符)。

        实测: embedding-3 单条 input 存在字符上限(约 4000+), 超限返回 400/1210;
        截断到 3000 字符以内必然成功(文档头部含 block_id/标签/级别统计, 从头部截断保留关键信息)。
        """
        last_exc = None
        for candidate in (text, text[:4096], text[:3000]):
            try:
                resp = self._client.embeddings.create(model=self.model, input=candidate)
                return resp.data[0].embedding
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(f"向量化失败(文本长度 {len(text)}): {last_exc}") from last_exc

    def embed_query(self, text: str) -> List[float]:
        resp = self._client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding


_zhipu_emb = ZhipuEmbeddings(ZHIPU_API_KEY)
_fake_emb = FakeEmbeddings(size=EMBEDDING_SIZE)


def _embed_documents(texts: List[str]) -> List[List[float]]:
    """统一向量化入口: 有智谱 Key 走真实 Embedding, 否则 FakeEmbeddings 降级。"""
    if _zhipu_emb.available:
        return _zhipu_emb.embed_texts(texts)
    print("[warn] ZHIPU_API_KEY 未配置, 使用 FakeEmbeddings 降级(仅演示)")
    return _fake_emb.embed_documents(texts)


def _embed_query(query: str) -> List[float]:
    if _zhipu_emb.available:
        return _zhipu_emb.embed_query(query)
    return _fake_emb.embed_query(query)


# ---------- 向量库管理(chromadb 原生 SDK) ----------
_collection = None


def get_collection():
    """获取 Chroma 集合(惰性初始化 + 进程内单例)。"""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _reset_collection():
    """删除旧集合并重建(用于替换内置模拟数据)。"""
    global _collection
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _collection = None
    return get_collection()


def _index_builtin_knowledge(coll) -> None:
    """降级: 无 HDFS 数据文件时灌入内置故障知识(原版行为)。"""
    knowledge = load_knowledge()
    ids = [item["id"] for item in knowledge]
    texts = [build_doc_text(item) for item in knowledge]
    metadatas = [{"fault_type": item["fault_type"]} for item in knowledge]
    coll.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=_embed_documents(texts))
    print(f"[内置知识] 已灌入 {len(knowledge)} 条(降级模式)")


def get_vectorstore():
    """预热接口(FastAPI 启动时调用): 确保集合可用。

    - HDFS 向量库已构建: 直接加载;
    - HDFS 数据文件存在但向量库为空: 提示先执行 python rag/data_loader.py;
    - 无 HDFS 数据文件: 灌入内置故障知识降级, 保持演示可用。
    """
    coll = get_collection()
    if coll.count() == 0:
        if _hdfs_files_exist():
            print("[提示] 检测到 HDFS 数据集但向量库为空, 请先执行: python rag/data_loader.py")
        else:
            _index_builtin_knowledge(coll)
    return coll


# ---------- 混合检索 ----------
def _keyword_overlap(query: str, text: str) -> float:
    """关键词重合度打分: 英数 token(双向匹配) + 中文 3~4 字窗口, 命中越多分越高。"""
    q = query.lower()
    t = text.lower()
    score = 0.0

    # 英文/数字 token(如 blk、ERROR、OOM), 双向匹配提升召回
    t_tokens = set(re.findall(r"[a-z0-9][a-z0-9_.]{1,20}", t))
    q_tokens = set(re.findall(r"[a-z0-9][a-z0-9_.]{1,20}", q))
    score += sum(1.0 for tok in t_tokens if tok in q)
    score += sum(1.0 for tok in q_tokens if tok in t)

    # 中文 n-gram 窗口(3/4 字, 2 字窗口区分度太低已剔除)
    for n in (4, 3):
        for i in range(len(q) - n + 1):
            if q[i:i + n] in t:
                score += 1.0
    return score


def _keyword_only_retrieve(coll, query: str, top_k: int) -> List[Document]:
    """降级检索: 智谱 Embedding 不可用(余额/网络)时, 纯关键词打分召回。

    与 hybrid_retrieve 共用同一关键词打分逻辑, 仅按关键词重合度排序,
    得分大于 0 的文档才会返回, 保证召回结果与查询相关。
    """
    res = coll.get(include=["documents"])
    texts = res.get("documents") or []
    scored = sorted(
        ((_keyword_overlap(query, t), t) for t in texts if _keyword_overlap(query, t) > 0),
        key=lambda x: x[0],
        reverse=True,
    )
    return [Document(page_content=t, metadata={}) for _, t in scored[:top_k]]


def hybrid_retrieve(query: str, top_k: int = 3) -> List[Document]:
    """混合检索: 向量相似度(30%) + 关键词重合度(70%) 加权排序。

    说明:
        - 真实 Embedding(智谱 embedding-3)下向量分可靠, 可调高向量权重;
        - 两个分数均在候选集内做 min-max 归一化后再加权, 避免量纲差异导致排序失真;
        - 智谱 Embedding 调用失败(如额度耗尽/网络异常)时, 自动降级为纯关键词检索, 保证服务可用。
    """
    if not query or not query.strip():
        return []

    vec_weight, kw_weight = 0.3, 0.7
    coll = get_collection()
    if coll.count() == 0:
        return []

    try:
        vec = _embed_query(query)
    except Exception as exc:
        print(f"[warn] 智谱 Embedding 查询失败({exc}), 降级为纯关键词检索")
        return _keyword_only_retrieve(coll, query, top_k)

    res = coll.query(
        query_embeddings=[vec],
        n_results=top_k * 4,
        include=["documents", "distances"],
    )
    docs = (res.get("documents") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    if not docs:
        return []

    # 向量分数归一化: cosine 距离越小越相似 -> vec_score 越大
    d_min, d_max = min(dists), max(dists)
    d_span = (d_max - d_min) or 1.0

    # 关键词分数归一化
    kw_scores = [_keyword_overlap(query, doc) for doc in docs]
    k_min, k_max = min(kw_scores), max(kw_scores)
    k_span = (k_max - k_min) or 1.0

    scored: List[tuple] = []
    for doc, dist, kw in zip(docs, dists, kw_scores):
        vec_score = 1.0 - (dist - d_min) / d_span
        kw_score = (kw - k_min) / k_span
        hybrid = vec_weight * vec_score + kw_weight * kw_score
        scored.append((hybrid, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [Document(page_content=text, metadata={}) for _, text in scored[:top_k]]


# ---------- HDFS 向量库构建 ----------
def build_hdfs_vectorstore() -> dict:
    """基于 Kaggle HDFS 数据集构建向量知识库(替换旧数据)。

    流程: 日志解析清洗 -> block 分块 -> 标签合并 -> 智谱 Embedding -> Chroma 入库。
    返回构建统计信息。
    """
    start = time.time()
    if not _hdfs_files_exist():
        missing = [f for f in (HDFS_LOG_FILE, HDFS_LABEL_FILE) if not os.path.exists(f)]
        raise FileNotFoundError(f"HDFS 数据集文件缺失: {missing}, 请放入 {DATA_DIR}")

    # 智谱 Embedding 探活
    if _zhipu_emb.available:
        probe = _zhipu_emb.embed_query("探活")
        print(f"[智谱 Embedding] API 连通 OK, 模型={ZHIPU_EMBED_MODEL}, 向量维度={len(probe)}")
    else:
        print("[警告] ZHIPU_API_KEY 未配置, 将使用 FakeEmbeddings 降级构建(向量质量有限)")

    blocks = load_hdfs_logs()
    labels = load_hdfs_labels()
    docs = build_hdfs_documents(blocks, labels)
    if not docs:
        raise RuntimeError("抽样后无可用文档, 请检查 HDFS 数据集格式")

    # 重建集合(清空旧的模拟数据)
    coll = _reset_collection()

    # 向量化: 逐条调用智谱 Embedding(_embed_one 内部自动截断重试, 规避单条字符上限)
    ok_pairs = []  # [(doc, vector)]
    for i, doc in enumerate(docs):
        try:
            vec = _zhipu_emb._embed_one(doc.page_content)
            ok_pairs.append((doc, vec))
        except Exception as exc:
            print(f"[warn] 跳过文档 {doc.metadata['block_id']} (长度 {len(doc.page_content)}): {exc}")
        if (i + 1) % 25 == 0 or i == len(docs) - 1:
            print(f"[Embedding] 进度 {i + 1}/{len(docs)} 条 (成功 {len(ok_pairs)})")
    if not ok_pairs:
        raise RuntimeError("全部文档向量化失败, 请检查智谱 API Key/额度/网络")

    docs = [d for d, _ in ok_pairs]
    vectors = [v for _, v in ok_pairs]
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    ids = [d.metadata["block_id"] for d in docs]
    coll.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=vectors)
    elapsed = round(time.time() - start, 1)

    label_counter = Counter(m["label"] for m in metadatas)
    stats = {
        "total_log_lines": sum(sum(v["levels"].values()) for v in blocks.values()),
        "blocks": len(blocks),
        "documents": len(docs),
        "label_dist": dict(label_counter),
        "embed_dim": len(vectors[0]) if vectors else 0,
        "elapsed_sec": elapsed,
    }
    print(f"[构建完成] 耗时 {elapsed}s | 向量维度 {stats['embed_dim']} | 标签分布 {stats['label_dist']}")
    return stats


if __name__ == "__main__":
    print("=" * 60)
    print("开始构建 Kaggle HDFS 日志故障向量知识库")
    print("=" * 60)
    try:
        stats = build_hdfs_vectorstore()
        print("=" * 60)
        print("向量库已使用Kaggle HDFS数据集构建完毕")
        print(f"  - 原始日志行: {stats['total_log_lines']:,}")
        print(f"  - 独立 block: {stats['blocks']:,}")
        print(f"  - 入库文档数: {stats['documents']:,} (标签分布 {stats['label_dist']})")
        print(f"  - 向量维度:   {stats['embed_dim']}")
        print(f"  - 构建耗时:   {stats['elapsed_sec']}s")
        print("=" * 60)
    except Exception as exc:
        print(f"[构建失败] {exc}")
        sys.exit(1)
