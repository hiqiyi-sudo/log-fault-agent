"""evaluate.py - 日志故障诊断 Agent 离线评估脚本

评估流程:
    1. 读取 rag/data/anomaly_label.csv 标签 + rag/data/HDFS.log 原始日志
    2. 均衡抽样 200 条样本(正常/异常各 100, 固定随机种子可复现)
    3. 逐条构造日志文本, 调用项目 RAG 检索(retrieve) + Agent 诊断(diagnose) 链路(多线程并行)
    4. 统计: 故障诊断准确率 / RAG Top3 召回率 / 单条平均诊断耗时
    5. 控制台输出汇总指标 + 生成 eval_result.csv(逐条明细: block_id/真实标签/预测结果/单次耗时)

说明:
    - 复用项目 rag/data_loader 与 graph/nodes 模块, 不修改任何业务代码
    - 评估链路 = retrieve -> diagnose(跳过 record 节点, 避免评估数据写入 SQLite 工单库)
    - 评估输入均为日志样本, 跳过意图识别环节(非评估对象)
    - 预测标签 = RAG Top3 检索文档中 [异常标签] 投票(任一 Anomaly 即判异常)
    - 智谱 Embedding 不可用(余额不足)时自动快速失败进入关键词降级, 避免逐条等待 SDK 重试
    - 多线程并行仅加速整体墙钟, 每条耗时仍独立计时, 指标不受影响

运行: python evaluate.py
"""
import csv
import io
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import rag.data_loader as dl  # noqa: E402
from rag.data_loader import load_hdfs_logs, load_hdfs_labels  # noqa: E402
from graph.nodes import diagnose, retrieve, INTENT_LOG  # noqa: E402

# ---------- 评估配置 ----------
SAMPLE_SIZE = 200            # 总样本数
SAMPLE_PER_LABEL = SAMPLE_SIZE // 2  # 每类样本数
RANDOM_SEED = 7              # 抽样种子(可复现)
TOP_K = 3                    # RAG 检索条数
MAX_INPUT_LINES = 30         # 单条输入日志行数上限
EVAL_WORKERS = 4             # 并行评估线程数
OUTPUT_CSV = "eval_result.csv"

_LABEL_RE = re.compile(r"\[异常标签\]\s+(Normal|Anomaly)")


def build_input_text(item: dict) -> str:
    """由 block 日志聚合数据构造评估输入文本(取前 MAX_INPUT_LINES 行)。"""
    lines = item.get("lines") or []
    return "\n".join(lines[:MAX_INPUT_LINES])


def predict_from_docs(docs) -> tuple:
    """RAG Top3 投票预测: 任一检索文档标签为 Anomaly -> 判异常; 无检索/无标签 -> (None, []).

    注意: graph.nodes.retrieve 返回的 retrieved_docs 为文本列表(str), 直接正则提取标签。
    """
    if not docs:
        return None, []
    labels = []
    for text in docs:
        m = _LABEL_RE.search(text)
        if m:
            labels.append(m.group(1))
    if not labels:
        return None, []
    return ("Anomaly" if "Anomaly" in labels else "Normal"), labels


def _patch_embedding_fast_fail() -> None:
    """智谱 Embedding 不可用(余额不足)时, 跳过 SDK 429 重试等待, 让检索快速进入关键词降级。"""
    if not dl._zhipu_emb.available:
        return
    try:
        dl._zhipu_emb.embed_query("探活")
    except Exception:
        def _fast_fail(_query: str):
            raise RuntimeError("智谱 Embedding 不可用(评估模式快速失败)")
        dl._embed_query = _fast_fail
        print("      [warn] 智谱 Embedding 不可用, 检索降级为纯关键词(已跳过逐条 429 等待)")


_retrieve_lock = threading.Lock()


def eval_one(block_id: str, true_label: str, blocks: dict) -> dict:
    """单条评估: RAG 检索(retrieve) + Agent 诊断(diagnose), 返回明细行。"""
    query = build_input_text(blocks.get(block_id, {}))
    t0 = time.perf_counter()
    # RAG 检索节点(chromadb 查询加锁保证线程安全, 检索耗时极短)
    with _retrieve_lock:
        with redirect_stdout(io.StringIO()):
            state = retrieve({"user_query": query, "intent": INTENT_LOG})
    t1 = time.perf_counter()
    # Agent 诊断节点(GLM 生成或演示模式降级, 网络 IO 可并行)
    with redirect_stdout(io.StringIO()):
        state = diagnose(state)
    t2 = time.perf_counter()

    docs = state.get("retrieved_docs") or []
    pred, ret_labels = predict_from_docs(docs)
    return {
        "block_id": block_id,
        "true_label": true_label,
        "predicted_label": pred if pred else "UNKNOWN",
        "retrieved_labels": "|".join(ret_labels) if ret_labels else "",
        "elapsed_ms": round((t2 - t0) * 1000, 1),
        "retrieve_ms": round((t1 - t0) * 1000, 1),
        "answer": (state.get("final_answer") or "")[:200].replace("\n", " "),
    }


def main() -> None:
    # 确保重定向到文件时输出实时落盘(避免块缓冲导致日志不完整)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass
    print("=" * 60)
    print("日志故障诊断 Agent 离线评估")
    print("=" * 60)

    # [1/4] 加载 HDFS 数据集
    print("[1/4] 加载 HDFS 数据集(约 1100 万行日志, 请耐心等待)...")
    t_load = time.perf_counter()
    blocks = load_hdfs_logs()
    labels = load_hdfs_labels()
    print(f"      加载完成 | 独立 block={len(blocks):,} | 标签={len(labels):,} | 耗时={time.perf_counter() - t_load:.1f}s")

    # [2/4] 均衡抽样
    normal_ids = [b for b, l in labels.items() if l == "Normal" and b in blocks]
    anomaly_ids = [b for b, l in labels.items() if l == "Anomaly" and b in blocks]
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(normal_ids)
    rng.shuffle(anomaly_ids)
    picked = [b for b in normal_ids[:SAMPLE_PER_LABEL]] + [b for b in anomaly_ids[:SAMPLE_PER_LABEL]]
    rng.shuffle(picked)
    true_map = {b: labels[b] for b in picked}
    print(f"[2/4] 抽样样本: {len(picked)} 条 (Normal {SAMPLE_PER_LABEL} / Anomaly {SAMPLE_PER_LABEL}, seed={RANDOM_SEED})")

    # Embedding 可用性探测(失败则快速降级)
    _patch_embedding_fast_fail()

    # [3/4] 并行评估
    print(f"[3/4] 并行评估 {len(picked)} 条 (workers={EVAL_WORKERS})...")
    rows = []
    order = {}
    done = 0
    with ThreadPoolExecutor(max_workers=EVAL_WORKERS) as ex:
        futures = {ex.submit(eval_one, b, true_map[b], blocks): b for b in picked}
        for fut in as_completed(futures):
            b = futures[fut]
            try:
                rows.append(fut.result())
            except Exception as exc:  # 单条异常不中断整体评估
                rows.append({
                    "block_id": b,
                    "true_label": true_map[b],
                    "predicted_label": "ERROR",
                    "retrieved_labels": "",
                    "elapsed_ms": -1,
                    "retrieve_ms": -1,
                    "answer": f"评估异常: {exc}",
                })
            done += 1
            if done % 25 == 0 or done == len(picked):
                print(f"      进度 {done}/{len(picked)}")
    rows.sort(key=lambda r: picked.index(r["block_id"]))  # 恢复抽样顺序, 便于追溯

    # [4/4] 指标计算
    valid = [r for r in rows if r["predicted_label"] in ("Normal", "Anomaly")]
    unknown = len(rows) - len(valid)
    errors = sum(1 for r in rows if r["predicted_label"] == "ERROR")
    correct = sum(r["true_label"] == r["predicted_label"] for r in valid)
    accuracy = correct / len(valid) * 100 if valid else 0.0

    anomaly_rows = [r for r in valid if r["true_label"] == "Anomaly"]
    recall_at3 = sum(r["predicted_label"] == "Anomaly" for r in anomaly_rows) / len(anomaly_rows) * 100 if anomaly_rows else 0.0

    tp = sum(r["true_label"] == "Anomaly" and r["predicted_label"] == "Anomaly" for r in valid)
    fp = sum(r["true_label"] == "Normal" and r["predicted_label"] == "Anomaly" for r in valid)
    fn = sum(r["true_label"] == "Anomaly" and r["predicted_label"] == "Normal" for r in valid)
    tn = sum(r["true_label"] == "Normal" and r["predicted_label"] == "Normal" for r in valid)
    precision = tp / (tp + fp) * 100 if tp + fp else 0.0

    timed = [r for r in rows if r["elapsed_ms"] >= 0]
    avg_retrieve_ms = sum(r["retrieve_ms"] for r in timed) / len(timed) if timed else 0.0
    avg_total_ms = sum(r["elapsed_ms"] for r in timed) / len(timed) if timed else 0.0

    print()
    print("=" * 60)
    print("评估汇总指标")
    print("=" * 60)
    print(f"  样本总数          : {len(rows)} 条 (Normal {SAMPLE_PER_LABEL} / Anomaly {SAMPLE_PER_LABEL})")
    if unknown or errors:
        print(f"  未参与计分        : {unknown + errors} 条 (无检索 {unknown} / 异常 {errors})")
    print(f"  故障诊断准确率    : {accuracy:.2f}%  ({correct}/{len(valid)})")
    print(f"  RAG Top{TOP_K}召回率   : {recall_at3:.2f}%  (真实异常 {len(anomaly_rows)} 条中 Top{TOP_K} 命中异常知识 {int(sum(1 for r in anomaly_rows if r['predicted_label'] == 'Anomaly'))} 条)")
    print(f"  异常精确率        : {precision:.2f}%")
    print(f"  混淆矩阵          : TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  单条平均检索耗时  : {avg_retrieve_ms:.1f} ms")
    print(f"  单条平均诊断耗时  : {avg_total_ms:.1f} ms (检索 + GLM 生成/降级)")
    print(f"  评估墙钟耗时      : {(sum(r['elapsed_ms'] for r in timed) / 1000 / EVAL_WORKERS):.1f} s (并行)")
    print("=" * 60)

    # 写入 eval_result.csv
    fieldnames = ["block_id", "true_label", "predicted_label", "retrieved_labels", "elapsed_ms", "retrieve_ms", "answer"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"逐条明细已保存: {os.path.join(PROJECT_ROOT, OUTPUT_CSV)}")
    print("评估完成。")


if __name__ == "__main__":
    main()
