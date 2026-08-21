"""
📌 RAG Knowledge Base — 评测脚本（评测集 + LLM-as-a-Judge）
==============================================
目标：给 RAG 问答系统补上"评测闭环"，量化检索质量和生成质量。

评测集覆盖三类场景（对应简历）：
    - 事实准确：回答是否事实正确、无幻觉
    - 引用正确：回答的关键论断能否在上下文中找到依据
    - 拒答/无关：上下文中无答案时，系统是否正确拒答

两段式评测：
    1. 检索评测（离线、无需 API Key）
       - recall@k / hit@k：目标片段是否被召回进 top-k（仅统计非拒答条目）
    2. 生成评测（LLM-as-a-Judge，无 Key 时走离线启发式）
       - faithfulness（忠实性）、relevance（相关性）、citation_accuracy（引用准确性）

面试可讲点：
    - "我为 RAG 建了评测集，分检索侧(recall@k)和生成侧(忠实性/相关性/引用准确性)两层评测"
    - "评测集按场景分类，拒答场景单独验证系统是否该拒答就拒答、不编造"

运行：
    python eval_rag.py
    结果保存到 data/rag_eval_report.md 与 data/rag_eval_result.json

说明：检索逻辑是 app.py 中 split_text/tokenize/vector_search 的等价实现，
     独立脚本便于在 CI 或无 FastAPI 环境直接跑；修改 app.py 检索逻辑时需同步。
"""

import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Windows 控制台默认 GBK，打印 emoji 会崩；统一走 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "rag_eval_dataset.json"
RESULT_PATH = BASE_DIR / "data" / "rag_eval_result.json"
REPORT_PATH = BASE_DIR / "data" / "rag_eval_report.md"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
JUDGE_CONFIGURED = bool(DEEPSEEK_API_KEY)

TOP_K = 3  # 检索召回条数，与 app.py 的 limit 保持一致


# ---------- 检索实现（与 app.py 等价） ----------
def split_text(text: str, size: int = 850, overlap: int = 130) -> list[str]:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        natural_break = text.rfind("\n", start, end)
        if natural_break > start + size * 0.55:
            end = natural_break
        content = text[start:end].strip()
        if len(content) > 20:
            chunks.append(content)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9_]+", text.lower())


def vector_search(question: str, chunks: list[dict], limit: int = TOP_K) -> list[dict]:
    query_tokens = tokenize(question)
    if not query_tokens or not chunks:
        return []
    doc_tokens = [tokenize(c["content"]) for c in chunks]
    doc_tf = [Counter(t) for t in doc_tokens]
    query_tf = Counter(query_tokens)
    df = Counter()
    for tokens in doc_tokens:
        df.update(set(tokens))

    def weight(token: str, count: int) -> float:
        idf = math.log((len(chunks) + 1) / (df.get(token, 0) + 1)) + 1
        return count * idf

    scored = []
    for chunk, tf in zip(chunks, doc_tf):
        dot = q_norm = d_norm = 0.0
        for token, count in query_tf.items():
            q_w = weight(token, count)
            d_w = weight(token, tf.get(token, 0))
            dot += q_w * d_w
            q_norm += q_w * q_w
        for token, count in tf.items():
            d_w = weight(token, count)
            d_norm += d_w * d_w
        score = dot / (math.sqrt(q_norm) * math.sqrt(d_norm) or 1)
        if score > 0:
            scored.append({**chunk, "score": score})
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:limit]


# ---------- 生成评测（LLM-as-a-Judge） ----------
def judge_generation(item: dict) -> dict:
    expect_refusal = item.get("expect_refusal", False)

    system_prompt = """你是严谨的 RAG 评测专家。请依据【上下文】评测【AI回答】的生成质量，每项 1-10 分：
- faithfulness（忠实性）：回答是否只基于上下文，有没有编造上下文之外的信息
- relevance（相关性）：回答是否直接针对问题，没有答非所问
- citation_accuracy（引用准确性）：回答中的每个关键论断是否都能在上下文中找到依据

输出 JSON：{"faithfulness": {"score": 分, "reason": "理由"}, "relevance": {...}, "citation_accuracy": {...}, "total": 加权平均分, "summary": "评价"}"""
    if expect_refusal:
        system_prompt += "\n\n注意：本题是「拒答/无关」场景，上下文中不含答案，AI 的正确行为是明确拒答（如“资料不足”）；若 AI 编造答案应判低分。"

    user_prompt = f"""【问题】{item['question']}

【上下文】{item['context']}

【AI回答】{item['candidate_answer']}

请逐项评分。"""

    if not JUDGE_CONFIGURED:
        if expect_refusal:
            refused = any(k in item["candidate_answer"] for k in ("不足", "无法", "没有", "未", "抱歉", "不能"))
            score = 9 if refused else 4
            return {
                "faithfulness": {"score": score, "reason": "离线演示：正确拒答则忠实性高，编造则低"},
                "relevance": {"score": score, "reason": "离线演示：正确拒答则相关性高"},
                "citation_accuracy": {"score": score, "reason": "离线演示：正确拒答则引用准确性高"},
                "total": score,
                "summary": "离线演示：拒答场景",
            }
        overlap = len(set(tokenize(item["candidate_answer"])) & set(tokenize(item["context"])))
        faithfulness = max(4, min(9, overlap))
        return {
            "faithfulness": {"score": faithfulness, "reason": "离线演示：按候选回答与上下文的词汇重叠度估算"},
            "relevance": {"score": 8, "reason": "离线演示：默认切题"},
            "citation_accuracy": {"score": max(4, min(9, overlap)), "reason": "离线演示：按词汇重叠度估算"},
            "total": round((faithfulness + 8 + max(4, min(9, overlap))) / 3, 1),
            "summary": "离线演示结果",
        }

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.2,
        max_tokens=2048,
    )
    raw = response.choices[0].message.content
    try:
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        return json.loads(raw.strip())
    except Exception:
        return {"total": 0, "summary": "解析失败"}


def main() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    rows = []

    print("=" * 64)
    print(f"  🔍 RAG 评测：{len(dataset)} 条评测集（事实准确/引用正确/拒答无关）")
    print("=" * 64)

    for item in dataset:
        expect_refusal = item.get("expect_refusal", False)

        # 1. 检索评测
        chunks = [{"content": c, "id": i} for i, c in enumerate(split_text(item["context"]))]
        retrieved = vector_search(item["question"], chunks, TOP_K)
        if expect_refusal:
            hit = None  # 拒答场景不计入检索命中
            rank = 0
        else:
            hit = any(item["golden_sentence"] in r["content"] for r in retrieved)
            rank = next((i + 1 for i, r in enumerate(retrieved) if item["golden_sentence"] in r["content"]), 0)

        # 2. 生成评测
        gen = judge_generation(item)

        rows.append({
            "id": item["id"], "scenario": item.get("scenario", ""), "question": item["question"],
            "hit": hit, "rank": rank,
            "generation": gen,
        })
        if expect_refusal:
            print(f"  #{item['id']} [{item['scenario']}] 拒答场景 | 生成总分: {gen.get('total', 0)}")
        else:
            flag = "✅" if hit else "❌"
            print(f"  #{item['id']} [{item['scenario']}] 检索命中: {flag} (rank={rank or '-'}) | 生成总分: {gen.get('total', 0)}")

    # 统计（检索指标只统计非拒答条目）
    retrieval_rows = [r for r in rows if r["hit"] is not None]
    hits = sum(1 for r in retrieval_rows if r["hit"])
    recall_at_k = hits / len(retrieval_rows) if retrieval_rows else 0
    mrr = sum(1 / r["rank"] for r in retrieval_rows if r["rank"]) / len(retrieval_rows) if retrieval_rows else 0
    avg_gen = round(sum(r["generation"].get("total", 0) for r in rows) / len(rows), 2)

    result = {
        "total": len(rows),
        "retrieval": {"recall_at_k": round(recall_at_k, 2), "mrr": round(mrr, 3), "hit": hits, "retrieval_count": len(retrieval_rows)},
        "generation_avg_total": avg_gen,
        "rows": rows,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# RAG 评测报告",
        "",
        f"- 评测集规模：{len(rows)} 条（事实准确 / 引用正确 / 拒答无关）",
        f"- 检索召回 top-{TOP_K}（检索指标仅统计非拒答条目）",
        "",
        "## 一、检索侧指标",
        "",
        f"- **Recall@{TOP_K}**：{recall_at_k*100:.0f}%（{hits}/{len(retrieval_rows)}）",
        f"- **MRR**（平均倒数排名）：{mrr:.3f}",
        "",
        "## 二、生成侧指标（LLM-as-a-Judge）",
        "",
        f"- 平均总分：**{avg_gen}/10**",
        "",
        "| # | 场景 | 问题 | 检索 | 忠实性 | 相关性 | 引用准确性 |",
        "|---|------|------|:---:|:---:|:---:|:---:|",
    ]
    for r in rows:
        g = r["generation"]
        f = g.get("faithfulness", {}).get("score", "-")
        rel = g.get("relevance", {}).get("score", "-")
        cit = g.get("citation_accuracy", {}).get("score", "-")
        if r["hit"] is None:
            retrieval_cell = "拒答"
        else:
            retrieval_cell = "✅" if r["hit"] else "❌"
        lines.append(f"| {r['id']} | {r['scenario']} | {r['question'][:16]} | {retrieval_cell} | {f} | {rel} | {cit} |")
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 64)
    print(f"  🎯 Recall@{TOP_K}: {recall_at_k*100:.0f}% ({hits}/{len(retrieval_rows)}) | MRR: {mrr:.3f}")
    print(f"  📝 生成平均分: {avg_gen}/10")
    print(f"  📄 报告已保存：{REPORT_PATH.relative_to(BASE_DIR)}")
    print("=" * 64)


if __name__ == "__main__":
    main()
