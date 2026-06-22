import io
import json
import math
import os
import re
import shutil
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
INDEX_PATH = DATA_DIR / "index.json"
PUBLIC_DIR = BASE_DIR / "public"

DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="RAG Knowledge Base")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_index() -> dict:
    if not INDEX_PATH.exists():
        return {"documents": [], "chunks": []}

    try:
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"documents": [], "chunks": []}


def save_index(index: dict) -> None:
    INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, size: int = 850, overlap: int = 130) -> list[str]:
    clean = normalize_text(text)
    chunks = []
    start = 0

    while start < len(clean):
        end = min(len(clean), start + size)
        natural_break = clean.rfind("\n", start, end)
        if natural_break > start + size * 0.55:
            end = natural_break

        content = clean[start:end].strip()
        if len(content) > 30:
            chunks.append(content)

        if end >= len(clean):
            break

        start = max(start + 1, end - overlap)

    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9_]+", text.lower())


def vector_search(question: str, chunks: list[dict], limit: int = 5) -> list[dict]:
    query_tokens = tokenize(question)
    if not query_tokens or not chunks:
        return []

    doc_tokens = [tokenize(chunk["content"]) for chunk in chunks]
    doc_tf = [Counter(tokens) for tokens in doc_tokens]
    query_tf = Counter(query_tokens)
    doc_count = len(chunks)
    df = Counter()

    for tokens in doc_tokens:
        df.update(set(tokens))

    def weight(token: str, count: int) -> float:
        idf = math.log((doc_count + 1) / (df.get(token, 0) + 1)) + 1
        return count * idf

    scored = []
    for chunk, tf in zip(chunks, doc_tf):
        dot = 0.0
        q_norm = 0.0
        d_norm = 0.0

        for token, count in query_tf.items():
            q_weight = weight(token, count)
            d_weight = weight(token, tf.get(token, 0))
            dot += q_weight * d_weight
            q_norm += q_weight * q_weight

        for token, count in tf.items():
            d_weight = weight(token, count)
            d_norm += d_weight * d_weight

        score = dot / (math.sqrt(q_norm) * math.sqrt(d_norm) or 1)
        if score > 0:
            scored.append({**chunk, "score": score})

    return sorted(scored, key=lambda item: item["score"], reverse=True)[:limit]


async def extract_text(file: UploadFile, content: bytes) -> str:
    ext = Path(file.filename or "").suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    return content.decode("utf-8", errors="ignore")


def build_messages(question: str, contexts: list[dict]) -> list[dict]:
    context_text = "\n\n".join(
        f"[{index + 1}] {item['documentName']}\n{item['content']}"
        for index, item in enumerate(contexts)
    )
    return [
        {
            "role": "system",
            "content": "你是严谨的知识库问答助手。只能根据给定资料回答；资料不足时直接说明不足。回答要简洁，并在关键结论后标注引用编号。",
        },
        {"role": "user", "content": f"资料：\n{context_text}\n\n问题：{question}"},
    ]


def call_model(question: str, contexts: list[dict]) -> str | None:
    api_key = os.getenv("AI_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv("AI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("AI_MODEL", "gpt-4o-mini")
    payload = json.dumps(
        {"model": model, "messages": build_messages(question, contexts), "temperature": 0.2},
        ensure_ascii=False,
    ).encode("utf-8")
    req = request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )

    try:
        with request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Model request failed: {exc.code} {detail}") from exc

    return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


def local_answer(question: str, contexts: list[dict]) -> str:
    if not contexts:
        return "知识库里没有检索到足够相关的内容。请先上传更相关的文档，或换一个更具体的问题。"

    query_tokens = set(tokenize(question))
    highlights = []

    for context in contexts:
        sentences = [item.strip() for item in re.split(r"(?<=[。！？.!?])\s+|\n+", context["content"]) if item.strip()]
        ranked = sorted(
            sentences,
            key=lambda sentence: sum(1 for token in tokenize(sentence) if token in query_tokens),
            reverse=True,
        )
        if ranked:
            highlights.append(f"- {ranked[0][:220]} [{context['rank']}]")

    return "根据当前知识库，最相关的信息如下：\n" + "\n".join(highlights[:4])


@app.get("/api/health")
def health() -> dict:
    index = load_index()
    return {
        "ok": True,
        "documents": len(index["documents"]),
        "chunks": len(index["chunks"]),
        "modelConfigured": bool(os.getenv("AI_API_KEY")),
    }


@app.get("/api/documents")
def documents() -> dict:
    index = load_index()
    return {"documents": index["documents"], "chunks": len(index["chunks"])}


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一个文档。")

    index = load_index()
    added = []

    for file in files:
        filename = file.filename or "untitled.txt"
        ext = Path(filename).suffix.lower()
        if ext not in {".txt", ".md", ".pdf"}:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型：{filename}")

        content = await file.read()
        text = normalize_text(await extract_text(file, content))
        if len(text) < 30:
            raise HTTPException(status_code=400, detail=f"{filename} 没有提取到足够文本。")

        document_id = str(uuid.uuid4())
        stored_name = f"{document_id}{ext}"
        (UPLOAD_DIR / stored_name).write_bytes(content)
        chunks = [
            {
                "id": str(uuid.uuid4()),
                "documentId": document_id,
                "documentName": filename,
                "chunkIndex": index_,
                "content": chunk,
            }
            for index_, chunk in enumerate(split_text(text))
        ]
        document = {
            "id": document_id,
            "name": filename,
            "storedName": stored_name,
            "size": len(content),
            "chunks": len(chunks),
            "createdAt": utc_now(),
        }
        index["documents"].append(document)
        index["chunks"].extend(chunks)
        added.append(document)

    save_index(index)
    return {"added": added, "totalDocuments": len(index["documents"]), "totalChunks": len(index["chunks"])}


@app.post("/api/ask")
def ask(payload: AskRequest) -> JSONResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空。")

    index = load_index()
    contexts = [
        {**item, "rank": index_ + 1}
        for index_, item in enumerate(vector_search(question, index["chunks"], limit=5))
    ]

    try:
        model_answer = call_model(question, contexts)
        return JSONResponse(
            {
                "answer": model_answer or local_answer(question, contexts),
                "mode": "llm" if model_answer else "local",
                "citations": [
                    {
                        "id": item["id"],
                        "rank": item["rank"],
                        "documentName": item["documentName"],
                        "chunkIndex": item["chunkIndex"],
                        "score": item["score"],
                        "preview": item["content"][:260],
                    }
                    for item in contexts
                ],
            }
        )
    except RuntimeError as exc:
        return JSONResponse(
            status_code=502,
            content={"error": str(exc), "fallback": local_answer(question, contexts), "citations": contexts},
        )


@app.post("/api/reset")
def reset() -> dict:
    save_index({"documents": [], "chunks": []})
    if UPLOAD_DIR.exists():
        for item in UPLOAD_DIR.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    return {"ok": True}


app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")
