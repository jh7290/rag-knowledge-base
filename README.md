# 基于 RAG 的知识库问答系统

这是一个完整可运行的私有知识库问答项目，支持上传文档、文本分块、检索召回、引用来源展示，以及可选的大模型生成回答。

## 功能

- 上传 `.txt`、`.md`、`.pdf` 文档
- 自动清洗文本并分块
- 基于 Python 实现 TF-IDF + 余弦相似度本地检索
- 返回答案引用来源和匹配片段
- 配置 `AI_API_KEY` 后调用 OpenAI 兼容接口生成答案
- 无 API Key 时自动使用本地抽取式回答，方便演示

## 启动

```bash
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 5180
```

打开：

```text
http://localhost:5180
```

## 环境配置

复制 `.env.example` 为 `.env`，填写：

```env
AI_BASE_URL=https://api.openai.com/v1
AI_API_KEY=你的密钥
AI_MODEL=gpt-4o-mini
```

DeepSeek 示例：

```env
AI_BASE_URL=https://api.deepseek.com
AI_API_KEY=你的 DeepSeek Key
AI_MODEL=deepseek-chat
```

## 技术栈

- Python
- FastAPI
- Uvicorn
- Pydantic
- pypdf
- HTML
- CSS
- JavaScript
- TF-IDF
- 余弦相似度
- RAG
- OpenAI-Compatible API
- dotenv

## 接口

- `GET /api/health`
- `GET /api/documents`
- `POST /api/upload`
- `POST /api/ask`
- `POST /api/reset`
