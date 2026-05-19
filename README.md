<div align="center">

<br/>

```
██████╗ ███████╗███████╗██╗   ██╗███╗   ███╗███████╗     ██████╗██╗  ██╗ █████╗ ████████╗
██╔══██╗██╔════╝██╔════╝██║   ██║████╗ ████║██╔════╝    ██╔════╝██║  ██║██╔══██╗╚══██╔══╝
██████╔╝█████╗  ███████╗██║   ██║██╔████╔██║█████╗      ██║     ███████║███████║   ██║   
██╔══██╗██╔══╝  ╚════██║██║   ██║██║╚██╔╝██║██╔══╝      ██║     ██╔══██║██╔══██║   ██║   
██║  ██║███████╗███████║╚██████╔╝██║ ╚═╝ ██║███████╗    ╚██████╗██║  ██║██║  ██║   ██║   
╚═╝  ╚═╝╚══════╝╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝     ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   
```

**AI-Powered Interview Preparation · RAG Pipeline · Production Ready**

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-1.2.6-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-000000?style=for-the-badge&logo=pinecone&logoColor=white)](https://pinecone.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

> *Upload your resume. Ask anything. Walk into every interview ready.*

<br/>

</div>

---

## ✦ What Is This?

**Resume Chat** is a production-grade **Retrieval-Augmented Generation (RAG)** API that transforms your resume into an interactive interview coach. Powered by **Google Gemini 2.5 Flash**, **Pinecone vector search**, and **LangChain 1.2.6**, it answers interview questions with specific, grounded examples pulled directly from your experience.

No hallucinations. No generic answers. Just *your* story, told compellingly.

---

## ⚡ Features

| Feature | Details |
|---|---|
| 📄 **PDF Ingestion** | Upload any resume PDF — text extracted with `pdfplumber` |
| 🧠 **Semantic Search** | `sentence-transformers/all-mpnet-base-v2` embeddings (768-dim) |
| 🔍 **Vector Store** | Pinecone serverless index with cosine similarity |
| 💬 **Conversational Memory** | Rolling 10-message chat history for contextual follow-ups |
| 🎯 **STAR Method Coaching** | Answers structured as Situation → Task → Action → Result |
| 🔗 **RAG Pipeline** | LangChain retrieval + document chains for grounded responses |
| 🚀 **Production FastAPI** | CORS, static files, structured Pydantic models, error handling |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT                                │
│              (Browser / API Consumer)                        │
└──────────────────────────┬──────────────────────────────────┘
                           │  HTTP
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    FASTAPI SERVER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ POST /upload │  │  POST /chat  │  │  GET /health     │  │
│  │   -resume    │  │              │  │  DELETE /resume  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────────┘  │
└─────────┼─────────────────┼───────────────────────────────-─┘
          │                 │
          ▼                 ▼
┌──────────────────┐  ┌─────────────────────────────────────┐
│  PDF PROCESSING  │  │         RAG CHAIN (LangChain)       │
│                  │  │                                     │
│  pdfplumber      │  │  ┌───────────┐   ┌──────────────┐  │
│       ↓          │  │  │ Retriever │ → │ Stuff Docs   │  │
│  Text Splitter   │  │  │ (k=3 sim) │   │    Chain     │  │
│  500 chunks /    │  │  └───────────┘   └──────┬───────┘  │
│  50 overlap      │  │                         │          │
└────────┬─────────┘  │               ┌─────────▼────────┐ │
         │            │               │  Gemini 2.5 Flash│ │
         ▼            │               └──────────────────┘ │
┌──────────────────┐  └─────────────────────────────────────┘
│    PINECONE      │
│  Vector Store    │
│  (cosine, 768d)  │
│  ns: "resume"    │
└──────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/resume-chat.git
cd resume-chat

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
GEMINI_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

> **Get your keys:**
> - Gemini → [aistudio.google.com](https://aistudio.google.com/app/apikey)
> - Pinecone → [app.pinecone.io](https://app.pinecone.io)

### 3. Run

```bash
python app.py
```

```
🚀 Initializing services with LangChain 1.2.6...
📦 Loading embedding model...
🔌 Connecting to Pinecone...
🤖 Initializing Gemini LLM...
✅ All services initialized!

======================================================================
🎉 Resume Chat API Started Successfully!
======================================================================
📍 API:              http://localhost:8000
📚 Interactive Docs: http://localhost:8000/docs
🔧 LangChain:        1.2.6
🧠 LLM:              gemini-2.5-flash
📊 Vector DB:        Pinecone (resume-chat)
======================================================================
```

---

## 📡 API Reference

### `POST /upload-resume`
Upload and index your resume PDF.

```bash
curl -X POST "http://localhost:8000/upload-resume" \
  -H "accept: application/json" \
  -F "file=@your_resume.pdf"
```

```json
{
  "message": "Resume uploaded and processed successfully!",
  "filename": "your_resume.pdf",
  "chunks_created": 42,
  "processing_time": 3.87
}
```

---

### `POST /chat`
Ask interview questions grounded in your resume.

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about a time you led a team under pressure"}'
```

```json
{
  "answer": "In my role at Acme Corp, I led a 5-person engineering team during a critical product launch...",
  "sources": ["Led cross-functional team of 5 engineers...", "Delivered project 2 weeks ahead..."],
  "processing_time": 1.24
}
```

---

### Other Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service status & readiness |
| `GET` | `/chat-history` | Retrieve conversation history |
| `POST` | `/reset-chat` | Clear conversation memory |
| `DELETE` | `/resume` | Wipe resume & all data |
| `GET` | `/docs` | Interactive Swagger UI |

---

## 💡 Example Prompts

```
"Walk me through your most impactful project"
"What's your experience with Python and backend systems?"
"Describe a conflict you resolved on a team"
"What are your top 3 technical strengths based on my resume?"
"Help me answer: Where do you see yourself in 5 years?"
"What's a weakness I should address given my background?"
```

---

## 🛠️ Tech Stack

```
Backend          FastAPI + Uvicorn
LLM              Google Gemini 2.5 Flash  
Embeddings       sentence-transformers/all-mpnet-base-v2 (HuggingFace)
Vector DB        Pinecone Serverless (AWS us-east-1)
RAG Framework    LangChain 1.2.6
PDF Parsing      pdfplumber
Config           python-dotenv
```

---

## 📁 Project Structure

```
resume-chat/
├── app.py               # Main FastAPI application
├── static/
│   └── index.html       # Frontend UI
├── uploads/             # Temporary PDF storage
├── .env                 # API keys (never commit this)
├── .env.example         # Template for environment setup
├── requirements.txt     # Python dependencies
└── README.md
```

---

## ⚙️ Configuration

All settings live in the `Settings` class inside `app.py`:

| Variable | Default | Description |
|---|---|---|
| `PINECONE_INDEX_NAME` | `resume-chat` | Pinecone index name |
| `EMBEDDING_MODEL` | `all-mpnet-base-v2` | HuggingFace embedding model |
| `LLM_MODEL` | `gemini-2.5-flash` | Google Gemini model |
| `CHUNK_SIZE` | `500` | Characters per text chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |

---

## 🔒 Security Notes

- ✅ API keys loaded from `.env` — never hardcoded
- ✅ Pinecone namespace isolation (`resume` namespace)
- ✅ PDF validation before processing
- ⚠️ CORS set to `*` — restrict `allow_origins` in production
- ⚠️ No auth layer by default — add API key middleware for public deployments

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

```bash
# Development setup
git checkout -b feature/your-feature
# make your changes
git commit -m "feat: your feature description"
git push origin feature/your-feature
```

---

## 📄 License

MIT © 2024 — see [LICENSE](LICENSE) for details.

---

<div align="center">

<br/>

**Built with LangChain · Gemini · Pinecone**

*Stop guessing what to say. Start telling your story.*

</div>