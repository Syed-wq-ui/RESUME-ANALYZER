from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import pdfplumber
import os
import time
from dotenv import load_dotenv
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory

from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# =========================================================
# SETTINGS
# =========================================================

class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    PINECONE_INDEX_NAME = "resume-chat"

    UPLOAD_DIR = "uploads"

    # LIGHTWEIGHT MODEL FOR RENDER FREE TIER
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    LLM_MODEL = "gemini-2.5-flash"

    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    @classmethod
    def validate(cls):
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY missing")

        if not cls.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY missing")


settings = Settings()
settings.validate()

# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Resume Chat API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

# =========================================================
# GLOBALS (LAZY LOADED)
# =========================================================

embeddings = None
llm = None
pc = None

vector_store = None
rag_chain = None

message_history = ChatMessageHistory()

# =========================================================
# LAZY LOADERS
# =========================================================

def get_embeddings():
    global embeddings

    if embeddings is None:
        print("Loading embeddings...")

        embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    return embeddings


def get_llm():
    global llm

    if llm is None:
        print("Loading Gemini LLM...")

        llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=0.7,
            google_api_key=settings.GEMINI_API_KEY
        )

    return llm


def get_pinecone():
    global pc

    if pc is None:
        print("Connecting to Pinecone...")

        pc = Pinecone(api_key=settings.PINECONE_API_KEY)

        existing_indexes = pc.list_indexes().names()

        if settings.PINECONE_INDEX_NAME not in existing_indexes:

            print("Creating Pinecone index...")

            pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

    return pc

# =========================================================
# MODELS
# =========================================================

class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int
    processing_time: float


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []
    processing_time: float


# =========================================================
# HELPERS
# =========================================================

def extract_text_from_pdf(file_path: str):

    text = ""

    with pdfplumber.open(file_path) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


# =========================================================
# ROUTES
# =========================================================

@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "vector_store_ready": vector_store is not None,
        "rag_chain_ready": rag_chain is not None
    }


@app.post("/upload-resume", response_model=UploadResponse)
async def upload_resume(file: UploadFile = File(...)):

    global vector_store
    global rag_chain

    start_time = time.time()

    try:

        if not file.filename.endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF supported"
            )

        embeddings = get_embeddings()
        llm = get_llm()
        pc = get_pinecone()

        file_path = os.path.join(
            settings.UPLOAD_DIR,
            file.filename
        )

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        print("Extracting PDF text...")

        resume_text = extract_text_from_pdf(file_path)

        if len(resume_text.strip()) < 100:
            raise HTTPException(
                status_code=400,
                detail="Could not extract enough text"
            )

        print("Splitting text...")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

        chunks = splitter.split_text(resume_text)

        print(f"Created {len(chunks)} chunks")

        index = pc.Index(settings.PINECONE_INDEX_NAME)

        try:
            index.delete(delete_all=True, namespace="resume")
        except:
            pass

        print("Creating vector store...")

        vector_store = PineconeVectorStore.from_texts(
            texts=chunks,
            embedding=embeddings,
            index_name=settings.PINECONE_INDEX_NAME,
            namespace="resume"
        )

        print("Creating RAG chain...")

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
You are an expert resume assistant.

Use the provided resume context to answer accurately.

{context}
"""
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        document_chain = create_stuff_documents_chain(
            llm,
            prompt
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        rag_chain = create_retrieval_chain(
            retriever,
            document_chain
        )

        processing_time = round(
            time.time() - start_time,
            2
        )

        return UploadResponse(
            message="Resume uploaded successfully",
            filename=file.filename,
            chunks_created=len(chunks),
            processing_time=processing_time
        )

    except Exception as e:

        import traceback

        print(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    global rag_chain
    global message_history

    if rag_chain is None:
        raise HTTPException(
            status_code=400,
            detail="Upload resume first"
        )

    start_time = time.time()

    try:

        result = rag_chain.invoke({
            "input": request.message,
            "chat_history": message_history.messages
        })

        answer = result["answer"]

        message_history.add_user_message(request.message)
        message_history.add_ai_message(answer)

        if len(message_history.messages) > 10:
            message_history.messages = message_history.messages[-10:]

        sources = []

        if "context" in result:

            for doc in result["context"][:3]:

                sources.append(
                    doc.page_content[:200] + "..."
                )

        processing_time = round(
            time.time() - start_time,
            2
        )

        return ChatResponse(
            answer=answer,
            sources=sources,
            processing_time=processing_time
        )

    except Exception as e:

        import traceback

        print(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/reset-chat")
async def reset_chat():

    global message_history

    message_history.clear()

    return {
        "message": "Chat reset successful"
    }


@app.delete("/resume")
async def delete_resume():

    global vector_store
    global rag_chain
    global message_history

    try:

        pc = get_pinecone()

        index = pc.Index(settings.PINECONE_INDEX_NAME)

        index.delete(
            delete_all=True,
            namespace="resume"
        )

        vector_store = None
        rag_chain = None

        message_history.clear()

        return {
            "message": "Resume deleted"
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
async def startup_event():

    print("=" * 60)
    print("Resume Chat API Started")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():

    print("Shutting down API...")

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )