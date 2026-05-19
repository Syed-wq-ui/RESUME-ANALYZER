# app.py - PRODUCTION v1.0 with LangChain 1.2.6
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pdfplumber
import os
from dotenv import load_dotenv
import time
from typing import List

# LangChain 1.2.6 imports - CORRECT for this version
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from pinecone import Pinecone, ServerlessSpec
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

load_dotenv()

# ==================== CONFIGURATION ====================
class Settings:
    """Application settings"""
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = "resume-chat"
    UPLOAD_DIR = "uploads"
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    LLM_MODEL = "gemini-2.5-flash"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    @classmethod
    def validate(cls):
        if not cls.GEMINI_API_KEY:
            raise ValueError("❌ GEMINI_API_KEY not found in .env")
        if not cls.PINECONE_API_KEY:
            raise ValueError("❌ PINECONE_API_KEY not found in .env")

settings = Settings()
settings.validate()

# ==================== FASTAPI APP ====================
app = FastAPI(
    title="Resume Chat API",
    description="Production RAG API with LangChain 1.2.6",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ==================== INITIALIZATION ====================
print("🚀 Initializing services with LangChain 1.2.6...")

# Embeddings
print("📦 Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name=settings.EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Pinecone
print("🔌 Connecting to Pinecone...")
pc = Pinecone(api_key=settings.PINECONE_API_KEY)

if settings.PINECONE_INDEX_NAME not in pc.list_indexes().names():
    print(f"🆕 Creating Pinecone index: {settings.PINECONE_INDEX_NAME}")
    pc.create_index(
        name=settings.PINECONE_INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("✅ Index created!")

# LLM
print("🤖 Initializing Gemini LLM...")
llm = ChatGoogleGenerativeAI(
    temperature=0.7,
    model=settings.LLM_MODEL,
    google_api_key=settings.GEMINI_API_KEY
)

# Global state
vector_store = None
rag_chain = None
message_history = ChatMessageHistory()

print("✅ All services initialized!")

# ==================== PYDANTIC MODELS ====================
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

class HealthResponse(BaseModel):
    status: str
    langchain_version: str
    vector_store_ready: bool
    rag_chain_ready: bool

# ==================== HELPER FUNCTIONS ====================
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        raise Exception(f"PDF extraction failed: {str(e)}")

def check_resume_exists() -> bool:
    """Check if vectors exist in Pinecone"""
    try:
        index = pc.Index(settings.PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        namespace_stats = stats.get('namespaces', {})
        resume_vectors = namespace_stats.get('resume', {}).get('vector_count', 0)
        return resume_vectors > 0
    except:
        return False

# ==================== API ENDPOINTS ====================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint serving the frontend"""
    return FileResponse("static/index.html")

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        langchain_version="1.2.6",
        vector_store_ready=vector_store is not None,
        rag_chain_ready=rag_chain is not None
    )

@app.post("/upload-resume", response_model=UploadResponse, tags=["Resume"])
async def upload_resume(file: UploadFile = File(...)):
    """
    Upload and process resume with LangChain 1.2.6
    
    - Accepts PDF files only
    - Uses RecursiveCharacterTextSplitter
    - Stores in Pinecone with PineconeVectorStore
    - Creates retrieval chain with document chain
    """
    global vector_store, rag_chain
    
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files supported")
        
        # Save file
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
        
        print(f"📁 Saving file: {file.filename}")
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Extract text
        print("📄 Extracting text from PDF...")
        resume_text = extract_text_from_pdf(file_path)
        
        if len(resume_text.strip()) < 100:
            raise HTTPException(
                status_code=400,
                detail="Could not extract sufficient text from PDF"
            )
        
        print(f"✅ Extracted {len(resume_text)} characters")
        
        # Split text with LangChain 1.2.6 text splitter
        print("✂️ Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False
        )
        chunks = text_splitter.split_text(resume_text)
        print(f"✅ Created {len(chunks)} chunks")
        
        # Clear old vectors
        print("🗑️ Clearing old vectors...")
        try:
            index = pc.Index(settings.PINECONE_INDEX_NAME)
            index.delete(delete_all=True, namespace="resume")
        except Exception as e:
            print(f"⚠️ Warning: {e}")
        
        # Create vector store with LangChain 1.2.6
        print("🔢 Creating vector store...")
        vector_store = PineconeVectorStore.from_texts(
            texts=chunks,
            embedding=embeddings,
            index_name=settings.PINECONE_INDEX_NAME,
            namespace="resume"
        )
        print("✅ Vector store created!")
        
        # Create RAG chain with LangChain 1.2.6
        print("🔗 Setting up RAG chain...")
        
        # Create prompt template with chat history support
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert interview preparation coach helping users prepare for job interviews.

Use the following context from the user's resume to provide specific, actionable interview answers:

{context}

Guidelines:
- Provide clear, professional interview answers
- Use the STAR method (Situation, Task, Action, Result) when appropriate
- Give specific examples from the resume
- Keep responses focused and concise
- Help communicate experience effectively"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Create document chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Create retrieval chain
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        rag_chain = create_retrieval_chain(retriever, document_chain)
        
        print("✅ RAG chain ready!")
        
        processing_time = time.time() - start_time
        
        return UploadResponse(
            message="Resume uploaded and processed successfully!",
            filename=file.filename,
            chunks_created=len(chunks),
            processing_time=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"❌ Error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Chat with resume using RAG chain
    
    - Maintains conversation history
    - Retrieves relevant context from resume
    - Generates interview-ready answers
    """
    global rag_chain, message_history
    
    if rag_chain is None:
        raise HTTPException(
            status_code=400,
            detail="Please upload a resume first using /upload-resume"
        )
    
    start_time = time.time()
    
    try:
        print(f"💬 Question: {request.message}")
        
        # Invoke RAG chain with chat history
        result = rag_chain.invoke({
            "input": request.message,
            "chat_history": message_history.messages
        })
        
        answer = result["answer"]
        print(f"✅ Answer generated")
        
        # Extract sources from context
        sources = []
        if "context" in result and result["context"]:
            sources = [
                doc.page_content[:200] + "..." 
                for doc in result["context"][:3]
            ]
        
        # Update message history
        message_history.add_user_message(request.message)
        message_history.add_ai_message(answer)
        
        # Keep only last 10 messages
        if len(message_history.messages) > 10:
            message_history.messages = message_history.messages[-10:]
        
        processing_time = time.time() - start_time
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            processing_time=round(processing_time, 2)
        )
        
    except Exception as e:
        import traceback
        print(f"❌ Error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.post("/reset-chat", tags=["Chat"])
async def reset_chat():
    """Reset conversation history"""
    global message_history
    message_history.clear()
    print("🔄 Chat history reset")
    return {"message": "Chat history reset successfully!"}

@app.get("/chat-history", tags=["Chat"])
async def get_chat_history():
    """Get current chat history"""
    return {
        "messages": [
            {
                "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content
            }
            for msg in message_history.messages
        ],
        "count": len(message_history.messages)
    }

@app.delete("/resume", tags=["Resume"])
async def delete_resume():
    """Delete resume and clear all data"""
    global vector_store, rag_chain, message_history
    
    try:
        # Clear Pinecone vectors
        index = pc.Index(settings.PINECONE_INDEX_NAME)
        index.delete(delete_all=True, namespace="resume")
        
        # Clear state
        vector_store = None
        rag_chain = None
        message_history.clear()
        
        print("🗑️ Resume and chat history deleted")
        
        return {"message": "Resume deleted successfully!"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

# ==================== STARTUP/SHUTDOWN ====================

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    print("\n" + "="*70)
    print("🎉 Resume Chat API Started Successfully!")
    print("="*70)
    print(f"📍 API: http://localhost:8000")
    print(f"📚 Interactive Docs: http://localhost:8000/docs")
    print(f"🔧 LangChain Version: 1.2.6")
    print(f"🧠 LLM: {settings.LLM_MODEL}")
    print(f"📊 Vector DB: Pinecone ({settings.PINECONE_INDEX_NAME})")
    print("="*70 + "\n")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    print("\n👋 Shutting down Resume Chat API...")

# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
    
import os

if __name__ == "__main__":
    # Render ka port lega, nahi toh local me 5000 par chalega
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)    