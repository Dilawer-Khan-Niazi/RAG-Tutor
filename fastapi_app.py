import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from book_manager import BookManager
from agentic_rag import AgenticRAG
from chat_storage import ChatStorage

app = FastAPI(
    title="RAG Tutor API",
    description="FastAPI wrapper for the Agentic RAG learning tutor",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

books = BookManager()
chats = ChatStorage()
rag = AgenticRAG(books.embedding_model)

# === Models ===
class AskRequest(BaseModel):
    question: str
    book_id: str
    difficulty: str = "beginner"
    use_agentic: bool = True
    session_id: str = None


class AskResponse(BaseModel):
    answer: str
    agent_log: str
    session_id: str
    chat_history_html: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/books")
def list_books():
    return books.get_all_books()


@app.get("/sessions")
def list_sessions():
    return chats.get_all_sessions()


@app.get("/history/{session_id}")
def get_history(session_id: str):
    session = chats.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session": session,
        "html": chats.get_chat_history_html(session_id)
    }


@app.post("/upload")
async def upload_book(file: UploadFile = File(...), title: str = Form(""), description: str = Form("")):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    tmp_path = os.path.join('books', f'tmp_{file.filename}')
    with open(tmp_path, 'wb') as f:
        f.write(await file.read())

    try:
        new_id = books.add_book(pdf_path=tmp_path, title=title or file.filename, description=description)
        new_book = books.get_book(new_id)
        return new_book
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Validate book exists
    book = books.get_book(request.book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    # Create or use existing session
    session_id = request.session_id
    if session_id and chats.get_session(session_id) is None:
        session_id = None

    if not session_id:
        session_id = chats.create_session(book_id=book['id'], book_title=book['title'])

    # Load vector DB
    index, chunks = books.load_book_index(request.book_id)

    if request.use_agentic:
        retrieved, agent_log = rag.retrieve_with_reasoning(
            question=request.question,
            index=index,
            chunks=chunks,
            max_iterations=2,
            top_k=5,
        )
    else:
        query_embedding = books.embedding_model.encode([request.question], convert_to_numpy=True)
        distances, indices = index.search(query_embedding.astype('float32'), 5)
        retrieved = [chunks[idx] for idx in indices[0]]
        agent_log = {
            'iterations': [{'iteration': 1, 'query': request.question, 'chunks_retrieved': len(retrieved)}],
            'total_chunks_retrieved': len(retrieved),
            'final_confidence': 0.8,
        }

    answer_text = rag.generate_answer(
        question=request.question,
        chunks=retrieved,
        difficulty=request.difficulty,
        agent_log=agent_log,
    )

    chats.add_message(
        session_id=session_id,
        question=request.question,
        answer=answer_text,
        difficulty=request.difficulty,
        agent_log=agent_log,
        book_id=request.book_id,
    )

    return AskResponse(
        answer=answer_text,
        agent_log=str(agent_log),
        session_id=session_id,
        chat_history_html=chats.get_chat_history_html(session_id),
    )
