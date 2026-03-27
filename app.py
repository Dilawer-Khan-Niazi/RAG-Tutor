import gradio as gr
from book_manager import BookManager
from agentic_rag import AgenticRAG
from chat_storage import ChatStorage
import os
from datetime import datetime
import logging
import traceback

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ============================================================
# Initialize System
# ============================================================

print("="*60)
print("🚀 Initializing Multi-Book Agentic RAG Learning Tutor")
print("="*60)

try:
    # Initialize components
    logger.info("📦 Loading Book Manager...")
    book_manager = BookManager()
    logger.info("✅ Book Manager loaded")
    
    logger.info("💾 Loading Chat Storage...")
    chat_storage = ChatStorage()
    logger.info("✅ Chat Storage loaded")
    
    logger.info("🤖 Loading Agentic RAG...")
    agentic_rag = AgenticRAG(book_manager.embedding_model)
    logger.info("✅ Agentic RAG loaded")
    
    print("\n✅ System ready!")
    print("="*60)
except Exception as e:
    logger.error(f"❌ Initialization error: {str(e)}")
    traceback.print_exc()
    raise

# Global state
current_session_id = None
current_book_id = None

# ============================================================
# Core Functions
# ============================================================

def get_books_list() -> list:
    """Get list of available books for dropdown."""
    books = book_manager.get_all_books()
    if not books:
        return ["No books available - Please upload a book first"]
    return [f"{book['id']}: {book['title']}" for book in books]


def get_first_book_or_none() -> str:
    """Get first book ID:Title or None message."""
    books = get_books_list()
    if books and books[0] != "No books available - Please upload a book first":
        return books[0]
    return "No books available - Please upload a book first"


def upload_new_book(pdf_file, book_title, book_description) -> tuple:
    """
    Handle book upload.
    
    Returns:
        (status_message, books_list_choices, books_list_value, book_select_choices, book_select_value)
    """
    if pdf_file is None:
        current_books = get_books_list()
        first_book = get_first_book_or_none()
        return (
            "⚠️ Please select a PDF file to upload.",
            gr.Dropdown(choices=current_books, value=first_book),  # books_list_dropdown
            gr.Dropdown(choices=current_books, value=first_book),  # book_select
        )
    
    try:
        # Get the file path (Gradio provides temporary path)
        pdf_path = pdf_file.name
        
        # Use filename as title if not provided
        if not book_title or book_title.strip() == "":
            book_title = os.path.basename(pdf_path).replace('.pdf', '')
        
        # Add book to system
        print(f"\n📤 Uploading book: {book_title}")
        book_id = book_manager.add_book(
            pdf_path=pdf_path,
            title=book_title,
            description=book_description or ""
        )
        
        book = book_manager.get_book(book_id)
        
        success_msg = f"""
✅ **Book uploaded successfully!**

📚 **Title:** {book['title']}
🆔 **ID:** {book_id}
📄 **Chunks:** {book['num_chunks']}
💾 **Size:** {book['file_size'] / 1024:.1f} KB

The book is now ready for questions!
        """
        
        # Get updated books list
        updated_books = get_books_list()
        new_book_entry = f"{book_id}: {book['title']}"
        
        print(f"✅ Book added: {new_book_entry}")
        
        return (
            success_msg,
            gr.Dropdown(choices=updated_books, value=new_book_entry),  # books_list_dropdown
            gr.Dropdown(choices=updated_books, value=new_book_entry),  # book_select
        )
        
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        current_books = get_books_list()
        first_book = get_first_book_or_none()
        return (
            f"❌ Error uploading book: {str(e)}",
            gr.Dropdown(choices=current_books, value=first_book),
            gr.Dropdown(choices=current_books, value=first_book),
        )


def delete_selected_book(book_dropdown) -> tuple:
    """Delete a book from the system."""
    if not book_dropdown or "No books available" in book_dropdown:
        current_books = get_books_list()
        first_book = get_first_book_or_none()
        return (
            "⚠️ No book selected.",
            gr.Dropdown(choices=current_books, value=first_book),
            gr.Dropdown(choices=current_books, value=first_book),
        )
    
    try:
        book_id = book_dropdown.split(":")[0].strip()
        book = book_manager.get_book(book_id)
        
        if book_manager.delete_book(book_id):
            updated_books = get_books_list()
            first_book = get_first_book_or_none()
            return (
                f"✅ Book '{book['title']}' deleted successfully!",
                gr.Dropdown(choices=updated_books, value=first_book),
                gr.Dropdown(choices=updated_books, value=first_book),
            )
        else:
            current_books = get_books_list()
            first_book = get_first_book_or_none()
            return (
                "❌ Failed to delete book.",
                gr.Dropdown(choices=current_books, value=first_book),
                gr.Dropdown(choices=current_books, value=first_book),
            )
    except Exception as e:
        current_books = get_books_list()
        first_book = get_first_book_or_none()
        return (
            f"❌ Error: {str(e)}",
            gr.Dropdown(choices=current_books, value=first_book),
            gr.Dropdown(choices=current_books, value=first_book),
        )


def ask_question(question, book_dropdown, difficulty, use_agentic, session_dropdown) -> tuple:
    """
    Process question with selected book using RAG.
    
    Returns:
        (answer, agent_log_text, session_dropdown, sessions_list, chat_history_html)
    """
    global current_session_id, current_book_id
    
    # Input validation
    if not question or question.strip() == "":
        return "⚠️ Please enter a question.", "", session_dropdown, get_sessions_list(), "<p style='color:#7f8c8d'>No chat history available.</p>"
    
    if not book_dropdown or "No books available" in book_dropdown:
        return "⚠️ Please upload a book first or select an available book.", "", session_dropdown, get_sessions_list(), "<p style='color:#7f8c8d'>No chat history available.</p>"
    
    try:
        logger.info(f"📝 Processing question: {question[:50]}...")
        
        # Extract and validate book ID
        try:
            book_id = book_dropdown.split(":")[0].strip()
            book = book_manager.get_book(book_id)
            if not book:
                return "❌ Selected book not found. Please select another book.", "", session_dropdown, get_sessions_list()
            current_book_id = book_id
        except Exception as e:
            logger.error(f"❌ Invalid book selection: {str(e)}")
            return f"❌ Error reading book: {str(e)}", "", session_dropdown, get_sessions_list()
        
        # Load book's vector database
        logger.info(f"📚 Loading vector database for: {book['title']}")
        try:
            index, chunks = book_manager.load_book_index(book_id)
            if not chunks or len(chunks) == 0:
                return f"❌ Book '{book['title']}' has no indexed content. Please re-upload the book.", "", session_dropdown, get_sessions_list()
            logger.info(f"✅ Loaded {len(chunks)} chunks from book")
        except Exception as e:
            logger.error(f"❌ Failed to load vector database: {str(e)}")
            return f"❌ Failed to load book index: {str(e)}.\n\nTry re-uploading the book.", "", session_dropdown, get_sessions_list()
        
        # Determine or create session
        try:
            if session_dropdown and session_dropdown != "-- New Chat --":
                session_id = session_dropdown.split(":")[0].strip()
                logger.info(f"📌 Continuing session: {session_id}")
            else:
                session_id = chat_storage.create_session(
                    book_id=book_id,
                    book_title=book['title']
                )
                current_session_id = session_id
                logger.info(f"✨ Created new session: {session_id}")
        except Exception as e:
            logger.error(f"❌ Session error: {str(e)}")
            return f"❌ Session error: {str(e)}", "", session_dropdown, get_sessions_list()
        
        # Perform retrieval and answer generation
        try:
            if use_agentic:
                logger.info("🤖 Using Agentic RAG (multi-step reasoning)...")
                retrieved_chunks, agent_log = agentic_rag.retrieve_with_reasoning(
                    question=question,
                    index=index,
                    chunks=chunks,
                    max_iterations=2,
                    top_k=5
                )
                
                if not retrieved_chunks or len(retrieved_chunks) == 0:
                    logger.warning("⚠️ No chunks retrieved in agentic mode")
                    return "❌ Could not find relevant information in the book for this question. Try:\n1. Simplifying your question\n2. Using different keywords\n3. Selecting a different book", "", session_dropdown, get_sessions_list()
                
            else:
                logger.info("⚡ Using Traditional RAG (single-step)...")
                try:
                    query_embedding = book_manager.embedding_model.encode([question], convert_to_numpy=True)
                    distances, indices = index.search(query_embedding.astype('float32'), 5)
                    retrieved_chunks = [chunks[idx] for idx in indices[0]]
                    
                    agent_log = {
                        "iterations": [{"iteration": 1, "query": question, "chunks_retrieved": len(retrieved_chunks)}],
                        "total_chunks_retrieved": len(retrieved_chunks),
                        "final_confidence": 0.8
                    }
                except Exception as e:
                    logger.error(f"❌ Retrieval error: {str(e)}")
                    return f"❌ Error retrieving information: {str(e)}", "", session_dropdown, get_sessions_list()
            
            # Generate answer
            logger.info("💭 Generating answer...")
            answer = agentic_rag.generate_answer(
                question=question,
                chunks=retrieved_chunks,
                difficulty=difficulty,
                agent_log=agent_log
            )
            
            if not answer or len(answer) == 0:
                return "❌ Failed to generate answer. Please try again.", "", session_dropdown, get_sessions_list()
            
            # Format agent log for display
            if use_agentic:
                agent_log_text = f"""
### 🤖 Agentic Retrieval Process

**Total Iterations:** {len(agent_log['iterations'])}  
**Chunks Retrieved:** {agent_log['total_chunks_retrieved']}  
**Final Confidence:** {agent_log['final_confidence']:.0%}

"""
                for iteration in agent_log['iterations']:
                    agent_log_text += f"""
**Iteration {iteration['iteration']}:**
- Query: {iteration['query']}
- Chunks: {iteration['chunks_retrieved']}
- Sufficient: {iteration['evaluation']['sufficient']}
- Confidence: {iteration['evaluation']['confidence']:.0%}

"""
            else:
                agent_log_text = f"""
### ⚡ Traditional RAG

**Single retrieval step used**  
**Chunks Retrieved:** {agent_log['total_chunks_retrieved']}  
**Confidence:** {agent_log['final_confidence']:.0%}
"""
            
            # Save to chat history
            try:
                chat_storage.add_message(
                    session_id=session_id,
                    question=question,
                    answer=answer,
                    difficulty=difficulty,
                    agent_log=agent_log,
                    book_id=book_id
                )
                logger.info("💾 Message saved to chat history")
            except Exception as e:
                logger.warning(f"⚠️ Failed to save chat history: {str(e)}")
                # Don't fail the question, but log the issue
            
            # Update session dropdown + history
            try:
                session = chat_storage.get_session(session_id)
                updated_session = f"{session_id}: {session['title']}"
                chat_html = load_chat_history(updated_session)
                logger.info("✅ Question processed successfully")
                return answer, agent_log_text, updated_session, get_sessions_list(), chat_html
            except Exception as e:
                logger.warning(f"⚠️ Failed to get session: {str(e)}")
                return answer, agent_log_text, session_dropdown, get_sessions_list(), "<p style='color:#7f8c8d'>No chat history available.</p>"
        
        except Exception as e:
            logger.error(f"❌ Generation error: {str(e)}")
            return f"❌ Error generating answer: {str(e)}\n\nPlease try:\n1. Reformulating your question\n2. Selecting a different book\n3. Trying again in a moment", "", session_dropdown, get_sessions_list()
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return f"❌ Unexpected error: {str(e)}\n\nPlease try again or contact support.", "", session_dropdown, get_sessions_list(), "<p style='color:#7f8c8d'>No chat history available.</p>"


def get_sessions_list() -> list:
    """Get list of chat sessions."""
    sessions = chat_storage.get_all_sessions()
    if not sessions:
        return ["-- New Chat --"]
    
    session_list = ["-- New Chat --"]
    for session in sessions:
        session_list.append(f"{session['id']}: {session['title']}")
    return session_list


def load_chat_history(session_dropdown) -> str:
    """Load chat history in HTML format."""
    if not session_dropdown or session_dropdown == "-- New Chat --":
        return "<p style='color: #7f8c8d; text-align: center; padding: 40px;'>Select a chat session to view conversation history.</p>"
    
    session_id = session_dropdown.split(":")[0].strip()
    return chat_storage.get_chat_history_html(session_id)


def delete_chat_session(session_dropdown) -> tuple:
    """Delete selected chat session."""
    if not session_dropdown or session_dropdown == "-- New Chat --":
        return "⚠️ No session selected.", get_sessions_list(), "<p>No session selected.</p>"
    
    session_id = session_dropdown.split(":")[0].strip()
    session = chat_storage.get_session(session_id)
    
    chat_storage.delete_session(session_id)
    
    return f"✅ Chat session '{session['title']}' deleted successfully!", get_sessions_list(), "<p>Session deleted.</p>"


def get_book_details(book_dropdown) -> str:
    """Display book details."""
    if not book_dropdown or "No books available" in book_dropdown:
        return "<p style='color: #7f8c8d;'>No book selected.</p>"
    
    try:
        book_id = book_dropdown.split(":")[0].strip()
        book = book_manager.get_book(book_id)
        
        return f"""
<div style='padding: 20px; background: #f8f9fa; border-radius: 10px; border-left: 4px solid #3498db;'>
    <h3 style='color: #2c3e50; margin-top: 0;'>📚 {book['title']}</h3>
    <p style='color: #7f8c8d; font-size: 0.9em; margin: 5px 0;'>
        <strong>ID:</strong> {book['id']}<br>
        <strong>Added:</strong> {book['added_at'][:19]}<br>
        <strong>Chunks:</strong> {book['num_chunks']}<br>
        <strong>File Size:</strong> {book['file_size'] / 1024:.1f} KB
    </p>
    {f"<p style='color: #34495e; margin-top: 15px;'><strong>Description:</strong><br>{book['description']}</p>" if book['description'] else ""}
</div>
        """
    except:
        return "<p style='color: #e74c3c;'>Error loading book details.</p>"


# ============================================================
# Build Gradio Interface
# ============================================================

# Custom CSS for clean, professional look
custom_css = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto;
    font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: #f9fafb;
    color: #1f2937;
}

#main-title {
    text-align: center;
    color: #111827;
    font-size: 2rem;
    font-weight: 600;
    margin-bottom: 5px;
}

#subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 1rem;
    margin-bottom: 20px;
}

.gradio-row, .gradio-column {
    gap: 10px !important;
}

.gradio-container .gradio-html p,
.gradio-container .gradio-markdown p {
    color: #374151;
}

.tab-nav button {
    font-size: 1rem !important;
    padding: 8px 16px !important;
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 0 !important;
}

.tab-nav button[aria-selected='true'] {
    background: #f3f4f6 !important;
    border-color: #9ca3af !important;
}

footer {
    display: none !important;
}
"""

# Build interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="RAG Learning Tutor") as demo:
    
    # Header
    gr.Markdown("<h1 id='main-title'>Agentic RAG Learning Tutor</h1>")
    # gr.Markdown("<p id='subtitle'>Simple minimal interface with book-based AI conversation.</p>")
    
    # Main tabs
    with gr.Tabs() as tabs:
        
        # ============================================================
        # TAB 1: Ask Questions
        # ============================================================
        with gr.Tab("💬 Ask Questions", id=0):
            with gr.Row():
                # Left column - Input
                with gr.Column(scale=1):
                    gr.Markdown("### 📚 Select Book")
                    book_select = gr.Dropdown(
                        choices=get_books_list(),
                        label="Choose a book to query",
                        interactive=True,
                        value=get_first_book_or_none()
                    )
                    
                    book_info_display = gr.HTML(
                        value="<p style='color: #7f8c8d;'>Select a book to see details</p>"
                    )
                    
                    gr.Markdown("---")
                    
                    gr.Markdown("### 📝 Your Question")
                    question_input = gr.Textbox(
                        label="",
                        placeholder="Example: What is machine learning? How do neural networks work?",
                        lines=4
                    )
                    
                    with gr.Row():
                        difficulty_select = gr.Radio(
                            choices=["beginner", "intermediate", "advanced"],
                            value="beginner",
                            label="📊 Difficulty Level"
                        )
                    
                    use_agentic = gr.Checkbox(
                        label="🤖 Use Agentic RAG (Multi-step reasoning - slower but more accurate)",
                        value=True
                    )
                    
                    gr.Markdown("### 💾 Session")
                    session_select = gr.Dropdown(
                        choices=get_sessions_list(),
                        value="-- New Chat --",
                        label="Continue conversation or start new",
                        interactive=True
                    )
                    
                    submit_btn = gr.Button("🚀 Get Answer", variant="primary", size="lg")
                    
                    # Example questions
                    gr.Markdown("### 💡 Example Questions")
                    gr.Examples(
                        examples=[
                            ["What is machine learning?", "beginner"],
                            ["Explain supervised learning with examples", "beginner"],
                            ["How do neural networks learn?", "intermediate"],
                            ["Compare gradient descent variants", "advanced"],
                        ],
                        inputs=[question_input, difficulty_select],
                    )
                
                # Right column - Output
                with gr.Column(scale=1):
                    gr.Markdown("### 💡 Answer")
                    answer_output = gr.Markdown(
                        value="_Your answer will appear here..._",
                        line_breaks=True
                    )
                    
                    with gr.Accordion("🔍 Retrieval Process Details", open=False):
                        agent_log_output = gr.Markdown(
                            value="_Agent retrieval log will appear here..._"
                        )
        
        # ============================================================
        # TAB 2: Chat History
        # ============================================================
        with gr.Tab("📜 Chat History", id=1):
            gr.Markdown("### 📚 Your Learning Conversations")
            
            with gr.Row():
                with gr.Column(scale=1):
                    history_session_select = gr.Dropdown(
                        choices=get_sessions_list(),
                        label="Select Chat Session",
                        interactive=True
                    )
                    
                    with gr.Row():
                        load_history_btn = gr.Button("📖 Load Chat", variant="primary")
                        delete_session_btn = gr.Button("🗑️ Delete", variant="stop")
                    
                    delete_status = gr.Markdown("")
                
                with gr.Column(scale=2):
                    chat_history_display = gr.HTML(
                        value="<p style='color: #7f8c8d; text-align: center; padding: 40px;'>Select a chat session to view history.</p>"
                    )
        
        # ============================================================
        # TAB 3: Manage Books
        # ============================================================
        with gr.Tab("📚 Manage Books", id=2):
            with gr.Row():
                # Upload section
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 Upload New Book")
                    
                    pdf_upload = gr.File(
                        label="Select PDF File",
                        file_types=[".pdf"],
                        type="filepath"
                    )
                    
                    book_title_input = gr.Textbox(
                        label="Book Title (optional - will use filename if empty)",
                        placeholder="e.g., Introduction to Machine Learning"
                    )
                    
                    book_desc_input = gr.Textbox(
                        label="Description (optional)",
                        placeholder="Brief description of the book content",
                        lines=3
                    )
                    
                    upload_btn = gr.Button("📤 Upload Book", variant="primary", size="lg")
                    
                    upload_status = gr.Markdown("")
                
                # Books list section
                with gr.Column(scale=1):
                    gr.Markdown("### 📚 Your Books Library")
                    
                    books_list_dropdown = gr.Dropdown(
                        choices=get_books_list(),
                        label="Select book to view or delete",
                        interactive=True,
                        value=get_first_book_or_none()
                    )
                    
                    book_details_display = gr.HTML(
                        value="<p style='color: #7f8c8d;'>Select a book to see details</p>"
                    )
                    
                    delete_book_btn = gr.Button("🗑️ Delete Selected Book", variant="stop")
                    
                    delete_book_status = gr.Markdown("")
    
    # ============================================================
    # Event Handlers
    # ============================================================
    
    # Ask question
    submit_btn.click(
        fn=ask_question,
        inputs=[question_input, book_select, difficulty_select, use_agentic, session_select],
        outputs=[answer_output, agent_log_output, session_select, history_session_select, chat_history_display]
    )
    
    question_input.submit(
        fn=ask_question,
        inputs=[question_input, book_select, difficulty_select, use_agentic, session_select],
        outputs=[answer_output, agent_log_output, session_select, history_session_select, chat_history_display]
    )
    
    # Book selection - show details
    book_select.change(
        fn=get_book_details,
        inputs=[book_select],
        outputs=[book_info_display]
    )
    
    # Upload book - FIXED
    upload_btn.click(
        fn=upload_new_book,
        inputs=[pdf_upload, book_title_input, book_desc_input],
        outputs=[upload_status, books_list_dropdown, book_select]
    )
    
    # Delete book - FIXED
    delete_book_btn.click(
        fn=delete_selected_book,
        inputs=[books_list_dropdown],
        outputs=[delete_book_status, books_list_dropdown, book_select]
    )
    
    # Show book details in library
    books_list_dropdown.change(
        fn=get_book_details,
        inputs=[books_list_dropdown],
        outputs=[book_details_display]
    )
    
    # Load chat history
    load_history_btn.click(
        fn=load_chat_history,
        inputs=[history_session_select],
        outputs=[chat_history_display]
    )
    
    # Delete chat session
    delete_session_btn.click(
        fn=delete_chat_session,
        inputs=[history_session_select],
        outputs=[delete_status, history_session_select, chat_history_display]
    )
    
    # Footer
    gr.Markdown("---")
    gr.Markdown(
        """
        <div style='text-align: center; color: #95a5a6; font-size: 0.9em; padding: 20px;'>
            <p>🤖 <strong>Agentic RAG System</strong> with Multi-Book Support | 
            📚 Intelligent Multi-Step Retrieval | 
            🧠 Powered by Google Gemini 2.5 Flash</p>
            <p style='font-size: 0.85em; margin-top: 10px;'>
            Built with Sentence Transformers, FAISS Vector DB, and Gradio
            </p>
        </div>
        """
    )

# ============================================================
# Launch
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🌐 Launching Multi-Book Agentic RAG Learning Tutor")
    print("="*60)
    print("\n📝 Features:")
    print("   ✅ Multi-book support with upload")
    print("   ✅ Agentic RAG (multi-step reasoning)")
    print("   ✅ Traditional RAG option")
    print("   ✅ Full chat history")
    print("   ✅ Professional UI")
    print("\n" + "="*60 + "\n")
    
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )