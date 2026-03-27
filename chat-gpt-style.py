import gradio as gr
from book_manager import BookManager
from agentic_rag import AgenticRAG
from chat_storage import ChatStorage
import os
from datetime import datetime

# ============================================================
# Initialize System
# ============================================================

print("="*60)
print("🚀 Initializing ChatGPT-Style RAG Learning Tutor")
print("="*60)

book_manager = BookManager()
chat_storage = ChatStorage()
agentic_rag = AgenticRAG(book_manager.embedding_model)

print("\n✅ System ready!")
print("="*60)

# ============================================================
# Helper Functions
# ============================================================

def get_books_list():
    """Get list of available books."""
    books = book_manager.get_all_books()
    if not books:
        return [{"label": "No books - Upload one first", "value": None}]
    return [{"label": f"📚 {book['title']}", "value": book['id']} for book in books]


def get_sessions_list():
    """Get list of chat sessions for sidebar."""
    sessions = chat_storage.get_all_sessions()
    if not sessions:
        return []
    
    sessions_html = ""
    for session in sessions[:20]:  # Show last 20 sessions
        book_emoji = "📚"
        truncated_title = session['title'][:40] + "..." if len(session['title']) > 40 else session['title']
        
        sessions_html += f"""
        <div class='session-item' onclick='loadSession("{session["id"]}")' 
             style='padding: 12px; margin: 5px 0; border-radius: 8px; cursor: pointer; 
                    background: #f8f9fa; border-left: 3px solid #667eea;
                    transition: all 0.2s;'
             onmouseover='this.style.background="#e9ecef"' 
             onmouseout='this.style.background="#f8f9fa"'>
            <div style='font-weight: 600; color: #2c3e50; font-size: 0.9em;'>
                {book_emoji} {truncated_title}
            </div>
            <div style='font-size: 0.75em; color: #7f8c8d; margin-top: 4px;'>
                {session['created_at'][:10]} • {len(session['messages'])} msgs
            </div>
        </div>
        """
    
    return sessions_html


def format_chat_messages(session_id):
    """Format chat messages in ChatGPT style."""
    if not session_id:
        return """
        <div style='text-align: center; padding: 100px 20px; color: #95a5a6;'>
            <h2 style='color: #7f8c8d;'>👋 Welcome to Your AI Tutor</h2>
            <p>Select a book and start asking questions to begin learning!</p>
        </div>
        """
    
    session = chat_storage.get_session(session_id)
    if not session:
        return "<div style='padding: 20px; color: #e74c3c;'>Session not found</div>"
    
    messages_html = ""
    
    for msg in session['messages']:
        # User message
        messages_html += f"""
        <div style='display: flex; justify-content: flex-end; margin: 20px 0;'>
            <div style='max-width: 70%; background: #667eea; color: white; 
                        padding: 12px 16px; border-radius: 18px; border-bottom-right-radius: 4px;'>
                <div style='font-size: 0.95em; line-height: 1.5;'>{msg['question']}</div>
                <div style='font-size: 0.7em; opacity: 0.8; margin-top: 6px; text-align: right;'>
                    {msg['timestamp'][11:16]} • {msg['difficulty']}
                </div>
            </div>
        </div>
        """
        
        # AI response
        messages_html += f"""
        <div style='display: flex; justify-content: flex-start; margin: 20px 0;'>
            <div style='max-width: 70%; background: #f1f3f5; color: #2c3e50;
                        padding: 12px 16px; border-radius: 18px; border-bottom-left-radius: 4px;
                        box-shadow: 0 1px 2px rgba(0,0,0,0.1);'>
                <div style='font-size: 0.95em; line-height: 1.6;'>
                    {msg['answer'].replace(chr(10), '<br>')}
                </div>
                <div style='font-size: 0.7em; color: #868e96; margin-top: 8px;'>
                    {msg['timestamp'][11:16]}
                """
        
        # Add agent info if available
        if msg.get('agent_log'):
            messages_html += f""" • 🤖 {msg['agent_log']['total_chunks_retrieved']} sources"""
        
        messages_html += """
                </div>
            </div>
        </div>
        """
    
    return f"<div style='padding: 20px;'>{messages_html}</div>"


def upload_book_handler(pdf_file, book_title, book_description):
    """Handle book upload."""
    if pdf_file is None:
        return "⚠️ Please select a PDF file", get_books_list()
    
    try:
        pdf_path = pdf_file.name
        
        if not book_title or book_title.strip() == "":
            book_title = os.path.basename(pdf_path)
        
        book_id = book_manager.add_book(
            pdf_path=pdf_path,
            title=book_title,
            description=book_description or ""
        )
        
        book = book_manager.get_book(book_id)
        
        return (
            f"✅ Book '{book['title']}' uploaded! ({book['num_chunks']} chunks ready)",
            get_books_list()
        )
    except Exception as e:
        return f"❌ Error: {str(e)}", get_books_list()


def send_message(message, book_id, difficulty, use_agentic, current_session, chat_history):
    """Send a message and get response."""
    if not message or not message.strip():
        return chat_history, "", current_session
    
    if not book_id:
        error_html = chat_history + """
        <div style='text-align: center; padding: 20px; color: #e74c3c;'>
            ⚠️ Please select a book first!
        </div>
        """
        return error_html, message, current_session
    
    try:
        # Load book
        index, chunks = book_manager.load_book_index(book_id)
        book = book_manager.get_book(book_id)
        
        # Create or use session
        if not current_session:
            current_session = chat_storage.create_session(
                book_id=book_id,
                book_title=book['title']
            )
        
        # Add user message to display immediately
        chat_history += f"""
        <div style='display: flex; justify-content: flex-end; margin: 20px 0;'>
            <div style='max-width: 70%; background: #667eea; color: white; 
                        padding: 12px 16px; border-radius: 18px; border-bottom-right-radius: 4px;'>
                <div style='font-size: 0.95em; line-height: 1.5;'>{message}</div>
                <div style='font-size: 0.7em; opacity: 0.8; margin-top: 6px; text-align: right;'>
                    {datetime.now().strftime('%H:%M')} • {difficulty}
                </div>
            </div>
        </div>
        """
        
        # Show thinking indicator
        chat_history += """
        <div style='display: flex; justify-content: flex-start; margin: 20px 0;'>
            <div style='max-width: 70%; background: #f1f3f5; color: #2c3e50;
                        padding: 12px 16px; border-radius: 18px;'>
                <div style='font-size: 0.95em;'>🤔 Thinking...</div>
            </div>
        </div>
        """
        
        # Get AI response
        if use_agentic:
            retrieved_chunks, agent_log = agentic_rag.retrieve_with_reasoning(
                question=message,
                index=index,
                chunks=chunks,
                max_iterations=2,
                top_k=5
            )
            answer = agentic_rag.generate_answer(message, retrieved_chunks, difficulty, agent_log)
        else:
            query_embedding = book_manager.embedding_model.encode([message], convert_to_numpy=True)
            distances, indices = index.search(query_embedding.astype('float32'), 5)
            retrieved_chunks = [chunks[idx] for idx in indices[0]]
            agent_log = {"total_chunks_retrieved": 5, "final_confidence": 0.8}
            answer = agentic_rag.generate_answer(message, retrieved_chunks, difficulty, agent_log)
        
        # Save to storage
        chat_storage.add_message(
            session_id=current_session,
            question=message,
            answer=answer,
            difficulty=difficulty,
            agent_log=agent_log,
            book_id=book_id
        )
        
        # Remove thinking indicator and add real response
        chat_history = chat_history.rsplit('<div style=', 1)[0]  # Remove thinking message
        
        chat_history += f"""
        <div style='display: flex; justify-content: flex-start; margin: 20px 0;'>
            <div style='max-width: 70%; background: #f1f3f5; color: #2c3e50;
                        padding: 12px 16px; border-radius: 18px; border-bottom-left-radius: 4px;
                        box-shadow: 0 1px 2px rgba(0,0,0,0.1);'>
                <div style='font-size: 0.95em; line-height: 1.6;'>
                    {answer.replace(chr(10), '<br>')}
                </div>
                <div style='font-size: 0.7em; color: #868e96; margin-top: 8px;'>
                    {datetime.now().strftime('%H:%M')} • 🤖 {agent_log['total_chunks_retrieved']} sources
                </div>
            </div>
        </div>
        """
        
        return chat_history, "", current_session
        
    except Exception as e:
        error_msg = f"""
        <div style='text-align: center; padding: 20px; color: #e74c3c;'>
            ❌ Error: {str(e)}
        </div>
        """
        return chat_history + error_msg, message, current_session


def new_chat():
    """Start a new chat session."""
    return "", None, """
    <div style='text-align: center; padding: 100px 20px; color: #95a5a6;'>
        <h2 style='color: #7f8c8d;'>👋 New Conversation</h2>
        <p>Select a book and start asking questions!</p>
    </div>
    """


# ============================================================
# Build ChatGPT-Style Interface
# ============================================================

custom_css = """
.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
}

#sidebar {
    background: #f8f9fa;
    border-right: 1px solid #dee2e6;
    height: 100vh;
    overflow-y: auto;
}

#main-chat {
    height: calc(100vh - 80px);
    overflow-y: auto;
    background: white;
}

#input-area {
    background: white;
    border-top: 1px solid #dee2e6;
    padding: 20px;
}

.send-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
}

footer {
    display: none !important;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="AI Learning Tutor") as demo:
    
    # Store session state
    current_session_state = gr.State(None)
    
    with gr.Row():
        # LEFT SIDEBAR (25% width)
        with gr.Column(scale=1, elem_id="sidebar"):
            gr.Markdown("### 🎓 AI Learning Tutor")
            
            # New chat button
            new_chat_btn = gr.Button("➕ New Chat", size="sm", variant="primary")
            
            gr.Markdown("---")
            
            # Book selector
            gr.Markdown("**📚 Select Book:**")
            book_radio = gr.Radio(
                choices=get_books_list(),
                label="",
                container=False
            )
            
            gr.Markdown("---")
            
            # Settings
            gr.Markdown("**⚙️ Settings:**")
            difficulty_radio = gr.Radio(
                choices=["beginner", "intermediate", "advanced"],
                value="beginner",
                label="Difficulty",
                container=False
            )
            
            agentic_check = gr.Checkbox(
                label="🤖 Agentic Mode",
                value=True,
                info="Multi-step reasoning"
            )
            
            gr.Markdown("---")
            
            # Chat history
            gr.Markdown("**💬 Recent Chats:**")
            sessions_display = gr.HTML(
                value=get_sessions_list(),
                elem_id="sessions-list"
            )
            
            gr.Markdown("---")
            
            # Upload book section
            with gr.Accordion("📤 Upload Book", open=False):
                upload_file = gr.File(label="PDF File", file_types=[".pdf"])
                upload_title = gr.Textbox(label="Title", placeholder="Book title")
                upload_desc = gr.Textbox(label="Description", lines=2)
                upload_btn = gr.Button("Upload", size="sm")
                upload_status = gr.Markdown("")
        
        # MAIN CHAT AREA (75% width)
        with gr.Column(scale=3):
            # Chat messages display
            chat_display = gr.HTML(
                value="""
                <div style='text-align: center; padding: 100px 20px; color: #95a5a6;'>
                    <h2 style='color: #7f8c8d;'>👋 Welcome to Your AI Tutor</h2>
                    <p>Select a book from the left and start asking questions!</p>
                </div>
                """,
                elem_id="main-chat"
            )
            
            # Input area at bottom
            with gr.Row(elem_id="input-area"):
                with gr.Column(scale=9):
                    message_input = gr.Textbox(
                        placeholder="Ask a question about your textbook...",
                        show_label=False,
                        container=False,
                        lines=1
                    )
                with gr.Column(scale=1):
                    send_btn = gr.Button("Send ➤", elem_classes=["send-btn"])
    
    # ============================================================
    # Event Handlers
    # ============================================================
    
    # Send message
    send_btn.click(
        fn=send_message,
        inputs=[message_input, book_radio, difficulty_radio, agentic_check, 
                current_session_state, chat_display],
        outputs=[chat_display, message_input, current_session_state]
    )
    
    message_input.submit(
        fn=send_message,
        inputs=[message_input, book_radio, difficulty_radio, agentic_check, 
                current_session_state, chat_display],
        outputs=[chat_display, message_input, current_session_state]
    )
    
    # New chat
    new_chat_btn.click(
        fn=new_chat,
        outputs=[message_input, current_session_state, chat_display]
    )
    
    # Upload book
    upload_btn.click(
        fn=upload_book_handler,
        inputs=[upload_file, upload_title, upload_desc],
        outputs=[upload_status, book_radio]
    )

# ============================================================
# Launch
# ============================================================

if __name__ == "__main__":
    print("\n🌐 Launching ChatGPT-Style Interface...")
    print("="*60 + "\n")
    
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )