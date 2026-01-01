import gradio as gr
from rag_tutor import RAGTutor
import os

# ============================================================
# Initialize RAG Tutor (load once when app starts)
# ============================================================

print("🚀 Starting RAG Learning Tutor Web App...")
tutor = RAGTutor()
print("✅ Tutor loaded and ready!")

# ============================================================
# Gradio Interface Functions
# ============================================================

def answer_question(question: str, difficulty: str, num_sources: int) -> tuple:
    """
    Process question and return answer with sources.
    
    Args:
        question: User's question
        difficulty: Learning level
        num_sources: Number of source chunks to retrieve
        
    Returns:
        Tuple of (answer, sources_text)
    """
    if not question.strip():
        return "⚠️ Please enter a question!", ""
    
    # Get answer from RAG system
    result = tutor.ask(question, difficulty=difficulty, top_k=num_sources)
    
    # Format sources for display
    sources_text = "📚 **Source Excerpts from Your Textbook:**\n\n"
    for i, source in enumerate(result["sources"], 1):
        sources_text += f"**[Excerpt {i}]**\n{source}\n\n---\n\n"
    
    return result["answer"], sources_text


# ============================================================
# Build Gradio Interface
# ============================================================

# Custom CSS for better styling
custom_css = """
.gradio-container {
    max-width: 1200px !important;
}
#title {
    text-align: center;
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 10px;
}
#subtitle {
    text-align: center;
    font-size: 1.2em;
    color: #666;
    margin-bottom: 30px;
}
"""

# Create the interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    # Header
    gr.Markdown("<div id='title'>🎓 RAG Learning Tutor</div>")
    gr.Markdown("<div id='subtitle'>Ask questions about your textbook and get instant, accurate answers!</div>")
    
    # Main content
    with gr.Row():
        with gr.Column(scale=1):
            # Input section
            gr.Markdown("### 📝 Ask Your Question")
            
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What is machine learning? Explain neural networks...",
                lines=3
            )
            
            with gr.Row():
                difficulty = gr.Radio(
                    choices=["beginner", "intermediate", "advanced"],
                    value="beginner",
                    label="📊 Difficulty Level",
                    info="Choose your learning level"
                )
            
            num_sources = gr.Slider(
                minimum=3,
                maximum=10,
                value=5,
                step=1,
                label="📚 Number of Source Excerpts",
                info="More sources = more context (but slower)"
            )
            
            submit_btn = gr.Button("🚀 Get Answer", variant="primary", size="lg")
            
            # Example questions
            gr.Markdown("### 💡 Example Questions")
            gr.Examples(
                examples=[
                    ["What is machine learning?", "beginner", 5],
                    ["Explain supervised learning", "beginner", 5],
                    ["What is the difference between classification and regression?", "intermediate", 5],
                    ["How does a neural network work?", "intermediate", 5],
                    ["Explain overfitting and how to prevent it", "advanced", 7],
                ],
                inputs=[question_input, difficulty, num_sources],
            )
        
        with gr.Column(scale=1):
            # Output section
            gr.Markdown("### 💡 Answer")
            answer_output = gr.Markdown(
                label="Answer",
                value="*Your answer will appear here...*"
            )
    
    # Sources section (full width)
    with gr.Accordion("📖 View Source Excerpts", open=False):
        sources_output = gr.Markdown(
            label="Sources",
            value="*Source excerpts will appear here...*"
        )
    
    # Connect button to function
    submit_btn.click(
        fn=answer_question,
        inputs=[question_input, difficulty, num_sources],
        outputs=[answer_output, sources_output]
    )
    
    # Also trigger on Enter key
    question_input.submit(
        fn=answer_question,
        inputs=[question_input, difficulty, num_sources],
        outputs=[answer_output, sources_output]
    )
    
    # Footer
    gr.Markdown("---")
    gr.Markdown(
        """
        <div style='text-align: center; color: #666;'>
        🤖 Powered by RAG (Retrieval-Augmented Generation) | 
        📚 Using your own textbooks | 
        🧠 Google Gemini AI
        </div>
        """
    )

# ============================================================
# Launch the app
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🌐 Launching web interface...")
    print("="*60)
    
    demo.launch(
        share=False,  # Set to True to get a public link
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )