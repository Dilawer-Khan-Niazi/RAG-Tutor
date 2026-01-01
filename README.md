# 📚 RAG Learning Tutor

An AI-powered learning assistant that reads your textbooks and answers questions using Retrieval-Augmented Generation (RAG).

## 🎯 What It Does

- 📖 Reads PDF textbooks and converts them into a searchable knowledge base
- 🔍 Finds relevant content using semantic search (understands meaning, not just keywords)
- 🤖 Generates accurate answers using Google Gemini AI
- 🌐 Provides a beautiful web interface for easy interaction
- 📚 Shows source excerpts for transparency

## 🏗️ Architecture

```
PDF → Text Extraction → Chunking → Embeddings → FAISS Database
                                                      ↓
User Question → Embedding → Semantic Search → Top 5 Chunks
                                                      ↓
                                    Chunks + Question → Gemini AI
                                                      ↓
                                                   Answer
```

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| PDF Processing | pypdf | Extract text from PDFs |
| Embeddings | sentence-transformers | Convert text to vectors |
| Vector DB | FAISS | Fast semantic search |
| AI Model | Google Gemini 2.5 Flash | Generate answers |
| Web Interface | Gradio | User-friendly UI |
| API Management | python-dotenv | Secure API keys |

## 📦 Installation

### 1. Clone or Download Project
```bash
mkdir rag-learning-tutor
cd rag-learning-tutor
```

### 2. Install Dependencies
```bash
pip install transformers sentence-transformers faiss-cpu pypdf gradio google-generativeai python-dotenv
```

### 3. Setup Project Structure
```bash
mkdir data vector_db
```

### 4. Get Google Gemini API Key
1. Visit: https://aistudio.google.com/
2. Create API key
3. Create `.env` file:
```
GOOGLE_API_KEY=your_api_key_here
```

## 🚀 Quick Start

### Step 1: Add Your Textbooks
Place PDF files in the `data/` folder:
```
data/
  └── your-textbook.pdf
```

### Step 2: Process Documents
```bash
python process_documents.py
```
This will:
- Extract text from PDFs
- Split into chunks
- Create embeddings
- Build FAISS vector database

**Output:** `vector_db/faiss_index.bin` and `vector_db/chunks.pkl`

### Step 3: Launch Web Interface
```bash
python app.py
```
Open browser to: http://127.0.0.1:7860

## 📂 Project Structure

```
rag-learning-tutor/
│
├── data/                          # Your PDF textbooks
├── vector_db/                     # Generated vector database
│   ├── faiss_index.bin           # FAISS index
│   └── chunks.pkl                # Text chunks
│
├── .env                          # API keys (keep secret!)
├── process_documents.py          # PDF → Vector DB
├── rag_tutor.py                  # RAG logic
└── app.py                        # Web interface
```

## 🔧 How It Works

### 1. Document Processing
- **Chunking:** Splits text into 500-character chunks with 50-character overlap
- **Embedding:** Converts each chunk to 384-dimensional vectors using `all-MiniLM-L6-v2`
- **Indexing:** Stores vectors in FAISS for fast similarity search

### 2. Query Pipeline
1. User asks question
2. Question converted to embedding
3. FAISS finds top 5 most similar chunks (semantic search)
4. Chunks sent to Gemini with prompt
5. AI generates answer based only on retrieved content
6. Answer + sources displayed

### 3. Key Features
- **No Hallucinations:** AI only uses your textbook content
- **Semantic Search:** Finds meaning, not just keywords
- **Adjustable Difficulty:** Beginner/Intermediate/Advanced explanations
- **Source Attribution:** See which textbook excerpts were used

## 📊 Configuration

### Chunking (in `process_documents.py`)
```python
chunk_size = 500      # Characters per chunk
overlap = 50          # Overlap between chunks
```

### Retrieval (in `rag_tutor.py`)
```python
top_k = 5            # Number of chunks to retrieve
model = "gemini-2.5-flash"  # AI model
```

### Web Interface (in `app.py`)
```python
share=False          # Set True for public link
server_port=7860     # Change port if needed
```

## 💡 Usage Examples

### In Python (Command Line)
```python
from rag_tutor import RAGTutor

tutor = RAGTutor()
result = tutor.ask("What is machine learning?", difficulty="beginner")
print(result["answer"])
```

### In Web Interface
1. Type question: "Explain neural networks"
2. Select difficulty: Beginner
3. Click "Get Answer"
4. View answer and sources

## 🎯 Example Questions

- What is machine learning?
- Explain supervised learning in simple terms
- What's the difference between classification and regression?
- How do neural networks work?
- What is overfitting and how to prevent it?

## ⚙️ Advanced Configuration

### Use Different Embedding Model
```python
model = SentenceTransformer("all-mpnet-base-v2")  # More accurate, slower
```

### Use Different Gemini Model
```python
model = genai.GenerativeModel('gemini-2.5-pro')  # More powerful, slower
```

### Adjust Retrieval Parameters
```python
top_k = 10           # Retrieve more chunks (more context)
```

## 🔐 Security Notes

- ✅ Never commit `.env` file to Git
- ✅ Add `.env` to `.gitignore`
- ✅ Keep API keys secret
- ✅ Revoke and regenerate if exposed

## 📈 Performance

| Metric | Value |
|--------|-------|
| **Processing Time** | ~2 min for 100-page textbook |
| **Query Time** | 2-5 seconds per question |
| **Retrieval Speed** | <100ms (FAISS) |
| **Chunks Supported** | 10,000+ |
| **Cost** | Free tier (Gemini API) |

## 🐛 Troubleshooting

### "No PDF files found"
- Ensure PDFs are in `data/` folder
- Check file extension is `.pdf`

### "API key not found"
- Check `.env` file exists in project root
- Verify `GOOGLE_API_KEY=...` is set correctly

### "Model not found error"
- Run `python test_gemini.py` to see available models
- Update model name in `rag_tutor.py`

### Slow performance
- Reduce `top_k` (fewer chunks)
- Use smaller embedding model
- Use `gemini-2.5-flash` instead of `pro`

## 🚀 Deployment

### Local (Default)
```bash
python app.py
```

### Public Link (Gradio Share)
Change in `app.py`:
```python
demo.launch(share=True)  # Creates public URL
```

### Hugging Face Spaces
1. Create Space at https://huggingface.co/spaces
2. Upload all files
3. Add `GOOGLE_API_KEY` in Settings → Secrets
4. Space auto-deploys

## 📚 Learn More

- **RAG:** [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- **FAISS:** [Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
- **Sentence Transformers:** [Documentation](https://www.sbert.net/)
- **Google Gemini:** [API Docs](https://ai.google.dev/)
- **Gradio:** [Documentation](https://www.gradio.app/)

## 🎓 Educational Value

This project demonstrates:
- ✅ Vector embeddings and semantic search
- ✅ RAG architecture implementation
- ✅ PDF processing and text extraction
- ✅ LLM integration and prompt engineering
- ✅ Web application development
- ✅ Production-ready AI system design

## 📝 License

MIT License - Free to use and modify

## 🤝 Contributing

Improvements welcome! Key areas:
- Support for more document formats (DOCX, TXT)
- Multi-language support
- Chat history and conversation context
- Page number citations
- Question answering evaluation metrics

## 📧 Support

For issues or questions:
- Check troubleshooting section
- Review code comments
- Test with `test_gemini.py`

---

**Built with ❤️ using Python, FAISS, and Google Gemini AI**
