import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Tuple

# Load environment variables
load_dotenv()

# ============================================================
# STEP 1: Load Vector Database
# ============================================================

def load_vector_database(db_path: str = "vector_db"):
    """
    Load the FAISS index and chunks from disk.
    
    Args:
        db_path: Path to the vector database folder
        
    Returns:
        FAISS index and list of chunks
    """
    print("📂 Loading vector database...")
    
    # Load FAISS index
    index_path = os.path.join(db_path, "faiss_index.bin")
    index = faiss.read_index(index_path)
    print(f"   ✅ Loaded FAISS index with {index.ntotal} vectors")
    
    # Load chunks
    chunks_path = os.path.join(db_path, "chunks.pkl")
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    print(f"   ✅ Loaded {len(chunks)} text chunks")
    
    return index, chunks


# ============================================================
# STEP 2: Search for Relevant Chunks
# ============================================================

def retrieve_relevant_chunks(
    query: str,
    index: faiss.Index,
    chunks: List[str],
    model: SentenceTransformer,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Search for most relevant chunks given a query.
    
    Args:
        query: User's question
        index: FAISS index
        chunks: List of text chunks
        model: Embedding model
        top_k: Number of chunks to retrieve
        
    Returns:
        List of (chunk_text, distance) tuples
    """
    print(f"\n🔍 Searching for relevant content...")
    
    # Convert query to embedding
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # Search in FAISS
    distances, indices = index.search(query_embedding.astype('float32'), top_k)
    
    # Get the chunks
    results = []
    for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
        results.append((chunks[idx], float(distance)))
        print(f"   📄 Result {i+1}: Distance = {distance:.4f}")
    
    return results


# ============================================================
# STEP 3: Generate Answer with Gemini
# ============================================================

def generate_answer(query: str, context_chunks: List[str], difficulty: str = "beginner") -> str:
    """
    Generate an answer using Google's Gemini model.
    
    Args:
        query: User's question
        context_chunks: Relevant text chunks from retrieval
        difficulty: Learning level (beginner/intermediate/advanced)
        
    Returns:
        Generated answer
    """
    print(f"\n🤖 Generating answer with Gemini...")
    
    # Configure Gemini
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "❌ Error: GOOGLE_API_KEY not found in .env file"
    
    genai.configure(api_key=api_key)
    
    # Use Gemini 2.5 Flash (fast and powerful)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Prepare context
    context = "\n\n".join([f"[Excerpt {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)])
    
    # Create prompt based on difficulty
    difficulty_instructions = {
        "beginner": "Explain in very simple terms, as if teaching someone new to this topic. Use analogies and examples.",
        "intermediate": "Explain with moderate technical detail. Assume basic understanding of the field.",
        "advanced": "Provide detailed technical explanation with advanced concepts and nuances."
    }
    
    prompt = f"""You are an expert tutor helping a student learn from their textbook.

Student's Level: {difficulty.upper()}
{difficulty_instructions.get(difficulty, difficulty_instructions["beginner"])}

Context from the textbook:
{context}

Student's Question: {query}

Instructions:
- Answer based ONLY on the context provided above
- If the context doesn't contain relevant information, say so clearly
- Explain step-by-step
- Use examples where helpful
- Be encouraging and supportive

Answer:"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"

class RAGTutor:
    """Complete RAG system for learning."""
    
    def __init__(self, db_path: str = "vector_db"):
        """Initialize the RAG tutor."""
        print("🚀 Initializing RAG Tutor...")
        
        # Load vector database
        self.index, self.chunks = load_vector_database(db_path)
        
        # Load embedding model
        print("🧠 Loading embedding model...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("   ✅ Model loaded")
        
        print("\n✅ RAG Tutor ready!")
    
    def ask(self, question: str, difficulty: str = "beginner", top_k: int = 5) -> dict:
        """
        Ask a question and get an answer.
        
        Args:
            question: The question to ask
            difficulty: Learning level
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer and sources
        """
        print("\n" + "="*60)
        print(f"❓ Question: {question}")
        print("="*60)
        
        # Retrieve relevant chunks
        results = retrieve_relevant_chunks(
            question,
            self.index,
            self.chunks,
            self.embedding_model,
            top_k
        )
        
        # Extract just the text (not distances)
        context_chunks = [chunk for chunk, _ in results]
        
        # Generate answer
        answer = generate_answer(question, context_chunks, difficulty)
        
        return {
            "question": question,
            "answer": answer,
            "sources": context_chunks,
            "difficulty": difficulty
        }


# ============================================================
# TEST THE SYSTEM
# ============================================================

if __name__ == "__main__":
    # Initialize tutor
    tutor = RAGTutor()
    
    # Test questions
    test_questions = [
        "What is machine learning?",
        "Explain supervised learning in simple terms",
        "What is the difference between classification and regression?"
    ]
    
    print("\n" + "="*60)
    print("🧪 TESTING RAG TUTOR")
    print("="*60)
    
    for question in test_questions:
        result = tutor.ask(question, difficulty="beginner")
        
        print(f"\n💡 Answer:")
        print(result["answer"])
        print("\n" + "-"*60)
        input("Press Enter for next question...")