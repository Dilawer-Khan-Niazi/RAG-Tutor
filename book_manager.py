import os
import json
import shutil
from datetime import datetime
from typing import List, Dict, Optional
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

class BookManager:
    """Manage multiple books and their vector databases."""
    
    def __init__(self, books_dir: str = "books", vector_dir: str = "vector_dbs"):
        """
        Initialize book manager.
        
        Args:
            books_dir: Directory to store uploaded PDFs
            vector_dir: Directory to store vector databases
        """
        self.books_dir = books_dir
        self.vector_dir = vector_dir
        self.metadata_file = os.path.join(books_dir, "books_metadata.json")
        
        # Create directories
        os.makedirs(books_dir, exist_ok=True)
        os.makedirs(vector_dir, exist_ok=True)
        
        # Load or create metadata
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
            self._save_metadata()
        
        # Load embedding model (shared across all books)
        print("🧠 Loading embedding model...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("   ✅ Model loaded")
    
    def _save_metadata(self):
        """Save metadata to disk."""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def _generate_book_id(self, filename: str) -> str:
        """Generate unique book ID from filename and timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(filename)[0][:20]  # First 20 chars
        return f"{base_name}_{timestamp}"
    
    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF."""
        print(f"   📖 Reading PDF...")
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        print(f"   ✅ Extracted {len(text)} characters")
        return text
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into chunks."""
        print(f"   ✂️ Chunking text...")
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start += chunk_size - overlap
        
        print(f"   ✅ Created {len(chunks)} chunks")
        return chunks
    
    def _create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Create embeddings for chunks."""
        print(f"   🔢 Creating embeddings...")
        embeddings = self.embedding_model.encode(
            chunks,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f"   ✅ Created embeddings: {embeddings.shape}")
        return embeddings
    
    def _create_vector_db(self, book_id: str, embeddings: np.ndarray, chunks: List[str]):
        """Create FAISS vector database for a book."""
        print(f"   🗄️ Building vector database...")
        
        # Create book-specific directory
        book_vector_dir = os.path.join(self.vector_dir, book_id)
        os.makedirs(book_vector_dir, exist_ok=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        # Save index
        index_path = os.path.join(book_vector_dir, "faiss_index.bin")
        faiss.write_index(index, index_path)
        
        # Save chunks
        chunks_path = os.path.join(book_vector_dir, "chunks.pkl")
        with open(chunks_path, 'wb') as f:
            pickle.dump(chunks, f)
        
        print(f"   ✅ Saved vector database to: {book_vector_dir}")
    
    def add_book(self, pdf_path: str, title: str = None, description: str = "") -> str:
        """
        Add a new book to the system.
        
        Args:
            pdf_path: Path to PDF file
            title: Book title (uses filename if not provided)
            description: Book description
            
        Returns:
            Book ID
        """
        print(f"\n{'='*60}")
        print(f"📚 Adding new book: {os.path.basename(pdf_path)}")
        print(f"{'='*60}")
        
        # Generate book ID
        filename = os.path.basename(pdf_path)
        book_id = self._generate_book_id(filename)
        
        # Copy PDF to books directory
        new_pdf_path = os.path.join(self.books_dir, f"{book_id}.pdf")
        shutil.copy(pdf_path, new_pdf_path)
        
        # Extract and process
        text = self._extract_pdf_text(pdf_path)
        chunks = self._chunk_text(text)
        embeddings = self._create_embeddings(chunks)
        self._create_vector_db(book_id, embeddings, chunks)
        
        # Save metadata
        self.metadata[book_id] = {
            "id": book_id,
            "title": title or filename,
            "description": description,
            "filename": filename,
            "added_at": datetime.now().isoformat(),
            "num_chunks": len(chunks),
            "file_size": os.path.getsize(pdf_path),
            "pdf_path": new_pdf_path
        }
        self._save_metadata()
        
        print(f"\n✅ Book added successfully!")
        print(f"   ID: {book_id}")
        print(f"   Chunks: {len(chunks)}")
        print(f"{'='*60}\n")
        
        return book_id
    
    def get_all_books(self) -> List[Dict]:
        """Get list of all books."""
        return list(self.metadata.values())
    
    def get_book(self, book_id: str) -> Optional[Dict]:
        """Get specific book metadata."""
        return self.metadata.get(book_id)
    
    def delete_book(self, book_id: str):
        """Delete a book and its vector database."""
        if book_id not in self.metadata:
            return False
        
        # Delete PDF
        if os.path.exists(self.metadata[book_id]["pdf_path"]):
            os.remove(self.metadata[book_id]["pdf_path"])
        
        # Delete vector database
        book_vector_dir = os.path.join(self.vector_dir, book_id)
        if os.path.exists(book_vector_dir):
            shutil.rmtree(book_vector_dir)
        
        # Remove from metadata
        del self.metadata[book_id]
        self._save_metadata()
        
        return True
    
    def load_book_index(self, book_id: str) -> tuple:
        """
        Load FAISS index and chunks for a specific book.
        
        Returns:
            (index, chunks) tuple
            
        Raises:
            FileNotFoundError: If index or chunks not found
            Exception: If loading fails
        """
        print(f"📂 Loading index for book: {book_id}")
        
        book_vector_dir = os.path.join(self.vector_dir, book_id)
        
        if not os.path.exists(book_vector_dir):
            raise FileNotFoundError(f"Vector database directory not found: {book_vector_dir}")
        
        # Load index
        index_path = os.path.join(book_vector_dir, "faiss_index.bin")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        
        try:
            print(f"   📇 Loading FAISS index...")
            index = faiss.read_index(index_path)
            print(f"   ✓ Index loaded ({index.ntotal} vectors)")
        except Exception as e:
            raise Exception(f"Failed to load FAISS index: {str(e)}")
        
        # Load chunks
        chunks_path = os.path.join(book_vector_dir, "chunks.pkl")
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
        
        try:
            print(f"   📦 Loading text chunks...")
            with open(chunks_path, 'rb') as f:
                chunks = pickle.load(f)
            print(f"   ✓ Chunks loaded ({len(chunks)} chunks)")
        except Exception as e:
            raise Exception(f"Failed to load chunks: {str(e)}")
        
        if not chunks or len(chunks) == 0:
            raise ValueError("Loaded chunks list is empty!")
        
        print(f"   ✅ Index ready for querying")
        return index, chunks