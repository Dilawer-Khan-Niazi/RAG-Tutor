import os
from pypdf import PdfReader
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

# ============================================================
# STEP 1: Load PDFs from folder
# ============================================================

def load_pdfs_from_folder(folder_path: str) -> List[str]:
    """
    Load all PDF files from a folder and extract text.
    
    Args:
        folder_path: Path to folder containing PDFs
        
    Returns:
        List of text content from all PDFs
    """
    all_documents = []
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"❌ Folder {folder_path} doesn't exist!")
        return all_documents
    
    # Get all PDF files
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"⚠️ No PDF files found in {folder_path}")
        return all_documents
    
    print(f"📖 Found {len(pdf_files)} PDF files")
    
    # Process each PDF
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"   Reading: {pdf_file}...")
        
        try:
            # Open and read PDF
            reader = PdfReader(pdf_path)
            text = ""
            
            # Extract text from each page
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            all_documents.append(text)
            print(f"   ✅ Extracted {len(text)} characters")
            
        except Exception as e:
            print(f"   ❌ Error reading {pdf_file}: {e}")
    
    return all_documents


# ============================================================
# STEP 2: Chunk the text
# ============================================================

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: The text to chunk
        chunk_size: Number of characters per chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        # Get chunk from start to start + chunk_size
        end = start + chunk_size
        chunk = text[start:end]
        
        # Only add non-empty chunks
        if chunk.strip():
            chunks.append(chunk.strip())
        
        # Move start position (with overlap)
        start += chunk_size - overlap
    
    return chunks


def process_all_documents(documents: List[str]) -> List[str]:
    """
    Chunk all documents and return flat list of chunks.
    """
    all_chunks = []
    
    for i, doc in enumerate(documents):
        print(f"\n✂️ Chunking document {i+1}...")
        chunks = chunk_text(doc)
        all_chunks.extend(chunks)
        print(f"   Created {len(chunks)} chunks")
    
    return all_chunks


# ============================================================
# STEP 3: Create Embeddings
# ============================================================

def create_embeddings(chunks: List[str], model_name: str = "all-MiniLM-L6-v2"):
    """
    Convert text chunks into numerical embeddings.
    
    Args:
        chunks: List of text chunks
        model_name: Name of the embedding model
        
    Returns:
        numpy array of embeddings and the model
    """
    print(f"\n🧠 Loading embedding model: {model_name}")
    print("   (This might take a minute on first run...)")
    
    # Load the model
    model = SentenceTransformer(model_name)
    
    print(f"\n🔢 Creating embeddings for {len(chunks)} chunks...")
    print("   (This may take a few minutes...)")
    
    # Create embeddings
    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print(f"✅ Created embeddings with shape: {embeddings.shape}")
    print(f"   Each chunk → {embeddings.shape[1]} numbers")
    
    return embeddings, model


# ============================================================
# STEP 4: Build Vector Database
# ============================================================

def create_vector_database(embeddings: np.ndarray, chunks: List[str], save_path: str = "vector_db"):
    """
    Create FAISS index and save it with chunks.
    
    Args:
        embeddings: Numpy array of embeddings
        chunks: Original text chunks
        save_path: Folder to save database
    """
    print(f"\n🗄️ Creating FAISS vector database...")
    
    # Get dimension of embeddings
    dimension = embeddings.shape[1]
    
    # Create FAISS index (using L2 distance)
    index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to index
    index.add(embeddings.astype('float32'))
    
    print(f"✅ Added {index.ntotal} vectors to database")
    
    # Create folder if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save index
    index_path = os.path.join(save_path, "faiss_index.bin")
    faiss.write_index(index, index_path)
    print(f"💾 Saved FAISS index to: {index_path}")
    
    # Save chunks
    chunks_path = os.path.join(save_path, "chunks.pkl")
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"💾 Saved chunks to: {chunks_path}")
    
    return index


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 RAG TUTOR - Document Processing Pipeline")
    print("=" * 60)
    
    # Step 1: Load PDFs
    print("\n📚 STEP 1: Loading PDFs...")
    documents = load_pdfs_from_folder("data")
    
    if not documents:
        print("\n❌ No documents found! Please add PDFs to 'data/' folder")
        exit()
    
    # Step 2: Chunk documents
    print("\n✂️ STEP 2: Chunking documents...")
    chunks = process_all_documents(documents)
    
    # Step 3: Create embeddings
    print("\n🔢 STEP 3: Creating embeddings...")
    embeddings, model = create_embeddings(chunks)
    
    # Step 4: Build vector database
    print("\n🗄️ STEP 4: Building vector database...")
    index = create_vector_database(embeddings, chunks)
    
    print("\n" + "=" * 60)
    print("✅ PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"📊 Statistics:")
    print(f"   - Documents processed: {len(documents)}")
    print(f"   - Total chunks: {len(chunks)}")
    print(f"   - Embedding dimensions: {embeddings.shape[1]}")
    print(f"   - Database location: vector_db/")
    # print("\n🎉 Ready to build the RAG system!")
    # print("\n💡 Next step: Run 'python rag_tutor.py' to test retrieval")
