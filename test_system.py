"""
Quick test script to verify all fixes are working.
Run this to validate the improved system.
"""

import os
import sys
import json
from pathlib import Path

def check_imports():
    """Check if all required packages are installed."""
    print("🔍 Checking dependencies...")
    
    required = {
        'transformers': 'transformers',
        'sentence-transformers': 'sentence-transformers',
        'faiss': 'faiss-cpu',
        'pypdf': 'pypdf',
        'gradio': 'gradio',
        'google.generativeai': 'google-generativeai',
        'dotenv': 'python-dotenv',
        'sqlalchemy': 'sqlalchemy',
        'PIL': 'pillow'
    }
    
    missing = []
    for import_name, package_name in required.items():
        try:
            __import__(import_name)
            print(f"   ✓ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name} - MISSING")
            missing.append(package_name)
    
    return len(missing) == 0, missing

def check_environment():
    """Check if environment variables are set."""
    print("\n🔐 Checking environment variables...")
    
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            if 'GOOGLE_API_KEY' in f.read():
                print("   ✓ GOOGLE_API_KEY found in .env")
                return True
            else:
                print("   ❌ GOOGLE_API_KEY not found in .env")
                return False
    else:
        print("   ❌ .env file not found")
        return False

def check_files():
    """Check if all required files exist."""
    print("\n📁 Checking required files...")
    
    files = [
        'app.py',
        'agentic_rag.py',
        'book_manager.py',
        'chat_storage.py',
        'process_documents.py',
        'rag_tutor.py',
        'requirements.txt',
        'cleanup_duplicates.py',
        'FIXES_AND_IMPROVEMENTS.md'
    ]
    
    missing = []
    for file in files:
        if os.path.exists(file):
            print(f"   ✓ {file}")
        else:
            print(f"   ❌ {file} - MISSING")
            missing.append(file)
    
    return len(missing) == 0, missing

def check_directories():
    """Check if required directories exist."""
    print("\n📂 Checking required directories...")
    
    dirs = ['books', 'vector_dbs', 'chat_history', 'data']
    
    for dir_name in dirs:
        if os.path.exists(dir_name):
            file_count = len(os.listdir(dir_name))
            print(f"   ✓ {dir_name}/ ({file_count} items)")
        else:
            print(f"   ⚠️ {dir_name}/ - Creating...")
            os.makedirs(dir_name, exist_ok=True)
    
    return True

def check_metadata():
    """Check books metadata."""
    print("\n📚 Checking books metadata...")
    
    metadata_file = 'books/books_metadata.json'
    if not os.path.exists(metadata_file):
        print("   ⚠️ No books uploaded yet")
        return True
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"   ✓ Books in metadata: {len(metadata)}")
        
        invalid = []
        for book_id, book_info in metadata.items():
            if not book_info.get('pdf_path'):
                invalid.append(f"Missing pdf_path: {book_id}")
            elif not os.path.exists(book_info['pdf_path']):
                invalid.append(f"PDF missing: {book_id}")
            
            # Check vector DB
            vector_dir = f"vector_dbs/{book_id}"
            if not os.path.exists(vector_dir):
                invalid.append(f"No vector DB: {book_id}")
            else:
                if not os.path.exists(f"{vector_dir}/faiss_index.bin"):
                    invalid.append(f"No FAISS index: {book_id}")
                if not os.path.exists(f"{vector_dir}/chunks.pkl"):
                    invalid.append(f"No chunks file: {book_id}")
        
        if invalid:
            print("\n   ⚠️ Issues found:")
            for issue in invalid:
                print(f"      - {issue}")
            print("\n   💡 Tip: Run cleanup_duplicates.py to fix metadata")
        else:
            print("   ✓ All books properly indexed")
        
        return len(invalid) == 0
    except Exception as e:
        print(f"   ❌ Error reading metadata: {str(e)}")
        return False

def check_code_quality():
    """Check if improvements are in place."""
    print("\n🔧 Checking improvements...")
    
    improvements = {
        'agentic_rag.py': [
            '_call_gemini_with_retry',
            'logger',
            'exponential backoff'
        ],
        'app.py': [
            'logging.basicConfig',
            'multi-stage validation',
            'detailed error messages'
        ],
        'book_manager.py': [
            'FileNotFoundError',
            'validate FAISS index'
        ]
    }
    
    all_good = True
    for file, keywords in improvements.items():
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            missing = [kw for kw in keywords if kw not in content]
            if missing:
                print(f"   ⚠️ {file}: Missing {missing}")
                all_good = False
            else:
                print(f"   ✓ {file}: All improvements present")
        except Exception as e:
            print(f"   ❌ {file}: Error checking - {str(e)}")
            all_good = False
    
    return all_good

def main():
    """Run all checks."""
    print("="*60)
    print("🧪 RAG Learning Tutor - System Check")
    print("="*60)
    
    checks = [
        ("Dependencies", check_imports),
        ("Environment", check_environment),
        ("Files", check_files),
        ("Directories", check_directories),
        ("Metadata", check_metadata),
        ("Code Quality", check_code_quality)
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            result = check_func()
            if isinstance(result, tuple):
                results[check_name] = result[0]
            else:
                results[check_name] = result
        except Exception as e:
            print(f"\n   ❌ Check failed: {str(e)}")
            results[check_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {check_name}")
    
    print(f"\n{passed}/{total} checks passed")
    
    if passed == total:
        print("\n✅ System is ready! You can start using the app.")
        print("\nNext steps:")
        print("1. Run: python app.py")
        print("2. Open http://localhost:7860 in browser")
        print("3. Upload a PDF book")
        print("4. Ask questions!")
    else:
        print("\n⚠️ Some checks failed. Please:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check .env file has GOOGLE_API_KEY")
        print("3. Run cleanup_duplicates.py if metadata issues")
    
    print("="*60)

if __name__ == "__main__":
    main()
