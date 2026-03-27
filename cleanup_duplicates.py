"""
Utility script to remove duplicate books and clean up the system.
Run this to consolidate your library after the fixes.
"""

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
import hashlib

def get_file_hash(filepath: str) -> str:
    """Calculate MD5 hash of a file."""
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def find_duplicates(books_metadata: dict, books_dir: str) -> dict:
    """
    Find duplicate books based on file hash.
    
    Returns:
        {
            "file_hash": {
                "count": int,
                "keeps": [book_ids],  # Keep these
                "removes": [book_ids]  # Remove these
            }
        }
    """
    file_hashes = defaultdict(list)
    
    print("🔍 Analyzing books for duplicates...")
    
    for book_id, book_info in books_metadata.items():
        pdf_path = book_info.get('pdf_path')
        if not pdf_path:
            print(f"  ⚠️ No PDF path for {book_id}, looking for backup...")
            # Try to find PDF
            book_pdf = os.path.join(books_dir, f"{book_id}.pdf")
            if os.path.exists(book_pdf):
                pdf_path = book_pdf
            else:
                print(f"  ❌ Cannot find PDF for {book_id}")
                continue
        
        if not os.path.exists(pdf_path):
            print(f"  ❌ PDF not found: {pdf_path}")
            continue
        
        try:
            file_hash = get_file_hash(pdf_path)
            file_hashes[file_hash].append(book_id)
            print(f"  ✓ {book_id}: {file_hash[:8]}...")
        except Exception as e:
            print(f"  ❌ Error hashing {book_id}: {str(e)}")
    
    # Find actual duplicates
    duplicates = {}
    for file_hash, book_ids in file_hashes.items():
        if len(book_ids) > 1:
            # Keep the oldest (first uploaded), remove the rest
            books_with_dates = [
                (book_id, books_metadata[book_id]['added_at'])
                for book_id in book_ids
            ]
            books_with_dates.sort(key=lambda x: x[1])
            
            duplicates[file_hash] = {
                "count": len(book_ids),
                "keeps": [books_with_dates[0][0]],
                "removes": [bid for bid, _ in books_with_dates[1:]]
            }
    
    return duplicates

def remove_duplicates(books_metadata: dict, duplicates: dict, books_dir: str, vector_dir: str, dry_run: bool = True):
    """
    Remove duplicate books.
    
    Args:
        books_metadata: Book metadata dictionary
        duplicates: Duplicates from find_duplicates()
        books_dir: Books directory
        vector_dir: Vector databases directory
        dry_run: If True, only show what would be deleted
    """
    total_removed = 0
    total_freed = 0
    
    for file_hash, dup_info in duplicates.items():
        print(f"\n📦 Hash: {file_hash[:8]}...")
        print(f"   Duplicates: {dup_info['count']}")
        print(f"   Keep: {dup_info['keeps'][0]}")
        
        for book_id in dup_info['removes']:
            book_info = books_metadata[book_id]
            print(f"   Remove: {book_id}")
            
            # Calculate space freed
            file_size = book_info.get('file_size', 0)
            vector_size = 0
            vector_dir_path = os.path.join(vector_dir, book_id)
            if os.path.exists(vector_dir_path):
                for root, dirs, files in os.walk(vector_dir_path):
                    for file in files:
                        vector_size += os.path.getsize(os.path.join(root, file))
            
            total_freed += file_size + vector_size
            print(f"      PDF: {file_size / 1024:.1f} KB + Vector DB: {vector_size / 1024:.1f} KB")
            
            if not dry_run:
                # Delete PDF
                pdf_path = book_info.get('pdf_path')
                if pdf_path and os.path.exists(pdf_path):
                    try:
                        os.remove(pdf_path)
                        print(f"      ✓ Deleted PDF")
                    except Exception as e:
                        print(f"      ❌ Failed to delete PDF: {str(e)}")
                
                # Delete vector database
                if os.path.exists(vector_dir_path):
                    try:
                        shutil.rmtree(vector_dir_path)
                        print(f"      ✓ Deleted vector database")
                    except Exception as e:
                        print(f"      ❌ Failed to delete vector DB: {str(e)}")
                
                # Remove from metadata
                del books_metadata[book_id]
                total_removed += 1
                print(f"      ✓ Removed from metadata")
    
    if dry_run:
        print(f"\n📊 DRY RUN RESULTS:")
        print(f"   Would remove: {len([bid for h, info in duplicates.items() for bid in info['removes']])} books")
        print(f"   Would free: {total_freed / (1024*1024):.1f} MB")
        print(f"\n   ⚠️ Run with dry_run=False to actually delete")
    else:
        print(f"\n✅ CLEANUP COMPLETE:")
        print(f"   Removed: {total_removed} books")
        print(f"   Freed: {total_freed / (1024*1024):.1f} MB")

def cleanup_stray_vector_dbs(books_metadata: dict, vector_dir: str, dry_run: bool = True):
    """
    Remove vector databases for books that no longer exist in metadata.
    """
    print(f"\n🧹 Looking for stray vector databases...")
    
    valid_book_ids = set(books_metadata.keys())
    stray_count = 0
    stray_size = 0
    
    if os.path.exists(vector_dir):
        for item in os.listdir(vector_dir):
            item_path = os.path.join(vector_dir, item)
            if os.path.isdir(item_path) and item not in valid_book_ids:
                # Calculate size
                dir_size = 0
                for root, dirs, files in os.walk(item_path):
                    for file in files:
                        dir_size += os.path.getsize(os.path.join(root, file))
                
                print(f"   Stray: {item} ({dir_size / 1024:.1f} KB)")
                stray_count += 1
                stray_size += dir_size
                
                if not dry_run:
                    try:
                        shutil.rmtree(item_path)
                        print(f"      ✓ Deleted")
                    except Exception as e:
                        print(f"      ❌ Failed: {str(e)}")
    
    if stray_count == 0:
        print(f"   ✓ No stray databases found")
    elif dry_run:
        print(f"\n   Would remove: {stray_count} stray databases ({stray_size / (1024*1024):.1f} MB)")

def main():
    """Main cleanup flow."""
    print("="*60)
    print("🧹 RAG Learning Tutor - Duplicate Cleanup Utility")
    print("="*60)
    
    books_dir = "books"
    vector_dir = "vector_dbs"
    metadata_file = os.path.join(books_dir, "books_metadata.json")
    
    # Load metadata
    if not os.path.exists(metadata_file):
        print("❌ Metadata file not found!")
        return
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        books_metadata = json.load(f)
    
    print(f"\n📖 Total books in metadata: {len(books_metadata)}")
    
    # Find duplicates
    duplicates = find_duplicates(books_metadata, books_dir)
    
    if not duplicates:
        print("\n✓ No duplicates found!")
    else:
        print(f"\n⚠️ Found {len(duplicates)} duplicate groups")
        
        # Show summary
        total_dups = sum(len(v['removes']) for v in duplicates.values())
        print(f"   Total duplicate entries: {total_dups}")
        
        # DRY RUN
        print("\n" + "="*60)
        print("DRY RUN (preview of changes):")
        print("="*60)
        remove_duplicates(books_metadata, duplicates, books_dir, vector_dir, dry_run=True)
        cleanup_stray_vector_dbs(books_metadata, vector_dir, dry_run=True)
        
        # Ask for confirmation
        response = input("\n⚠️ Do you want to proceed with cleanup? (yes/no): ").strip().lower()
        if response == 'yes':
            print("\n" + "="*60)
            print("EXECUTING CLEANUP:")
            print("="*60)
            remove_duplicates(books_metadata, duplicates, books_dir, vector_dir, dry_run=False)
            cleanup_stray_vector_dbs(books_metadata, vector_dir, dry_run=False)
            
            # Save cleaned metadata
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(books_metadata, f, indent=2, ensure_ascii=False)
            print("\n✅ Metadata saved!")
        else:
            print("\n❌ Cleanup cancelled")
    
    print("\n" + "="*60)
    print("✅ Cleanup utility finished")
    print("="*60)

if __name__ == "__main__":
    main()
