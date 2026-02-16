import os
import json
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

# --- Configuration ---
PDF_PATH = "textbook.pdf"
OUTPUT_FILE = "knowledge_base.json"
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200 # Overlap to preserve context

def create_sliding_window_chunks(text, chunk_size, overlap):
    """
    Splits text into overlapping chunks.
    Simple sliding window approach (Research Standard for basic RAG).
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # basic cleanup
        chunk = chunk.replace('\n', ' ').strip()
        
        if len(chunk) > 50: # Filter tiny chunks
            chunks.append(chunk)
        
        # Move forward by step size (chunk_size - overlap)
        start += (chunk_size - overlap)
        
    return chunks

def create_knowledge_base():
    # 1. Load Embedding Model
    print("Download/Loading local embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 
    
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: {PDF_PATH} not found.")
        return

    print(f"📖 Reading {PDF_PATH}...")
    doc = fitz.open(PDF_PATH)
    
    # 2. Extract ALL Text First (Preserves cross-page context)
    full_text = ""
    page_map = [] # To track roughly where text came from
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        full_text += text
        # Store page break markers if you want to trace back later (optional)
        # full_text += f" [PAGE_BREAK_{page_num+1}] " 

    print(f"   Extracted {len(full_text)} characters. Generating chunks...")

    # 3. Create Semantic Chunks
    text_chunks = create_sliding_window_chunks(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"   Created {len(text_chunks)} chunks (Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}).")

    knowledge_base = []
    
    # 4. Generate Embeddings for Chunks
    for i, chunk in enumerate(text_chunks):
        if i % 50 == 0: print(f"   Embedding chunk {i}/{len(text_chunks)}...")
        
        try:
            vector = embedding_model.encode(chunk).tolist()
            knowledge_base.append({
                "id": f"chunk_{i}",
                "text": chunk,
                "embedding": vector
            })
        except Exception as e:
            print(f"   ⚠️ Error on chunk {i}: {e}")

    # 5. Save to disk
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(knowledge_base, f)
    
    print(f"✅ Success! Knowledge Base saved to {OUTPUT_FILE}.")

if __name__ == "__main__":
    create_knowledge_base()