import os
import json
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer # <--- NEW: Local Embeddings

# Configuration
PDF_PATH = "textbook.pdf" 
OUTPUT_FILE = "knowledge_base.json"

def create_knowledge_base():
    # 1. Load Local Model (Downloads once, then runs offline)
    print("Download/Loading local embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 
    
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: {PDF_PATH} not found.")
        return

    print(f"📖 Reading {PDF_PATH}...")
    doc = fitz.open(PDF_PATH)
    
    knowledge_base = []
    
    # Process pages
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if len(text) < 100: continue 
        
        print(f"   Processing Page {page_num + 1}...")
        
        try:
            # 2. Generate Embedding LOCALLY
            # .tolist() converts the numpy array to a standard list for JSON saving
            vector = embedding_model.encode(text).tolist()
            
            knowledge_base.append({
                "id": f"Page {page_num + 1}",
                "text": text,
                "embedding": vector
            })
        except Exception as e:
            print(f"   ⚠️ Error on page {page_num}: {e}")

    # 3. Save to disk
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(knowledge_base, f)
    
    print(f"✅ Success! Knowledge Base saved to {OUTPUT_FILE} ({len(knowledge_base)} pages indexed locally).")

if __name__ == "__main__":
    create_knowledge_base()