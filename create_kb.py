# Save this as: create_kb.py
import os
import json
import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np

# --- Configuration ---
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found.")
genai.configure(api_key=GEMINI_API_KEY)

# Use a lightweight embedding model
EMBEDDING_MODEL = "models/text-embedding-004"

def extract_and_chunk_pdf(pdf_path, chunk_size=1000, overlap=100):
    """
    Reads a PDF and splits it into overlapping chunks of text.
    Overlapping helps preserve context across cuts.
    """
    print(f"📄 Processing {pdf_path}...")
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    
    # Simple sliding window chunking
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    
    print(f"✅ Created {len(chunks)} chunks.")
    return chunks

def create_vector_store(chunks):
    """
    Sends text chunks to Gemini API to get 'Embeddings' (Vector representations).
    """
    print("🧠 Generating Embeddings (this may take a moment)...")
    knowledge_base = []
    
    # Process in batches to avoid hitting API rate limits
    batch_size = 20 
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        try:
            # Generate embeddings for the batch
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=batch,
                task_type="retrieval_document"
            )
            
            # Store text + embedding pairing
            for text, embedding in zip(batch, result['embedding']):
                knowledge_base.append({
                    "text": text,
                    "embedding": embedding
                })
            print(f"   Processed batch {i} to {i+len(batch)}")
            
        except Exception as e:
            print(f"   ❌ Error on batch {i}: {e}")

    return knowledge_base

if __name__ == "__main__":
    # --- INSTRUCTIONS ---
    # 1. Put your large textbook PDF in the same folder.
    # 2. Rename it to 'textbook.pdf' or update the line below.
    PDF_FILE = "textbook.pdf" 
    OUTPUT_FILE = "knowledge_base.json"

    if not os.path.exists(PDF_FILE):
        print(f"❌ Error: {PDF_FILE} not found. Please add a PDF file.")
    else:
        text_chunks = extract_and_chunk_pdf(PDF_FILE)
        if text_chunks:
            kb_data = create_vector_store(text_chunks)
            
            # Save to disk
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(kb_data, f)
            print(f"🎉 Success! Knowledge base saved to {OUTPUT_FILE}")