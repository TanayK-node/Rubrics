import os
import io
import json
import time
import numpy as np 
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer, CrossEncoder # <--- NEW IMPORT

# --- 1. Configuration & Setup ---
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)
CORS(app)

generation_config = {
    "temperature": 0.3, 
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash", # Updated to latest efficient model
    generation_config=generation_config
)

# --- 2. OCR Function (Unchanged) ---
def extract_text_hybrid(pdf_file_storage):
    full_text = ""
    method_log = []
    try:
        pdf_data = pdf_file_storage.read()
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if len(text.strip()) > 50:
                full_text += f"\n--- Page {page_num + 1} (Native) ---\n{text}\n"
                method_log.append(f"Page {page_num+1}: Native")
            else:
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img)
                full_text += f"\n--- Page {page_num + 1} (OCR) ---\n{text}\n"
                method_log.append(f"Page {page_num+1}: OCR")
        return full_text, method_log
    except Exception as e:
        raise Exception(f"Failed to process PDF: {e}")

# --- 3. [UPDATED] Local Knowledge Retriever with Re-ranking ---
class KnowledgeRetriever:
    def __init__(self, kb_path="knowledge_base.json"):
        self.kb_path = kb_path
        self.knowledge_base = []
        self.is_ready = False
        
        # 1. Load Bi-Encoder (Fast Retrieval)
        print("Loading Bi-Encoder (SentenceTransformer)...")
        self.bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 2. Load Cross-Encoder (High Accuracy Re-ranking)
        # 'ms-marco-MiniLM-L-6-v2' is a standard research baseline for re-ranking
        print("Loading Cross-Encoder (Re-ranker)...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        self.load_kb()

    def load_kb(self):
        if os.path.exists(self.kb_path):
            try:
                with open(self.kb_path, 'r') as f:
                    self.knowledge_base = json.load(f)
                
                # Pre-convert embeddings to numpy for speed
                for entry in self.knowledge_base:
                    entry['embedding'] = np.array(entry['embedding'])
                    
                print(f"📚 RAG System: Loaded {len(self.knowledge_base)} chunks from disk.")
                self.is_ready = True
            except Exception as e:
                print(f"⚠️ RAG Error: Failed to load knowledge base: {e}")
        else:
            print("⚠️ RAG Warning: knowledge_base.json not found.")

    def retrieve(self, query, top_k=5, candidate_pool_size=20):
        """
        Two-Stage Retrieval:
        1. Bi-Encoder: Finds top 20 'likely' chunks (Fast)
        2. Cross-Encoder: Re-ranks them to find top 5 'actual' matches (Accurate)
        """
        if not self.is_ready: return ""

        try:
            # --- Stage 1: Fast Vector Search (Bi-Encoder) ---
            query_embedding = self.bi_encoder.encode(query)
            
            candidates = []
            for entry in self.knowledge_base:
                # Cosine Similarity
                score = np.dot(query_embedding, entry['embedding'])
                candidates.append({
                    "id": entry['id'],
                    "text": entry['text'],
                    "initial_score": score
                })
            
            # Sort by vector score and take top N candidates
            candidates.sort(key=lambda x: x["initial_score"], reverse=True)
            top_candidates = candidates[:candidate_pool_size]

            # --- Stage 2: Re-ranking (Cross-Encoder) ---
            # Create pairs [ [query, doc1], [query, doc2], ... ]
            sentence_combinations = [[query, doc['text']] for doc in top_candidates]
            
            # Predict similarity scores
            cross_scores = self.cross_encoder.predict(sentence_combinations)
            
            # Attach new scores
            for i, doc in enumerate(top_candidates):
                doc['cross_score'] = cross_scores[i]
                
            # Sort by CROSS score (The "Gold Standard" rank)
            top_candidates.sort(key=lambda x: x['cross_score'], reverse=True)
            
            # Select Final Top K
            final_results = top_candidates[:top_k]
            
            # Format for the Agent
            context_str = "\n".join([f"--- Source: {x['id']} (Relevance: {x['cross_score']:.2f}) ---\n{x['text']}" for x in final_results])
            return context_str

        except Exception as e:
            print(f"Retrieval Error: {e}")
            return ""

# Initialize Global Retriever
retriever = KnowledgeRetriever()

# --- 4. The Agentic Reflexion Loop (Unchanged) ---
class ReflexionLoop:
    def __init__(self, rubric, student_text):
        self.rubric = rubric
        self.student_text = student_text
        self.logs = [] 
        
        # Retrieve Truth using the new Two-Stage Retriever
        self.truth_context = retriever.retrieve(rubric)
        if self.truth_context:
            self.logs.append({"system_action": "RAG_RETRIEVAL", "status": "Success", "context_length": len(self.truth_context)})
        else:
            self.logs.append({"system_action": "RAG_RETRIEVAL", "status": "Skipped (No Knowledge Base)"})

    def _call_gemini(self, prompt, role="Agent"):
        start = time.time()
        try:
            response = model.generate_content(prompt)
            duration = round(time.time() - start, 2)
            self.logs.append({
                "role": role,
                "latency": duration,
                "output_snippet": response.text[:100] + "..."
            })
            return json.loads(response.text)
        except Exception as e:
            # Error handling for JSON parsing or API issues
            self.logs.append({"role": role, "error": str(e)})
            return {"error": str(e)}

    def run(self):
        # PHASE 1: The Actor (Draft)
        actor_prompt = f"""
        You are an academic grader. 
        
        <Instructions>
        1. Grade the student answer based on the <Rubric>.
        2. Use the <TruthContext> (Textbook Material) to verify facts.
        3. If the student's answer contradicts the <TruthContext>, deduct marks.
        </Instructions>
        
        <TruthContext>
        {self.truth_context}
        </TruthContext>

        <Rubric>{self.rubric}</Rubric>
        <StudentAnswer>{self.student_text}</StudentAnswer>
        
        Return JSON matching schema:
        {{ "evaluation": [ {{ "question_id": "string", "score_awarded": float, "max_score": float, "reasoning_trace": "string", "evidence_quote": "string" }} ], "total_score_awarded": float, "total_max_score": float, "summary_feedback": "string" }}
        """
        initial_grade = self._call_gemini(actor_prompt, role="Actor (Draft)")

        # PHASE 2: The Critic (Auditor)
        critic_prompt = f"""
        You are a QA Auditor. 
        
        <Inputs>
        <TruthContext>{self.truth_context}</TruthContext>
        <StudentText>{self.student_text}</StudentText>
        <ProposedGrading>{json.dumps(initial_grade)}</ProposedGrading>
        </Inputs>
        
        Task:
        1. Did the grader ignore the Textbook facts?
        2. Are the quotes real?
        
        Return JSON: {{ "critique_valid": boolean, "critique_notes": "string", "confidence_score": float }}
        """
        critique = self._call_gemini(critic_prompt, role="Critic (Auditor)")

        # PHASE 3: Reflexion
        if critique.get("critique_valid", True) and critique.get("confidence_score", 1.0) > 0.85:
            self.logs.append({"action": "Critique Passed. Keeping Draft."})
            return initial_grade, self.logs
        else:
            self.logs.append({"action": "Critique Failed. Regenerating..."})
            revision_prompt = f"""
            You are a Senior Grader. Re-grade this based on the Auditor's feedback.
            <AuditorFeedback>{critique.get('critique_notes')}</AuditorFeedback>
            <Draft>{json.dumps(initial_grade)}</Draft>
            """
            final_grade = self._call_gemini(revision_prompt, role="Actor (Revision)")
            return final_grade, self.logs

# --- 5. API Endpoint (Standard) ---
@app.route('/grade', methods=['POST'])
def grade_answer():
    start_time = time.time()
    try:
        if 'student_answer_pdf' not in request.files:
            return jsonify({"error": "No PDF file provided."}), 400
        
        rubric = request.form.get('rubric')
        pdf_file = request.files['student_answer_pdf']
        if not rubric: return jsonify({"error": "Rubric is required."}), 400
        
        student_text, extraction_methods = extract_text_hybrid(pdf_file)
        if not student_text.strip(): return jsonify({"error": "Extraction Failed."}), 400

        agent_system = ReflexionLoop(rubric, student_text)
        final_json, agent_logs = agent_system.run()

        total_time = round(time.time() - start_time, 2)
        final_output = {
            "result": final_json,
            "research_metadata": {
                "processing_time_seconds": total_time,
                "extraction_method_breakdown": extraction_methods,
                "model_used": "gemini-2.0-flash (Hybrid RAG + Reflexion)",
                "agent_trace": agent_logs 
            }
        }
        return jsonify(final_output)

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)