import os
import io
import json
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# --- 1. Configuration & Setup ---
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)
CORS(app)

# JSON-optimized config
generation_config = {
    "temperature": 0.3, 
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
]

# Use 1.5-flash for speed/cost (Adjust model name if you have access to others)
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash", 
    generation_config=generation_config,
    safety_settings=safety_settings
)

# --- 2. Advanced Hybrid OCR ---
def extract_text_hybrid(pdf_file_storage):
    """
    Research-Grade Extraction:
    Falls back to OCR only if native text is insufficient.
    """
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
        print(f"Extraction Error: {e}")
        raise Exception(f"Failed to process PDF: {e}")

# --- 3. The Research Novelty: Actor-Critic Reflexion Loop ---
class ReflexionLoop:
    def __init__(self, rubric, student_text):
        self.rubric = rubric
        self.student_text = student_text
        self.logs = [] # To store the "thought process" for your paper

    def _call_gemini(self, prompt, role="Agent"):
        """Helper to call Gemini and log the event."""
        start = time.time()
        response = model.generate_content(prompt)
        duration = round(time.time() - start, 2)
        self.logs.append({
            "role": role,
            "latency": duration,
            "output_snippet": response.text[:100] + "..."
        })
        return json.loads(response.text)

    def run(self):
        # --- PHASE 1: The Actor (Initial Grading) ---
        actor_prompt = f"""
        You are an academic grader. Grade the student answer based STRICTLY on the rubric.
        
        <Rubric>{self.rubric}</Rubric>
        <StudentAnswer>{self.student_text}</StudentAnswer>
        
        Return JSON matching this schema:
        {{
            "evaluation": [
                {{
                    "question_id": "string", 
                    "score_awarded": float, 
                    "max_score": float, 
                    "reasoning_trace": "string", 
                    "evidence_quote": "string"
                }}
            ],
            "total_score_awarded": float,
            "total_max_score": float,
            "summary_feedback": "string"
        }}
        """
        initial_grade = self._call_gemini(actor_prompt, role="Actor (Draft)")

        # --- PHASE 2: The Critic (Auditing) ---
        # The Critic looks for specific failures: laziness, hallucinations, or harshness.
        critic_prompt = f"""
        You are a QA Auditor. Review this grading JSON against the student text and rubric.
        
        <Rubric>{self.rubric}</Rubric>
        <StudentText>{self.student_text}</StudentText>
        <ProposedGrading>{json.dumps(initial_grade)}</ProposedGrading>
        
        Task:
        1. Check if 'evidence_quote' actually exists in StudentText.
        2. Check if 'score_awarded' aligns with 'reasoning_trace'.
        
        Return JSON:
        {{
            "critique_valid": boolean, // True if the grading is acceptable
            "critique_notes": "string", // If invalid, explain what to fix
            "confidence_score": float // 0.0 to 1.0
        }}
        """
        critique = self._call_gemini(critic_prompt, role="Critic (Auditor)")

        # --- PHASE 3: The Resolver (Reflexion) ---
        # If the critic is not satisfied (or confidence is low), we RE-GRADE.
        if critique.get("critique_valid", True) and critique.get("confidence_score", 1.0) > 0.85:
            # Acceptance: The draft is good enough.
            self.logs.append({"action": "Critique Passed. Keeping Draft."})
            return initial_grade, self.logs
        else:
            # Rejection: We must regenerate using the critic's feedback.
            self.logs.append({"action": "Critique Failed. Regenerating..."})
            
            revision_prompt = f"""
            You are a Senior Grader. The previous grading attempt was rejected by the auditor.
            
            <PreviousDraft>{json.dumps(initial_grade)}</PreviousDraft>
            <AuditorFeedback>{critique.get('critique_notes')}</AuditorFeedback>
            
            Please re-grade the paper, specifically addressing the Auditor's feedback.
            Return the final JSON in the same format as the initial draft.
            """
            final_grade = self._call_gemini(revision_prompt, role="Actor (Revision)")
            return final_grade, self.logs

# --- 4. API Endpoint ---
@app.route('/grade', methods=['POST'])
def grade_answer():
    start_time = time.time()
    
    try:
        if 'student_answer_pdf' not in request.files:
            return jsonify({"error": "No PDF file provided."}), 400
        
        rubric = request.form.get('rubric')
        pdf_file = request.files['student_answer_pdf']

        if not rubric:
            return jsonify({"error": "Rubric is required."}), 400
        
        # Step A: Hybrid Extraction
        student_text, extraction_methods = extract_text_hybrid(pdf_file)
        
        if not student_text.strip():
             return jsonify({"error": "Extraction Failed: Document appears empty."}), 400

        # Step B: Run the Reflexion Loop (Agentic Workflow)
        agent_system = ReflexionLoop(rubric, student_text)
        final_json, agent_logs = agent_system.run()

        # Step C: Metrics
        total_time = round(time.time() - start_time, 2)
        
        final_output = {
            "result": final_json,
            "research_metadata": {
                "processing_time_seconds": total_time,
                "extraction_method_breakdown": extraction_methods,
                "model_used": "gemini-1.5-flash (Reflexion Arch)",
                "agent_trace": agent_logs # Pass this to UI to show "Thought Process"
            }
        }

        return jsonify(final_output)

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)  