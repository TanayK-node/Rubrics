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

# We use a JSON-optimized configuration
generation_config = {
    "temperature": 0.3,  # Slight creativity allowed for reasoning
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json", # <--- FORCE JSON OUTPUT
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
]

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash", # Use 1.5-flash for speed/cost efficiency in research
    generation_config=generation_config,
    safety_settings=safety_settings
)

# --- 2. Advanced Hybrid OCR (Methodology Improvement) ---
def extract_text_hybrid(pdf_file_storage):
    """
    Research-Grade Extraction:
    1. Attempts low-cost native text extraction first.
    2. Falls back to expensive OCR only if native text is insufficient (< 50 chars).
    Returns: Tuple (text, method_used) for your research data logs.
    """
    full_text = ""
    method_log = []
    
    try:
        pdf_data = pdf_file_storage.read()
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        
        for page_num, page in enumerate(doc):
            # Attempt 1: Native Extraction
            text = page.get_text()
            
            # Metric: Text Density Check
            if len(text.strip()) > 50:
                full_text += f"\n--- Page {page_num + 1} (Native) ---\n{text}\n"
                method_log.append(f"Page {page_num+1}: Native")
            else:
                # Attempt 2: OCR Fallback
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img)
                full_text += f"\n--- Page {page_num + 1} (OCR) ---\n{text}\n"
                method_log.append(f"Page {page_num+1}: OCR")

        return full_text, method_log
        
    except Exception as e:
        print(f"Extraction Error: {e}")
        raise Exception(f"Failed to process PDF: {e}")

# --- 3. The 'Critic' Architecture (Novelty) ---
def grade_with_critic_loop(rubric, student_text):
    """
    Implements a 'Reflexion' loop:
    1. Agent generates initial grade.
    2. (Optional future step) Critic reviews it.
    For now, we implement a structured Chain-of-Thought prompt.
    """
    
    # Prompt 1: The Grader
    # We ask for a "Reasoning Trace" before the score to improve accuracy (CoT).
    prompt = f"""
    You are an automated academic evaluator. Your goal is to grade a student answer based STRICTLY on the provided rubric.

    **Input Data:**
    <Rubric>
    {rubric}
    </Rubric>

    <StudentAnswer>
    {student_text}
    </StudentAnswer>

    **Task:**
    1. Analyze the student's answer against the rubric criteria.
    2. Extract specific quotes from the student's text that support your evaluation.
    3. Assign a score based ONLY on the rubric.
    
    **Output Schema (JSON):**
    You must return a JSON object with this exact structure:
    {{
      "evaluation": [
        {{
          "question_id": "string (e.g., Q1)",
          "score_awarded": float,
          "max_score": float,
          "reasoning_trace": "string (Explain the logic for the score)",
          "evidence_quote": "string (Direct quote from student text)",
          "improvement_suggestion": "string (How the student could get full marks)"
        }}
      ],
      "total_score_awarded": float,
      "total_max_score": float,
      "summary_feedback": "string"
    }}
    """
    
    response = model.generate_content(prompt)
    return response.text

# --- 4. API Endpoint ---
@app.route('/grade', methods=['POST'])
def grade_answer():
    start_time = time.time() # Metric: Latency
    
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

        # Step B: AI Grading (The "Black Box")
        # We parse the string response into a real JSON object
        raw_response = grade_with_critic_loop(rubric, student_text)
        json_response = json.loads(raw_response)

        # Step C: Calculate Research Metrics
        processing_time = round(time.time() - start_time, 2)
        
        # Step D: Construct Final Response
        # We include 'metadata' for your research paper logging
        final_output = {
            "result": json_response,
            "research_metadata": {
                "processing_time_seconds": processing_time,
                "extraction_method_breakdown": extraction_methods,
                "model_used": "gemini-1.5-flash",
                "character_count": len(student_text)
            }
        }

        return jsonify(final_output)

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)