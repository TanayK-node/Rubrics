import os
import io
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# --- 1. Config ---
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)
CORS(app)

# Lower temperature for precision, but enough for the model to "search" the text
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash", 
    generation_config={"temperature": 0.3},
    safety_settings=[
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    ]
)

# --- 2. OCR Function (Unchanged) ---
def extract_text_from_pdf(pdf_file_storage):
    extracted_text = ""
    try:
        pdf_data = pdf_file_storage.read()
        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            text = pytesseract.image_to_string(image, lang='eng')
            extracted_text += f"\n--- Page {page_num + 1} ---\n{text}\n"
        pdf_document.close()
        return extracted_text
    except Exception as e:
        print(f"Error during OCR: {e}")
        raise Exception(f"Failed to process PDF: {e}")

# --- 3. The Adaptive Multi-Agent Prompts ---

def get_agent_1_grader_prompt(rubric, student_answer):
    return f"""
    ROLE: You are 'Agent 1', an expert academic evaluator. You are capable of grading both Structured Exams (Q&A) and Unstructured Submissions (Reports, Essays, Papers).

    INPUTS:
    --- RUBRIC ---
    {rubric}
    --- STUDENT SUBMISSION ---
    {student_answer}

    TASK:
    1. **Analyze the Rubric**: Determine if the rubric is looking for specific answers (e.g., "Q1", "Q2") or broad sections (e.g., "Introduction", "Methodology", "Grammar").
    2. **Evaluate**: For EACH criterion/question in the rubric:
       - **Locate Evidence**: 
         - If it's a Q&A: Find the specific answer to that question.
         - If it's a Report: Scan the ENTIRE text to find relevant evidence for that criterion (e.g., for "Clarity," look at the whole text; for "Methodology," find that section).
       - **Assign Score**: Based strictly on the rubric.
    
    OUTPUT FORMAT (Strict Markdown):
    For each Rubric Item/Question:
    ### [Item Name/Number]
    - **Type**: [Q&A OR Report-Criterion]
    - **Evidence**: "[Quote relevant text from student. If scattered, summarize key evidence.]"
    - **Reasoning**: Compare evidence vs. rubric requirements.
    - **Score**: [Points awarded] / [Max Points]
    """

def get_agent_2_critic_prompt(rubric, student_answer, agent_1_output):
    return f"""
    ROLE: You are 'Agent 2', the Lead Auditor.
    TASK: Critique the evaluation performed by Agent 1 to ensure fairness and accuracy.

    CONTEXT:
    Agent 1 has just graded a submission. It might be a Q&A sheet OR a Report/Essay. 
    
    INPUTS:
    - Original Rubric: {rubric}
    - Student Submission: {student_answer}
    - Agent 1's Evaluation: {agent_1_output}
    
    CRITIQUE CHECKLIST:
    1. **Evidence Mapping**: Did Agent 1 find the correct section? (e.g., Did they grade the 'Conclusion' based on the 'Introduction' text by mistake?)
    2. **Hallucination**: Did Agent 1 quote text that isn't there?
    3. **Rubric Adherence**: If the rubric asks for "Cititations," did Agent 1 actually check for citations?
    
    OUTPUT:
    - If Agent 1 is 100% solid, output: "AGREED".
    - If there are errors, write a "CRITIQUE REPORT" listing specific criteria where Agent 1 failed and provide the corrected reasoning.
    """

def get_agent_3_adjudicator_prompt(rubric, agent_1_output, agent_2_critique):
    return f"""
    ROLE: You are 'Agent 3', the Final Adjudicator.
    TASK: Finalize the grade report.

    INPUTS:
    - Rubric: {rubric}
    - Agent 1 Draft: {agent_1_output}
    - Agent 2 Critique: {agent_2_critique}
    
    INSTRUCTIONS:
    - Review the critique. 
    - If Agent 2 found specific errors (e.g., "Agent 1 missed the methodology section"), YOU must fix the score for that section.
    - Produce the FINAL, clean grading report.
    - Format it beautifully in Markdown.
    """

# --- 4. The Logic Flow ---
@app.route('/grade', methods=['POST'])
def grade_answer():
    try:
        if 'student_answer_pdf' not in request.files:
            return jsonify({"error": "No PDF file provided."}), 400
        
        rubric = request.form.get('rubric')
        pdf_file = request.files['student_answer_pdf']

        # 1. OCR
        print("--- Step 0: OCR Extraction ---")
        student_text = extract_text_from_pdf(pdf_file)
        if not student_text.strip():
             return jsonify({"error": "OCR Failure: No text detected."}), 400

        # 2. Agent 1: The Adaptive Grader
        print("--- Step 1: Agent 1 (Grader) is evaluating... ---")
        prompt_1 = get_agent_1_grader_prompt(rubric, student_text)
        resp_1 = model.generate_content(prompt_1)
        out_1 = resp_1.text
        print("Agent 1 finished.")

        # 3. Agent 2: The Critic
        print("--- Step 2: Agent 2 (Critic) is auditing... ---")
        prompt_2 = get_agent_2_critic_prompt(rubric, student_text, out_1)
        resp_2 = model.generate_content(prompt_2)
        out_2 = resp_2.text
        print(f"Agent 2 Verdict: {out_2[:50]}...") 

        # 4. Agent 3: The Adjudicator
        print("--- Step 3: Agent 3 (Adjudicator) is finalizing... ---")
        prompt_3 = get_agent_3_adjudicator_prompt(rubric, out_1, out_2)
        resp_3 = model.generate_content(prompt_3)
        final_output = resp_3.text
        print("Agent 3 finished.")

        # Return full trace for research transparency
        full_trace = f"""
--- 🕵️ INTERNAL REASONING TRACE (Research View) ---

[AGENT 1 - DRAFT EVALUATION]
{out_1}

[AGENT 2 - AUDIT REPORT]
{out_2}

--- 📝 FINAL EVALUATION ---
{final_output}
        """

        return jsonify({"evaluation": full_trace})

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)