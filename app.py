import os
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai  # <-- Import Gemini
from dotenv import load_dotenv          # <-- Import dotenv
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# --- 1. Load .env and Configure API Key ---
load_dotenv()  # <-- Load variables from .env file
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

# --- 2. Initialize App & Model ---
app = Flask(__name__)
CORS(app)

# Set up the Gemini model
generation_config = {
  "temperature": 0.2, # Lower temperature for more consistent grading
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 8192,
}
safety_settings = [
  {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
  {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
  {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
  {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
# Use a model that's good with text and reasoning, like 1.5 Pro
model = genai.GenerativeModel(
    model_name="gemini-2.5-pro-latest",
    generation_config=generation_config,
    safety_settings=safety_settings
)

# --- 3. The OCR Function (No Changes) ---
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

# --- 4. The Prompt Engineering Function (Slightly simplified for Gemini) ---
def create_grading_prompt(rubric, student_answer):
    """
    Creates the master prompt for Gemini.
    """
    # Gemini responds well to direct instruction without system/user roles
    return f"""
    You are an expert, strict, and fair teaching assistant. Your task is to grade a student's answer paper based on question-wise rubrics.

    **Strict Rules:**
    1.  You will be given <Rubrics> provided by the teacher.
    2.  You will be given <StudentAnswerText> extracted by OCR.
    3.  You **MUST** grade the student's text *only* using the criteria in the rubrics.
    4.  The rubrics are "question-wise." Match the student's answer for "Question 1" to the "Rubric for Question 1," and so on.
    5.  The student's text may have OCR errors (e.g., "Photosynthosis" instead of "Photosynthesis"). Be tolerant of such minor typos.
    6.  For **EACH** question, provide a score for **EACH** criterion.
    7.  For **EACH** criterion, you **MUST** provide a short justification for the score, quoting the student's text where possible to support your reasoning.
    8.  Calculate a total score for each question.
    9.  Format your output clearly using Markdown.

    ---
    **<Rubrics>**
    {rubric}
    ---
    **<StudentAnswerText>**
    {student_answer}
    ---
    **<EvaluationOutput>**
    [Begin your detailed, well-justified evaluation here]
    """

# --- 5. The API Endpoint (Updated for Gemini) ---
@app.route('/grade', methods=['POST'])
def grade_answer():
    try:
        if 'student_answer_pdf' not in request.files:
            return jsonify({"error": "No PDF file provided."}), 400
        
        rubric = request.form.get('rubric')
        pdf_file = request.files['student_answer_pdf']

        if not rubric:
            return jsonify({"error": "Rubric is required."}), 400
        
        # --- Step A: Perform OCR ---
        print("Starting PDF text extraction...")
        student_answer_text = extract_text_from_pdf(pdf_file)
        print("...Extraction complete.")
        
        if not student_answer_text.strip():
             return jsonify({"error": "OCR Failure: Could not detect any text in the PDF."}), 400

        # --- Step B: Create the Prompt ---
        prompt = create_grading_prompt(rubric, student_answer_text)

        # --- Step C: Call Gemini API ---
        print("Sending request to Gemini API...")
        # Send the prompt to the model
        response = model.generate_content(prompt)
        # Access the generated text
        ai_response = response.text
        print("...Response received.")

        # --- Step D: Send Response to Frontend ---
        return jsonify({"evaluation": ai_response})

    except Exception as e:
        print(f"Server Error: {e}")
        # Handle potential safety blocks from Gemini
        if "response.prompt_feedback" in locals():
            print(f"Gemini Safety Feedback: {response.prompt_feedback}")
            return jsonify({"error": f"Content blocked by safety settings: {response.prompt_feedback}"}), 500
        return jsonify({"error": str(e)}), 500

# --- 6. Run the Server ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)