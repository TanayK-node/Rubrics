import os
import io
import ast
import json
import time
import uuid
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from google import genai
from google.genai import types
from dotenv import load_dotenv
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer

# --- 1. Configuration & Setup ---
load_dotenv()
app = Flask(__name__)
CORS(app)

generation_config = {
    "temperature": 0.3,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
}

# Global dictionary to store background task statuses
grading_tasks = {}


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_MODEL = None
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_CONTEXT_CHUNKS = 6
MAX_CONTEXT_CHARS = 12000
QUERY_TEXT_LIMIT = 4000


def get_embedding_model():
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return EMBEDDING_MODEL


def create_sliding_window_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].replace('\n', ' ').strip()

        if len(chunk) > 50:
            chunks.append(chunk)

        start += max(1, chunk_size - overlap)

    return chunks


def build_kb_chunks(kb_files_data):
    kb_chunks = []

    for kb_file in kb_files_data:
        file_text, _ = extract_text_hybrid(kb_file["content"])
        file_chunks = create_sliding_window_chunks(file_text)

        for chunk_index, chunk_text in enumerate(file_chunks):
            kb_chunks.append({
                "id": f"{kb_file['filename']}_chunk_{chunk_index}",
                "source": kb_file["filename"],
                "text": chunk_text,
            })

    return kb_chunks


def select_top_k_chunks(kb_chunks, query_text, top_k=TOP_K_CONTEXT_CHUNKS):
    if not kb_chunks:
        return []

    embedding_model = get_embedding_model()
    query_text = (query_text or "")[:QUERY_TEXT_LIMIT]

    chunk_texts = [chunk["text"] for chunk in kb_chunks]
    embeddings = embedding_model.encode(
        [query_text] + chunk_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = embeddings[0]
    chunk_embeddings = embeddings[1:]
    scores = np.dot(chunk_embeddings, query_embedding)

    ranked_indices = np.argsort(-scores)
    selected_chunks = []
    total_chars = 0

    for index in ranked_indices[:top_k]:
        chunk = kb_chunks[int(index)]
        chunk_text = chunk["text"]

        if total_chars + len(chunk_text) > MAX_CONTEXT_CHARS:
            break

        selected_chunks.append({
            **chunk,
            "score": float(scores[int(index)]),
        })
        total_chars += len(chunk_text)

    return selected_chunks

# --- 2. Hybrid OCR Function ---
def extract_text_hybrid(file_bytes):
    """
    Takes PDF file bytes, extracts text using PyMuPDF, and falls back to OCR if needed.
    """
    full_text = ""
    method_log = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
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


# --- 3. The Agentic Reflexion Loop ---
class ReflexionLoop:
    def __init__(self, client, rubric, student_text, truth_context):
        self.client = client
        self.rubric = rubric
        self.student_text = student_text
        self.truth_context = truth_context
        self.logs = [] 

    def _parse_json_response(self, response_text):
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text.strip("`")
            if cleaned_text.startswith("json"):
                cleaned_text = cleaned_text[4:].strip()

        candidate_texts = [cleaned_text]

        start_index = cleaned_text.find("{")
        end_index = cleaned_text.rfind("}")
        if start_index != -1 and end_index != -1 and end_index > start_index:
            candidate_texts.append(cleaned_text[start_index:end_index + 1])

        bracket_start = cleaned_text.find("[")
        bracket_end = cleaned_text.rfind("]")
        if bracket_start != -1 and bracket_end != -1 and bracket_end > bracket_start:
            candidate_texts.append(cleaned_text[bracket_start:bracket_end + 1])

        last_error = None
        for candidate_text in candidate_texts:
            try:
                return json.loads(candidate_text)
            except json.JSONDecodeError as json_error:
                last_error = json_error
                try:
                    return ast.literal_eval(candidate_text)
                except Exception:
                    continue

        raise last_error or ValueError("Unable to parse JSON response from model.")

    def _repair_json_response(self, response_text, role="Agent"):
        repair_prompt = f"""
        Convert the following grading response into valid JSON only.
        Return only a JSON object that matches this schema:
        {{ "evaluation": [ {{ "question_id": "string", "score_awarded": float, "max_score": float, "reasoning_trace": "string", "evidence_quote": "string" }} ], "total_score_awarded": float, "total_max_score": float, "summary_feedback": "string" }}

        Do not add markdown, comments, or explanation.

        <BrokenResponse>
        {response_text}
        </BrokenResponse>
        """
        repair_response = self.client.models.generate_content(
            model="gemini-3.5-flash",
            contents=repair_prompt,
            config={
                **generation_config,
                "response_mime_type": "application/json"
            }
        )
        repair_text = repair_response.text or ""
        self.logs.append({
            "role": f"{role} (Repair)",
            "latency": 0,
            "output_snippet": repair_text[:100] + ("..." if len(repair_text) > 100 else "")
        })
        return self._parse_json_response(repair_text)

    def _call_gemini(self, prompt, role="Agent"):
        start = time.time()
        try:
            response = self.client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt,
                config={
                    **generation_config,
                    "response_mime_type": "application/json"
                }
            )
            duration = round(time.time() - start, 2)
            response_text = response.text or ""
            self.logs.append({
                "role": role,
                "latency": duration,
                "output_snippet": response_text[:100] + ("..." if len(response_text) > 100 else "")
            })
            try:
                parsed_response = self._parse_json_response(response_text)
            except Exception:
                parsed_response = self._repair_json_response(response_text, role=role)
            if isinstance(parsed_response, dict) and parsed_response.get("error"):
                raise ValueError(parsed_response["error"])
            return parsed_response
        except Exception as e:
            self.logs.append({"role": role, "error": str(e)})
            return {"error": str(e), "raw_response_snippet": response_text[:500] if 'response_text' in locals() else ""}

    def _normalize_grade_payload(self, grade_payload):
        if not isinstance(grade_payload, dict):
            raise ValueError("Grading model returned an invalid payload format.")

        if grade_payload.get("error"):
            raise ValueError(grade_payload["error"])

        evaluation = grade_payload.get("evaluation")
        if not isinstance(evaluation, list) or not evaluation:
            raise ValueError("Grading model returned no evaluation details.")

        total_score_awarded = grade_payload.get("total_score_awarded")
        total_max_score = grade_payload.get("total_max_score")

        normalized_scores = []
        normalized_max_scores = []

        for item in evaluation:
            if not isinstance(item, dict):
                continue

            score_value = item.get("score_awarded")
            max_value = item.get("max_score")

            try:
                if score_value is not None:
                    normalized_scores.append(float(score_value))
                if max_value is not None:
                    normalized_max_scores.append(float(max_value))
            except (TypeError, ValueError):
                continue

        if total_score_awarded is None:
            if normalized_scores:
                total_score_awarded = round(sum(normalized_scores), 2)
            else:
                raise ValueError("Grading model did not provide a usable total score.")

        if total_max_score is None and normalized_max_scores:
            total_max_score = round(sum(normalized_max_scores), 2)

        summary_feedback = grade_payload.get("summary_feedback")
        if not summary_feedback:
            feedback_lines = []
            for item in evaluation[:3]:
                if not isinstance(item, dict):
                    continue
                question_id = item.get("question_id", "Unknown question")
                reasoning = item.get("reasoning_trace") or item.get("evidence_quote") or "No reasoning provided."
                feedback_lines.append(f"{question_id}: {reasoning}")

            summary_feedback = "\n".join(feedback_lines) if feedback_lines else "No summary feedback was returned by the model."

        grade_payload["total_score_awarded"] = round(float(total_score_awarded), 2)
        if total_max_score is not None:
            grade_payload["total_max_score"] = round(float(total_max_score), 2)
        grade_payload["summary_feedback"] = summary_feedback

        return grade_payload

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


# --- 4. Background Grading Task ---
def background_grading_task(task_id, answer_files_data, kb_files_data, api_key, rubric):
    """
    Runs in the background, extracts KB text once, then iterates through all student papers.
    """
    try:
        client = genai.Client(api_key=api_key)
        
        # Step 1: Chunk and embed Course Content (Knowledge Base)
        kb_chunks = build_kb_chunks(kb_files_data)

        # Step 2: Loop through each student answer paper
        for i, file_data in enumerate(answer_files_data):
            grading_tasks[task_id]["results"][i]["status"] = "processing"
            
            start_time = time.time()
            try:
                # Extract Student Text
                student_text, extraction_methods = extract_text_hybrid(file_data["content"])
                
                if not student_text.strip():
                    raise ValueError("Failed to extract readable text from PDF.")

                relevant_chunks = select_top_k_chunks(
                    kb_chunks,
                    f"{rubric}\n\n{student_text}"
                )

                if not relevant_chunks:
                    raise ValueError("No relevant knowledge base chunks could be selected.")

                truth_context = ""
                for chunk in relevant_chunks:
                    truth_context += (
                        f"\n--- Source: {chunk['source']} | Chunk: {chunk['id']} | Similarity: {chunk['score']:.4f} ---\n"
                        f"{chunk['text']}\n"
                    )

                # Initialize and run the Reflexion Loop
                agent_system = ReflexionLoop(client, rubric, student_text, truth_context)
                final_json, agent_logs = agent_system.run()

                final_json = agent_system._normalize_grade_payload(final_json)

                if final_json.get("error"):
                    raise ValueError(final_json["error"])

                total_time = round(time.time() - start_time, 2)
                
                # Update global state with success
                grading_tasks[task_id]["results"][i].update({
                    "status": "completed",
                    "score": final_json.get("total_score_awarded", "N/A"),
                    "feedback": final_json.get("summary_feedback", "No feedback provided."),
                    "details": final_json,
                    "metadata": {
                        "processing_time": total_time,
                        "extraction_methods": extraction_methods,
                        "logs": agent_logs,
                        "selected_kb_chunks": [
                            {
                                "id": chunk["id"],
                                "source": chunk["source"],
                                "score": chunk["score"],
                            }
                            for chunk in relevant_chunks
                        ]
                    }
                })

            except Exception as e:
                # Update global state with error for this specific file
                grading_tasks[task_id]["results"][i].update({
                    "status": "error",
                    "score": "N/A",
                    "feedback": str(e),
                    "details": {"error": str(e)},
                    "metadata": {
                        "processing_time": round(time.time() - start_time, 2)
                    }
                })
            
            # Prevent hitting API rate limits (15 Requests Per Minute limit safe buffer)
            time.sleep(4)

    except Exception as e:
        print(f"Critical Task Error: {str(e)}")


# --- 5. API Endpoints ---
@app.route('/api/grade-batch', methods=['POST'])
def grade_batch():
    # 1. Gather text inputs
    api_key = request.form.get('apiKey')
    rubric = request.form.get('rubric')
    
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")

    # 2. Gather file inputs
    uploaded_answer_papers = request.files.getlist('answerPapers')
    uploaded_kb_files = request.files.getlist('knowledgeBase')
    
    if not uploaded_answer_papers or not uploaded_kb_files or not api_key or not rubric:
        return jsonify({"error": "Missing required fields, API key, or files"}), 400

    # 3. Initialize Task ID
    task_id = str(uuid.uuid4())
    grading_tasks[task_id] = {
        "total": len(uploaded_answer_papers),
        "results": [{"filename": f.filename, "status": "pending"} for f in uploaded_answer_papers]
    }

    # 4. Read files into memory (Flask drops files after route returns)
    answer_files_data = [{"filename": f.filename, "content": f.read()} for f in uploaded_answer_papers]
    kb_files_data = [{"filename": f.filename, "content": f.read()} for f in uploaded_kb_files]

    # 5. Spawn background thread
    thread = threading.Thread(
        target=background_grading_task,
        args=(task_id, answer_files_data, kb_files_data, api_key, rubric)
    )
    thread.start()

    # 6. Return 202 Accepted immediately
    return jsonify({"task_id": task_id}), 202


@app.route('/api/status/<task_id>', methods=['GET'])
def get_status(task_id):
    task = grading_tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(task)


@app.route('/')
def serve_index():
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/app.js')
def serve_app_js():
    return send_from_directory(BASE_DIR, 'app.js')


if __name__ == '__main__':
    app.run(debug=True, port=5000)