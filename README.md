# Automated Bulk Grading System

A Flask web application that grades multiple student answer-paper PDFs using a Gemini model, a user-provided rubric, and uploaded course material. It extracts text from PDFs, falls back to OCR for scanned pages, retrieves relevant course-content chunks, and returns scores with feedback in the browser.

## Requirements

- Python 3.10 or newer
- A Google Gemini API key
- Tesseract OCR installed and available on `PATH` for scanned PDFs
- Internet access on the first run so `sentence-transformers` can download `all-MiniLM-L6-v2`

On Windows, install Tesseract from the official installer and add its installation directory to `PATH`.

## Setup

1. Create and activate a virtual environment:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. Install the dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   The application imports the current `google.genai` client. If that import is unavailable after installing the requirements, install the matching SDK:

   ```powershell
   pip install google-genai
   ```

3. Create a `.env` file in the project root:

   ```env
   GEMINI_API_KEY=your_gemini_api_key
   ```

   The API key can also be entered in the web form when the application is running.

## Run the application

Start the backend from the project directory:

```powershell
python multiagent.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in a browser. The backend serves the API on port `5000`; open `index.html` directly if the backend does not serve the frontend in your local setup.

## Using the grader

1. Enter a Gemini API key, or configure `GEMINI_API_KEY` in `.env`.
2. Upload one or more course-content PDFs.
3. Paste the assignment grading rubric.
4. Upload one or more student answer PDFs.
5. Select **Grade Papers** and wait for the dashboard to finish processing.
6. Select **View Feedback** for a paper's detailed result.

## Optional utilities

- `python check.py` lists Gemini models available to the configured API key.
- `python test_models.py` performs a basic Gemini client/model check.
- `python create_kb.py` creates `knowledge_base.json` from `textbook.pdf`. The current web workflow builds its knowledge base from PDFs uploaded through the UI.

## Project structure

| File | Purpose |
| --- | --- |
| `multiagent.py` | Flask server, PDF/OCR extraction, retrieval, and multi-agent grading workflow |
| `index.html` | Browser interface for uploading files and viewing results |
| `app.js` | Frontend submission, polling, and feedback display logic |
| `check.py` | Lists models available through the Gemini API |
| `test_models.py` | Basic Gemini API connectivity test |
| `create_kb.py` | Builds an embedding-based knowledge base from `textbook.pdf` |
| `knowledge_base.json` | Generated knowledge-base data |
| `requirements.txt` | Python dependencies |

## Notes

- Grading is performed asynchronously in a background thread, and the frontend polls `/api/status/<task_id>` for updates.
- Keep API keys private. Do not commit `.env` or expose keys in shared screenshots or logs.
- Model availability and pricing are controlled by the configured Gemini account and may change over time.
