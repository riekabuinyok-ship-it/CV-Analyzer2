import os
import json
import io
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from docx import Document
import fitz  # PyMuPDF

# --- Configuration ---
# DeepSeek API Configuration
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
if not DEEPSEEK_API_KEY:
    raise ValueError("No DEEPSEEK_API_KEY set in environment variables")

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

app = Flask(__name__)

# Configure CORS - update with your actual domain in production
CORS(app, resources={r"/analyze_cv": {"origins": "*"}})

# --- File Extraction Functions ---
def extract_text_from_pdf(file_stream):
    """Extracts text from a PDF file stream."""
    try:
        doc = fitz.open(stream=file_stream.read(), filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def extract_text_from_docx(file_stream):
    """Extracts text from a DOCX file stream."""
    try:
        document = Document(file_stream)
        return '\n'.join(p.text for p in document.paragraphs)
    except Exception as e:
        return f"Error reading DOCX: {e}"

def call_deepseek_api(prompt):
    """Calls the DeepSeek API with the given prompt and returns JSON response."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"DeepSeek API request failed: {str(e)}")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise Exception(f"Failed to parse DeepSeek API response: {str(e)}")

# --- Main API Endpoint ---
@app.route('/analyze_cv', methods=['POST'])
def analyze_cv():
    if 'cv_file' not in request.files or 'job_description' not in request.form:
        return jsonify({"error": "Missing CV file or job description"}), 400

    cv_file = request.files['cv_file']
    job_description = request.form['job_description']
    file_extension = cv_file.filename.split('.')[-1].lower()
    
    # Read the file data into a stream for parsing
    file_stream = io.BytesIO(cv_file.read())

    # 1. CV Text Extraction
    if file_extension == 'pdf':
        cv_text = extract_text_from_pdf(file_stream)
    elif file_extension in ['doc', 'docx']:
        cv_text = extract_text_from_docx(file_stream)
    else:
        return jsonify({"error": "Unsupported file type. Please upload a PDF or DOCX."}), 415

    if cv_text.startswith("Error"):
        return jsonify({"error": f"Failed to parse CV: {cv_text}"}), 500

    # 2. DeepSeek Analysis Prompt (Tuned for JSON Output)
    prompt = f"""
    You are an expert HR CV analyst. Compare the Candidate CV with the Job Description (JD).
    
    **CANDIDATE CV TEXT:**
    ---
    {cv_text}
    ---

    **JOB DESCRIPTION (JD):**
    ---
    {job_description}
    ---

    Generate a JSON response with the following four keys ONLY:
    1.  **fit_score_percent**: (Integer 0-100) The numerical match percentage.
    2.  **summary**: (String) A 3-sentence summary of the CV's alignment.
    3.  **issues_to_update**: (List of Strings) 3-5 specific, actionable bullet points the user must update or add to their CV (e.g., "Add quantified results for your experience.").
    4.  **alternative_summary**: (String) A new, best-alternative professional summary (4-6 lines) written for the CV, optimized specifically for this JD.
    
    Output ONLY the valid JSON object. Do not include any other text or markdown formatting.
    """
    
    try:
        # 3. Call the DeepSeek API
        ai_response = call_deepseek_api(prompt)
        
        # Load the JSON from the response
        ai_result = json.loads(ai_response)
        
        return jsonify(ai_result), 200

    except Exception as e:
        print(f"DeepSeek API Error: {e}")
        return jsonify({"error": f"AI analysis failed due to a server error. Details: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({"status": "healthy", "message": "CV Analyzer API is running"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)