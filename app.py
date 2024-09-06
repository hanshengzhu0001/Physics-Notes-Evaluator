import os
import pytesseract
import spacy
import tempfile
from flask import Flask, request, jsonify, render_template
from PIL import Image
from textstat import flesch_kincaid_grade
from pdf2image import convert_from_path
import language_tool_python
import logging
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load Spacy model
nlp = spacy.load("en_core_web_sm")

# Initialize LanguageTool
tool = language_tool_python.LanguageTool('en-US')

# Load the trained BERT model and tokenizer
model_path = './best_model'
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

def ocr_tesseract(image):
    logging.info("Starting OCR processing")
    text = pytesseract.image_to_string(image)
    logging.info("Completed OCR processing")
    return text

def text_preprocessing(text):
    logging.info("Starting text preprocessing")
    doc = nlp(text)
    cleaned_text = " ".join([token.text for token in doc if not token.is_stop and not token.is_punct])
    logging.info("Completed text preprocessing")
    return cleaned_text

def readability_check(text):
    logging.info("Starting readability check")
    score = flesch_kincaid_grade(text)
    logging.info("Completed readability check")
    return score

def structure_check(text):
    logging.info("Starting structure check")
    doc = nlp(text)
    structure_issues = []
    if not any([token.text for token in doc.sents if token.text.istitle()]):
        structure_issues.append("No headings found.")
    logging.info("Completed structure check")
    return structure_issues

def grammar_and_coherence_check(text, num_pages):
    logging.info("Starting grammar and coherence check")
    matches = tool.check(text)
    filtered_matches = [match for match in matches if match.ruleIssueType != 'misspelling']
    total_issues = len(filtered_matches)
    avg_issues_per_page = total_issues / num_pages if num_pages > 0 else 0
    logging.info("Completed grammar and coherence check")
    return {"average_issues_per_page": avg_issues_per_page, "total_issues": total_issues}

def content_relevance_check(text, subject):
    logging.info("Starting content relevance check")
    relevant_keywords = {
        'mathematics': ['math', 'algebra', 'geometry', 'calculus'],
        'history': ['history', 'historical', 'event', 'era', 'period'],
        'biology': ['biology', 'life', 'cell', 'organism', 'genetics']
    }
    if subject in relevant_keywords:
        if any(keyword in text.lower() for keyword in relevant_keywords[subject]):
            logging.info("Content is relevant")
            return "Content is relevant."
        else:
            logging.info("Content may not be relevant")
            return "Content may not be relevant."
    logging.info("Subject not recognized")
    return "Subject not recognized."

def statement_extraction(text):
    logging.info("Starting statement extraction")
    doc = nlp(text)
    statements = [sent.text for sent in doc.sents]
    logging.info("Completed statement extraction")
    return statements

def check_statement_correctness(statement):
    inputs = tokenizer(statement, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return "correct" if prediction == 1 else "incorrect"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    logging.info("Received file upload request")
    if 'file' not in request.files:
        logging.error("No file part in the request")
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        logging.error("No selected file")
        return jsonify({"error": "No selected file"}), 400

    if file:
        logging.info("Processing uploaded PDF file")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            file.save(temp.name)
            temp_path = temp.name
        
        images = convert_from_path(temp_path, 300)
        num_pages = len(images)
        text = ""
        for image in images:
            text += ocr_tesseract(image) + "\n"
        
        os.remove(temp_path)  # Clean up the temporary file
        
        cleaned_text = text_preprocessing(text)
        readability = readability_check(cleaned_text)
        structure_issues = structure_check(cleaned_text)
        grammar_coherence = grammar_and_coherence_check(cleaned_text, num_pages)
        subject = request.form.get('subject', 'general')
        content_relevance = content_relevance_check(cleaned_text, subject)
        
        # Statement extraction and correctness check
        statements = statement_extraction(cleaned_text)
        statement_correctness = {stmt: check_statement_correctness(stmt) for stmt in statements}
        
        report = {
            "readability": readability,
            "structure_issues": structure_issues,
            "grammar_and_coherence": grammar_coherence,
            "content_relevance": content_relevance,
            "statement_correctness": statement_correctness
        }
        logging.info("Generated report")
        return jsonify(report)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
