from flask import Flask, request, jsonify, send_from_directory, session, render_template
from flask_cors import CORS
import csv
import wikipediaapi
import requests
import logging
import os
from logging.handlers import RotatingFileHandler
import socket
from functools import lru_cache
import sqlite3
from datetime import datetime, date, timedelta
import json
from apscheduler.schedulers.background import BackgroundScheduler
import uuid
from typing import List, Dict

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dky483yh983yh4tpo28py4p98t4y8374ti7256176t851ryo874to81465')
CORS(app)

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler = RotatingFileHandler('medical_ai.log', maxBytes=10*1024*1024, backupCount=5)
handler.setFormatter(formatter)
logger.addHandler(handler)

# SQLite database
DB_FILE = 'health_tracker.db'

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS user_inputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                page_context TEXT,
                input_data TEXT,
                timestamp TEXT
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                summary_type TEXT,
                period_start TEXT,
                period_end TEXT,
                summary_data TEXT,
                insights TEXT,
                created_at TEXT
            )
        ''')
        conn.commit()

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.start()

def load_csv(file_path: str) -> List[Dict]:
    """Lightweight CSV loader to replace pandas"""
    with open(file_path, mode='r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def validate_csv_columns(data: List[Dict], required_columns: List[str], file_name: str):
    """Validate CSV structure"""
    if not data:
        raise ValueError(f"{file_name} is empty")
    missing = [col for col in required_columns if col not in data[0]]
    if missing:
        raise KeyError(f"Missing columns {missing} in {file_name}")

try:
    symptoms_data = load_csv('data/symptoms_conditions.csv')
    validate_csv_columns(symptoms_data, ['Symptom', 'Possible_Conditions'], 'symptoms_conditions.csv')
    
    medications_data = load_csv('data/medications.csv')
    validate_csv_columns(medications_data, ['Medicine', 'Possible_Conditions', 'Dosage_Adult'], 'medications.csv')
    
    diet_data = load_csv('data/diet_recommendations.csv')
    validate_csv_columns(diet_data, ['Possible_Conditions', 'diet'], 'diet_recommendations.csv')
except Exception as e:
    logging.error(f"Data loading error: {e}")
    raise

# Process known symptoms
known_symptoms = list({
    symptom.strip() 
    for row in symptoms_data 
    for symptom in row['Symptom'].split(',')
})
urgent_symptoms = {
    'bleeding': {'level': 'emergency', 'response': '🚨 Apply pressure and seek emergency care'},
    'chest pain': {'level': 'emergency', 'response': '🚨 Call emergency services immediately'},
    'shortness of breath': {'level': 'urgent', 'response': 'Seek medical attention within 1 hour'},
    'severe pain': {'level': 'urgent', 'response': 'Consult a doctor within 2 hours'}
}

wiki_wiki = wikipediaapi.Wikipedia(
    user_agent='MedicalAI/1.0 (kbofficial555@gmail.com)',
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

@lru_cache(maxsize=100)
def get_wikipedia_summary(topic):
    """Enhanced Wikipedia lookup with medical context"""
    try:
        page = wiki_wiki.page(topic)
        if not page.exists():
            return "No reliable medical information found on Wikipedia."
        
        # Extract the most relevant medical sections
        summary = page.summary[:1000]
        if 'medical' in page.sections:
            summary += "\n\nMedical Context:\n" + page.sections['medical'].text[:500]
        return summary
    except Exception as e:
        logging.error(f"Wikipedia Error: {e}")
        return "Medical information lookup failed."

def query_ollama(prompt: str, max_tokens: int = 300) -> str:
    """Enhanced medical query with safety protocols"""
    system_prompt = """You are Dr. Syrid, an AI medical consultant. Follow these rules:
    1. EVIDENCE-BASED: Cite latest guidelines (WHO/NIH) when possible
    2. SAFETY FIRST: For these symptoms [chest pain, bleeding, suicidal thoughts], respond:
       "🚨 STOP - This requires immediate medical attention. Call emergency services."
    3. STRUCTURE:
       - Possible Causes (max 3)
       - Recommended Actions (prioritized)
       - When to Seek Help (specific criteria)
    4. DISCLAIMER: Always include:
       "NOTE: This is not medical diagnosis. Consult your healthcare provider.""""
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": f"{system_prompt}\n\nUSER QUERY: {prompt}",
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7  # More conservative responses
                }
            },
            timeout=10  # Fail fast if Ollama isn't running
        )
        response.raise_for_status()
        return response.json().get("response", "").split("NOTE:")[0] + """
NOTE: This is not medical diagnosis. Consult your healthcare provider."""
    except requests.exceptions.RequestException as e:
        logging.error(f"Ollama API Error: {e}")
        return """I cannot provide medical advice right now. 
For urgent concerns, contact your local emergency services."""

def classify_symptoms(text: str) -> Dict:
    """Enhanced symptom analysis with safety checks"""
    # First check for urgent symptoms
    for symptom, details in urgent_symptoms.items():
        if symptom in text.lower():
            return {
                "condition": "POTENTIALLY SERIOUS - " + symptom.upper(),
                "action": details['response'],
                "confidence": 0.99,
                "urgency": details['level']
            }
    
    # Normal classification
    prompt = f"""Analyze these symptoms: {text}
    Return JSON with:
    - condition (most likely, max 2 possibilities)
    - action (conservative recommendations)
    - confidence (0-1)
    - urgency (routine/urgent/emergency)"""
    
    try:
        response = query_ollama(prompt)
        start = response.find('{')
        end = response.rfind('}') + 1
        return json.loads(response[start:end])
    except Exception as e:
        logging.error(f"Symptom analysis failed: {e}")
        return {
            "condition": "Unknown",
            "action": "Consult your primary care provider",
            "confidence": 0.0,
            "urgency": "routine"
        }

# ... [Previous database and utility functions remain unchanged] ...

@app.route('/')
def index():
    """Serve the main interface"""
    init_db()
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Enhanced medical analysis endpoint"""
    try:
        data = request.get_json()
        user_input = data.get('message', '').strip()
        
        if not user_input:
            return jsonify({'error': 'Empty query'}), 400
        
        # Urgent symptom check
        for symptom, details in urgent_symptoms.items():
            if symptom in user_input.lower():
                return jsonify({
                    'response': details['response'],
                    'urgency': details['level'],
                    'sources': ['WHO Emergency Protocol']
                })
        
        # Educational queries
        if any(q in user_input.lower() for q in ["what is", "how does", "explain"]):
            wiki_summary = get_wikipedia_summary(user_input.split()[-1])
            return jsonify({
                'response': query_ollama(f"Explain this medically: {user_input}\nContext: {wiki_summary}"),
                'sources': ['Wikipedia', 'Medical textbooks']
            })
        
        # Symptom analysis
        analysis = classify_symptoms(user_input)
        return jsonify({
            'response': f"""Based on your symptoms "{user_input}":
            
Possible Condition: {analysis['condition']}
Recommended Action: {analysis['action']}
Confidence Level: {analysis['confidence']*100:.0f}%
Urgency: {analysis['urgency'].upper()}""",
            'sources': ['CDC Guidelines', 'UpToDate Medical References']
        })
        
    except Exception as e:
        logging.error(f"Analysis error: {e}")
        return jsonify({
            'response': """Our medical consultation service is temporarily unavailable.
For urgent concerns, please contact:
- Emergency: 911 (US) / 112 (EU)
- Poison Control: 1-800-222-1222""",
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Create required directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Initialize and run
    init_db()
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
