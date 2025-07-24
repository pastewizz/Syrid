from flask import Flask, request, jsonify, send_from_directory, session
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

app = Flask(__name__, static_folder='static', static_url_path='')
app.secret_key = 'den38uy9385tyifuh745y3uht93fh9uh022uchgunhrungt9245uyt94hf8937yt5087y249uthycvyt29yt92m0nty'
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

def load_csv(file_path):
    """Load CSV file without pandas"""
    data = []
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def validate_csv_columns(data, required_columns, file_name):
    """Validate CSV structure without pandas"""
    if not data:
        raise ValueError(f"{file_name} is empty")
    missing = [col for col in required_columns if col not in data[0]]
    if missing:
        raise KeyError(f"Missing columns {missing} in {file_name}")

try:
    # Load CSV files with pure Python
    symptoms_data = load_csv('symptoms_conditions.csv')
    validate_csv_columns(symptoms_data, ['Symptom', 'Possible_Conditions'], 'symptoms_conditions.csv')
    
    medications_data = load_csv('medications.csv')
    validate_csv_columns(medications_data, ['Medicine', 'Possible_Conditions', 'Dosage_Adult'], 'medications.csv')
    
    diet_data = load_csv('diet_recommendations.csv')
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
    language='en'
)

@lru_cache(maxsize=100)
def get_wikipedia_summary(topic):
    try:
        page = wiki_wiki.page(topic)
        return page.summary[:1000] if page.exists() else "No reliable Wikipedia info found."
    except Exception as e:
        logging.error(f"Wikipedia Error for topic '{topic}': {e}")
        return "Wikipedia lookup failed."

def query_ollama(prompt, max_tokens=150):
    doctor_prompt = "You are Syrid, a qualified medical professional. Analyze the following user health data and speak like a calm, knowledgeable doctor. "
    prompt = doctor_prompt + prompt
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens}
        })
        return response.json().get("response", "Model did not respond.")
    except Exception as e:
        logging.error(f"Ollama API Error: {e}")
        return "I apologize, but I couldn't process your request. Please try again or consult a healthcare professional."

def classify_symptoms(text):
    """Simplified symptom classification without pandas"""
    try:
        # Find matching symptoms
        matched_symptoms = [
            s for s in known_symptoms 
            if s.lower() in text.lower()
        ]
        
        if not matched_symptoms:
            return "Unknown", "None", "None", "None", 0.0
            
        # Get first matching symptom's conditions
        condition = next(
            row['Possible_Conditions'] 
            for row in symptoms_data 
            if matched_symptoms[0].lower() in row['Symptom'].lower()
        )
        
        # Find related medication
        medicine = next(
            (row['Medicine'] for row in medications_data 
            if condition.split(',')[0].lower() in row['Possible_Conditions'].lower()
        ), "None")
        
        # Get dosage
        dosage = next(
            (row['Dosage_Adult'] for row in medications_data 
            if row['Medicine'] == medicine
        ), "None")
        
        # Get diet recommendation
        diet = next(
            (row['diet'] for row in diet_data 
            if condition.split(',')[0].lower() in row['Possible_Conditions'].lower()
        ), "None")
        
        confidence = min(0.9, len(matched_symptoms) * 0.3)  # Simple confidence score
        
        return condition, medicine, dosage, diet, confidence
        
    except Exception as e:
        logging.error(f"Classification error: {e}")
        return "Unknown", "None", "None", "None", 0.0

def handle_urgent_symptom(symptom, user_input):
    if 'bleeding' in symptom.lower():
        session['context'] = {'state': 'awaiting_bleeding_details', 'symptom': symptom}
        return "I'm here to help. Is the bleeding heavy or light? Where is it occurring (e.g., nose, cut, internal)? Please provide more details."
    return None

def save_user_input(user_id, page_context, input_data, timestamp):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO user_inputs (user_id, page_context, input_data, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (user_id, page_context, json.dumps(input_data), timestamp))
        conn.commit()

def generate_summary(user_id, summary_type, period_start, period_end):
    try:
        with sqlite3.connect(DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute('''
                SELECT * FROM user_inputs 
                WHERE user_id = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            ''', (user_id, period_start, period_end))
            inputs = [dict(row) for row in c.fetchall()]
            
            summary_data = {'user_inputs': inputs}
            prompt = f"User asked: Please provide {summary_type} insights based on the following health data from {period_start} to {period_end}. Provide insights as a calm, knowledgeable doctor."
            insights = query_ollama(json.dumps(summary_data) + prompt)
            
            c.execute('''
                INSERT INTO summaries (user_id, summary_type, period_start, period_end, summary_data, insights, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                summary_type,
                period_start,
                period_end,
                json.dumps(summary_data),
                insights,
                datetime.now().isoformat()
            ))
            conn.commit()
            
            return insights
    except Exception as e:
        logging.error(f"Summary Generation Error: {e}")
        return "Unable to generate summary at this time. Please try again later."

def schedule_summaries():
    try:
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            c.execute('SELECT DISTINCT user_id FROM user_inputs')
            user_ids = [row[0] for row in c.fetchall()]
            
            today = datetime.now().date()
            for user_id in user_ids:
                week_start = today - timedelta(days=7)
                generate_summary(user_id, 'weekly', week_start.isoformat(), today.isoformat())
                month_start = today - timedelta(days=30)
                generate_summary(user_id, 'monthly', month_start.isoformat(), today.isoformat())
                year_start = today - timedelta(days=365)
                generate_summary(user_id, 'yearly', year_start.isoformat(), today.isoformat())
    except Exception as e:
        logging.error(f"Scheduled Summary Error: {e}")

scheduler.add_job(schedule_summaries, 'interval', days=1)

@app.route('/')
def serve_frontend():
    init_db()
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/analyze-user-input', methods=['POST'])
def handle_analyze_user_input():
    try:
        data = request.get_json()
        user_id = data.get('user_id', str(uuid.uuid4()))
        page_context = data.get('page_context', '').strip()
        input_data = data.get('input_data', {})
        timestamp = data.get('timestamp', datetime.now().isoformat())
        request_type = data.get('request_type', '').strip()
        
        if not user_id or not page_context or not input_data:
            return jsonify({'error': 'Missing required fields'}), 400
            
        save_user_input(user_id, page_context, input_data, timestamp)
        
        if request_type.lower() == 'monthly_inspector':
            month_end = datetime.now().date()
            month_start = month_end - timedelta(days=30)
            with sqlite3.connect(DB_FILE) as conn:
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                c.execute('''
                    SELECT * FROM user_inputs 
                    WHERE user_id = ? AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                ''', (user_id, month_start.isoformat(), month_end.isoformat()))
                inputs = [dict(row) for row in c.fetchall()]
                
                summary_data = {'user_inputs': inputs}
                prompt = f"User asked: Please provide monthly insights based on the following health data from {month_start} to {month_end}. Provide insights as a calm, knowledgeable doctor."
                insights = query_ollama(json.dumps(summary_data) + prompt)
                
                return jsonify({
                    'user_id': user_id,
                    'insights': insights,
                    'summary_data': summary_data
                })
        
        else:
            user_input = input_data.get('message', '')
            if not user_input:
                return jsonify({'reply': 'Please enter a valid message.'})
                
            if 'context' in session:
                context = session['context']
                if context['state'] == 'awaiting_bleeding_details':
                    if 'heavy' in user_input.lower():
                        reply = "Apply firm pressure to the wound with a clean cloth for 10 minutes. Elevate the area if possible. Seek medical attention immediately."
                    elif 'light' in user_input.lower():
                        reply = "Clean the wound with water, apply an adhesive bandage, and monitor for infection. See a doctor if it persists."
                    else:
                        reply = "Please clarify if the bleeding is heavy or light, and where it's occurring."
                    session.pop('context', None)
                    prompt = f"As a doctor, explain these steps clearly: {reply}"
                    doctor_reply = query_ollama(prompt)
                    return jsonify({'reply': doctor_reply})
            
            for symptom, details in urgent_symptoms.items():
                if symptom.lower() in user_input.lower():
                    urgent_reply = details['response']
                    return jsonify({'reply': urgent_reply})
            
            is_educational = any(x in user_input.lower() for x in ["what is", "how does", "explain", "difference between"])
            if is_educational:
                topic = user_input.split()[-1]
                wiki = get_wikipedia_summary(topic)
                prompt = f"User asked: {user_input}\nWikipedia says:\n{wiki}\nExplain clearly like a doctor in 500 words or less:"
                reply = query_ollama(prompt)
                return jsonify({'reply': reply})
            
            symptoms = [s for s in known_symptoms if s.lower() in user_input.lower()]
            if symptoms:
                condition, medicine, dosage, diet, confidence = classify_symptoms(', '.join(symptoms))
                prompt = (
                    f"Patient reports: {user_input}\n"
                    f"Diagnosis: {condition} (confidence: {confidence:.1%})\n"
                    f"Medication: {medicine}, Dosage: {dosage}\n"
                    f"Diet: {diet}\n"
                    f"As a doctor, explain the diagnosis, recommended treatment, and diet in a clear, empathetic way. Provide actionable advice."
                )
                reply = query_ollama(prompt)
                reply += "\n\n⚠️ Monitor your symptoms and consult a doctor if they worsen."
                return jsonify({'reply': reply})
            
            prompt = f"The user said: {user_input}\nRespond clearly and politely as a medical assistant."
            reply = query_ollama(prompt)
            return jsonify({'reply': reply})
    
    except Exception as e:
        logging.error(f"Analyze User Input Error: {e}")
        return jsonify({'error': 'Failed to process request'}), 500
if __name__ == '__main__':
    init_db()
    port = int(os.environ.get("PORT", 5000))  # Use Render's PORT or fallback to 5000 locally
    
    try:
        network_ip = socket.gethostbyname(socket.gethostname())
        if network_ip not in ip_addresses:
            ip_addresses.append(network_ip)
    except socket.gaierror as e:
        logging.error(f"Network IP retrieval failed: {e}")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(('8.8.8.8', 80))  # Google DNS
            local_ip = s.getsockname()[0]
            if local_ip not in ip_addresses:
                ip_addresses.append(local_ip)
    except Exception as e:
        logging.error(f"Local IP retrieval failed: {e}")
    print("Application is running on the following URLs:")
for ip in ip_addresses:
    print(f"  http://{ip}:{port}/")

     app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
