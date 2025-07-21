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
from typing import List, Dict

app = Flask(__name__, static_folder='static', static_url_path='')
app.secret_key = 'dky483yh983yh4tpo28py4p98t4y8374ti7256176t851ryo874to81465'
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
    with open(file_path, mode='r') as f:
        return list(csv.DictReader(f))

def validate_csv_columns(data: List[Dict], required_columns: List[str], file_name: str):
    """Validate CSV structure"""
    if not data:
        raise ValueError(f"{file_name} is empty")
    missing = [col for col in required_columns if col not in data[0]]
    if missing:
        raise KeyError(f"Missing columns {missing} in {file_name}")

try:
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
urgent_symptoms = ['bleeding', 'chest pain', 'shortness of breath', 'severe pain']

wiki_wiki = wikipediaapi.Wikipedia(user_agent='MedicalAI/1.0 (your.email@example.com)', language='en')

@lru_cache(maxsize=100)
def get_wikipedia_summary(topic):
    try:
        page = wiki_wiki.page(topic)
        return page.summary[:1000] if page.exists() else "No reliable Wikipedia info found."
    except Exception as e:
        logging.error(f"Wikipedia Error for topic '{topic}': {e}")
        return "Wikipedia lookup failed."

def query_ollama(prompt, max_tokens=150):
    """Enhanced query function with medical context"""
    system_prompt = """You are Dr. Mistral, an AI medical assistant. Provide:
    - Evidence-based advice
    - Clear explanations
    - Conservative recommendations
    - Always suggest professional care for serious symptoms"""
    
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "mistral",
            "prompt": f"{system_prompt}\n\n{prompt}",
            "stream": False,
            "options": {"num_predict": max_tokens}
        })
        return response.json().get("response", "Model did not respond.")
    except Exception as e:
        logging.error(f"Ollama API Error: {e}")
        return "I apologize, but I couldn't process your request. Please try again or consult a healthcare professional."

def classify_symptoms(text):
    """New LLM-based symptom classification"""
    prompt = f"""Analyze these symptoms: {text}
    Return JSON format with:
    - condition (most likely medical condition)
    - medicine (recommended medication)
    - dosage (adult dosage recommendation)
    - diet (dietary recommendations)
    - confidence (0-1 confidence score)
    
    Example:
    {{
        "condition": "migraine",
        "medicine": "ibuprofen",
        "dosage": "200-400mg every 4-6 hours",
        "diet": "stay hydrated, avoid caffeine",
        "confidence": 0.85
    }}"""
    
    try:
        response = query_ollama(prompt, max_tokens=300)
        # Extract JSON from response
        start = response.find('{')
        end = response.rfind('}') + 1
        json_str = response[start:end]
        result = json.loads(json_str)
        
        # Set defaults for missing fields
        return {
            "condition": result.get("condition", "Unknown"),
            "medicine": result.get("medicine", "None"),
            "dosage": result.get("dosage", "None"),
            "diet": result.get("diet", "None"),
            "confidence": float(result.get("confidence", 0.0))
        }
    except Exception as e:
        logging.error(f"Symptom analysis failed: {e}")
        return {
            "condition": "Unknown",
            "medicine": "None",
            "dosage": "None",
            "diet": "None",
            "confidence": 0.0
        }

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
            prompt = f"User asked: Please analyze the following health data for a {summary_type} summary from {period_start} to {period_end}. Provide insights as a calm, knowledgeable doctor."
            insights = query_ollama(json.dumps(summary_data) + prompt, max_tokens=400)
            
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
        request_type = data.get('request_type', 'free_text').strip().lower()
        
        if not user_id or not page_context or not input_data:
            return jsonify({'error': 'Missing required fields'}), 400
            
        save_user_input(user_id, page_context, input_data, timestamp)
        
        if request_type == 'monthly_inspector':
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
                insights = query_ollama(json.dumps(summary_data) + prompt, max_tokens=400)
                
                return jsonify({
                    'user_id': user_id,
                    'insights': insights,
                    'summary_data': summary_data
                })
        
        elif request_type == 'structured_log':
            stats = json.dumps(input_data)
            prompt = f"A user logged their daily health stats:\n{stats}\nPlease analyze and give advice like a calm doctor."
            reply = query_ollama(prompt)
            return jsonify({'reply': reply})
            
        else:  # Default to free_text handling
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
            
            for symptom in urgent_symptoms:
                if symptom.lower() in user_input.lower():
                    urgent_reply = handle_urgent_symptom(symptom, user_input)
                    if urgent_reply:
                        return jsonify({'reply': urgent_reply})
            
            is_educational = any(x in user_input.lower() for x in ["what is", "how does", "explain", "difference between"])
            if is_educational:
                topic = user_input.split()[-1]
                wiki = get_wikipedia_summary(topic)
                prompt = f"User asked: {user_input}\nWikipedia says:\n{wiki}\nExplain clearly like a doctor in 150 words or less:"
                reply = query_ollama(prompt)
                return jsonify({'reply': reply})
            
            symptoms = [s for s in known_symptoms if s.lower() in user_input.lower()]
            if symptoms:
                analysis = classify_symptoms(', '.join(symptoms))
                prompt = (
                    f"Patient reports: {user_input}\n"
                    f"Diagnosis: {analysis['condition']} (confidence: {analysis['confidence']:.1%})\n"
                    f"Medication: {analysis['medicine']}, Dosage: {analysis['dosage']}\n"
                    f"Diet: {analysis['diet']}\n"
                    f"As a doctor, explain this in a clear, empathetic way."
                )
                reply = query_ollama(prompt)
                reply += "\n\n⚠️ Monitor symptoms and consult a doctor if they worsen."
                return jsonify({'reply': reply})
            
            prompt = f"The user said: {user_input}\nRespond clearly and politely as a medical assistant."
            reply = query_ollama(prompt)
            return jsonify({'reply': reply})
    
    except Exception as e:
        logging.error(f"Analyze User Input Error: {e}")
        return jsonify({'error': 'Failed to process request'}), 500

if __name__ == '__main__':
    init_db()
    port = 5000
    ip_addresses = ['127.0.0.1']
    try:
        network_ip = socket.gethostbyname(socket.gethostname())
        if network_ip not in ip_addresses:
            ip_addresses.append(network_ip)
    except socket.gaierror as e:
        logging.error(f"Network IP retrieval failed: {e}")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(('8.8.8.8', 80))
            local_ip = s.getsockname()[0]
            if local_ip not in ip_addresses:
                ip_addresses.append(local_ip)
    except Exception as e:
        logging.error(f"Local IP retrieval failed: {e}")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', port))
    except OSError as e:
        logging.error(f"Port binding failed: {e}")
        exit(1)
    print("Application is running on the following URLs:")
    for ip in ip_addresses:
        print(f"  http://{ip}:{port}/")
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
