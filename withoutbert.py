from flask import Flask, request, jsonify, send_from_directory, session, render_template
from flask_cors import CORS
from flask import send_file
import os
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
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='static', static_url_path='')
app.secret_key = os.getenv('FLASK_SECRET_KEY')
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

# ... (Previous imports, app setup, logging, database init, CSV loading, handle_urgent_symptom, save_user_input, generate_summary, schedule_summaries, etc., remain unchanged)

def query_medical_ai(prompt, max_tokens=767):
    """
    Uses Groq API to process medical queries with optimized prompts, preserving original flexibility.
    Returns: Generated response as string
    """
    system_prompt = (
        "You are Syrid, a qualified medical professional. Provide accurate, compassionate, and clear medical advice, mimicking a doctor's approach. "
        "For symptom-related queries:\n"
        "1. Analyze symptoms, duration, severity, and context (e.g., age, medical history) if provided.\n"
        "2. If input is vague, ask 1-2 targeted, empathetic follow-up questions.\n"
        "3. For clear inputs, suggest 3-5 possible conditions, an urgency level (Low: self-care; Moderate: see doctor within 24-48 hours; High: seek immediate care), and next steps.\n"
        "4. Use empathetic, non-technical language and avoid jargon unless explained.\n"
        "5. Always include: 'I am not a doctor; please consult one for a professional diagnosis.'\n"
        "6. For urgent symptoms (e.g., chest pain, severe bleeding), prioritize immediate medical attention.\n"
        "For non-symptom queries (e.g., educational or general health questions), respond clearly and concisely as a knowledgeable doctor."
    )
    try:
        client = Groq(api_key=os.getenv("KYLE"))
        response = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=max_tokens,
            top_p=0.9
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Groq API Error: {e}")
        return "I apologize, but I couldn't process your medical request at this time. Please try again later."

def classify_symptoms(text, user_context=None):
    """
    Classify symptoms using CSV or knowledge base, with dynamic questioning.
    Returns: condition, medicine, dosage, diet, confidence, follow_up (or None)
    """
    try:
        matched_symptoms = [s for s in known_symptoms if s.lower() in text.lower()]
        if not matched_symptoms:
            follow_up = (
                "I'm sorry you're not feeling well. Could you clarify your symptoms? For example, how long have they lasted, how severe are they, or do you have other symptoms like fever or pain?"
            )
            return "Unknown", "None", "None", "None", 0.0, follow_up
        
        # Optional: Query knowledge base (e.g., Infermedica API)
        use_knowledge_base = os.getenv("USE_KNOWLEDGE_BASE", "false").lower() == "true"
        if use_knowledge_base:
            try:
                kb_response = requests.post(
                    "https://api.infermedica.com/v3/parse",
                    headers={"App-Id": os.getenv("INFERMEDICA_APP_ID"), "App-Key": os.getenv("INFERMEDICA_APP_KEY")},
                    json={"text": text, "context": user_context or {}}
                )
                kb_response.raise_for_status()
                data = kb_response.json()
                conditions = data.get("conditions", [])
                top_condition = conditions[0]["name"] if conditions else "Unknown"
                confidence = conditions[0]["probability"] if conditions else 0.0
                medicine = data.get("recommended_medication", "None")
                dosage = data.get("dosage", "None")
                diet = data.get("diet_recommendation", "None")
            except Exception as e:
                logging.error(f"Knowledge base query error: {e}")
                use_knowledge_base = False  # Fallback to CSV
        
        # Fallback to CSV if knowledge base is disabled or fails
        if not use_knowledge_base:
            top_condition = next(
                (row['Possible_Conditions'] for row in symptoms_data 
                 if matched_symptoms[0].lower() in row['Symptom'].lower()), 
                "Unknown"
            )
            medicine = next(
                (row['Medicine'] for row in medications_data 
                 if top_condition.split(',')[0].lower() in row['Possible_Conditions'].lower()),
                "None"
            )
            dosage = next(
                (row['Dosage_Adult'] for row in medications_data 
                 if row['Medicine'] == medicine),
                "None"
            )
            diet = next(
                (row['diet'] for row in diet_data 
                 if top_condition.split(',')[0].lower() in row['Possible_Conditions'].lower()),
                "None"
            )
            confidence = min(0.9, len(matched_symptoms) * 0.3)
        
        # Check for missing context
        if user_context and not any(k in user_context for k in ['duration', 'severity']):
            follow_up = (
                "To better understand your symptoms, could you share how long you've had them and how severe they are (e.g., mild, moderate, severe)?"
            )
            return top_condition, medicine, dosage, diet, confidence, follow_up
        return top_condition, medicine, dosage, diet, confidence, None
    except Exception as e:
        logging.error(f"Classification error: {e}")
        return "Unknown", "None", "None", "None", 0.0, "Please provide more details about your symptoms."

@app.route('/')
def serve_frontend():
    """
    Serve the frontend index.html file (restored from original code).
    """
    init_db()
    index_path = os.path.join(os.path.dirname(__file__), 'index.html')
    if not os.path.exists(index_path):
        return f"index.html not found at {index_path}", 500
    return send_file(index_path)

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
                prompt = (
                    f"User asked: Please provide monthly insights based on the following health data from {month_start} to {month_end}. "
                    "Provide insights as a calm, knowledgeable doctor. "
                    "Include a disclaimer: 'I am not a doctor; please consult one for a professional diagnosis.'"
                )
                insights = query_medical_ai(prompt)
                return jsonify({
                    'user_id': user_id,
                    'insights': insights,
                    'summary_data': summary_data
                })
        
        else:
            user_input = input_data.get('message', '')
            if not user_input:
                return jsonify({'reply': 'Please enter a valid message.'})
                
            # Handle urgent symptoms (including bleeding context)
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
                    prompt = (
                        f"As a doctor, explain these steps clearly: {reply}\n"
                        "Include a disclaimer: 'I am not a doctor; please consult one for a professional diagnosis.'"
                    )
                    doctor_reply = query_medical_ai(prompt)
                    return jsonify({'reply': doctor_reply})
            
            for symptom, details in urgent_symptoms.items():
                if symptom.lower() in user_input.lower():
                    urgent_reply = handle_urgent_symptom(symptom, user_input)
                    if urgent_reply:
                        return jsonify({'reply': urgent_reply})
                    urgent_reply = details['response']
                    prompt = (
                        f"As a doctor, explain this urgent advice clearly: {urgent_reply}\n"
                        "Include a disclaimer: 'I am not a doctor; please consult one for a professional diagnosis.'"
                    )
                    doctor_reply = query_medical_ai(prompt)
                    return jsonify({'reply': doctor_reply})
            
            # Handle educational queries
            is_educational = any(x in user_input.lower() for x in ["what is", "how does", "explain", "difference between"])
            if is_educational:
                topic = user_input.split()[-1]
                wiki = get_wikipedia_summary(topic)
                prompt = (
                    f"User asked: {user_input}\nWikipedia says:\n{wiki}\n"
                    "Explain clearly like a doctor in 500 words or less. "
                    "Include a disclaimer: 'I am not a doctor; please consult one for a professional diagnosis.'"
                    "For symptom-related queries, ensure the response aligns with medical accuracy."
                )
                reply = query_medical_ai(prompt)
                return jsonify({'reply': reply})
            
            # Process symptoms with context
            user_context = input_data.get('context', {})
            condition, medicine, dosage, diet, confidence, follow_up = classify_symptoms(user_input, user_context)
            
            if follow_up:
                prompt = (
                    f"The user said: {user_input}\n"
                    f"The input lacks critical details. Respond empathetically and ask: {follow_up}\n"
                    "Include a disclaimer: 'I am not a doctor; please consult one for a professional diagnosis.'"
                )
                reply = query_medical_ai(prompt)
                return jsonify({'reply': reply})
            
            prompt = (
                f"Patient reports: {user_input}\n"
                f"Diagnosis: {condition} (confidence: {confidence:.1%})\n"
                f"Medication: {medicine}, Dosage: {dosage}\n"
                f"Diet: {diet}\n"
                "As a doctor, provide a structured response with:\n"
                "- **Possible Causes**: List 3-5 conditions with brief explanations.\n"
                "- **Urgency Level**: Low (self-care), Moderate (see doctor within 24-48 hours), or High (seek immediate care), with reason.\n"
                "- **Next Steps**: Specific actions (e.g., rest, hydrate, doctor visit).\n"
                "- **Disclaimer**: 'I am not a doctor; please consult one for a professional diagnosis.'\n"
                "Use empathetic, clear language in 100-150 words."
            )
            reply = query_medical_ai(prompt)
            reply += "\n\n⚠️ Monitor your symptoms and consult a doctor if they worsen."
            return jsonify({'reply': reply})
    
    except Exception as e:
        logging.error(f"Analyze User Input Error: {e}")
        return jsonify({'error': 'Failed to process request'}), 500

# ... (Rest of the code, including handle_urgent_symptom, save_user_input, generate_summary, schedule_summaries, and app.run, remains unchanged)
if __name__ == '__main__':
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
