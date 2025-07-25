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
import re

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
        c.execute('''
            CREATE TABLE IF NOT EXISTS conversation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                role TEXT,        -- "user" or "assistant"
                message TEXT,
                timestamp TEXT
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

def clean_response(ai_response):
    """Removes internal <think>...</think> blocks from AI output."""
    cleaned = re.sub(r"<think>.*?</think>", "", ai_response, flags=re.DOTALL)
    return cleaned.strip()

def query_medical_ai(prompt, max_tokens=767):
    """
    Uses Groq API to process medical queries with optimized prompts.
    Returns: Generated response as string
    """
    system_prompt = (
        "You are Syrid, a qualified medical professional. Provide accurate, compassionate, and clear medical advice in a structured format, mimicking a doctor's approach. "
        "Follow these guidelines:\n"
        "1. For symptom-related queries, analyze provided symptoms and context. If input is vague, respond with 1-2 empathetic, targeted follow-up questions (e.g., 'How long have you had the symptoms?' or 'How severe are they?').\n"
        "2. For clear symptom inputs, provide a structured response with:\n"
        "   - **Possible Causes**: List 3-5 conditions with brief explanations.\n"
        "   - **Urgency Level**: Low (self-care), Moderate (see doctor within 24-48 hours), or High (seek immediate care), with a clear reason.\n"
        "   - **Next Steps**: Specific, actionable advice (e.g., rest, hydrate, doctor visit).\n"
        "   - **Disclaimer**: 'I am not a doctor; please consult one for a professional diagnosis.'\n"
        "3. For urgent symptoms (e.g., chest pain, bleeding), prioritize immediate medical attention with clear instructions.\n"
        "4. For educational queries (e.g., 'what is diabetes'), provide a clear, concise explanation in 500 words or less using reliable information.\n"
        "5. For summary requests (e.g., weekly/monthly insights), analyze health data and provide actionable insights in 500 words or less.\n"
        "6. Use empathetic, non-technical language and avoid medical jargon unless explained.\n"
        "7. Do not include internal reasoning, thinking steps, or debugging information in the response.\n"
        "8. For general or unclear queries, respond politely as a medical assistant, asking clarifying questions if needed.\n"
        "Return only the final response, formatted as specified, with no additional tags or comments."
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

def save_message(user_id, role, message, timestamp):
    """Save a message to the conversation history"""
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO conversation_logs (user_id, role, message, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (user_id, role, message, timestamp))
        conn.commit()

def get_recent_conversation(user_id, limit=6):
    """Get recent conversation history for context"""
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('''
            SELECT role, message FROM conversation_logs
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (user_id, limit))
        rows = c.fetchall()[::-1]  # reverse to show oldest first
        return [(row['role'], row['message']) for row in rows]

def build_conversation_prompt(user_id, user_input):
    """Build the conversation context for the AI prompt"""
    history = get_recent_conversation(user_id)
    formatted_history = ""
    
    for role, msg in history:
        speaker = "User" if role == "user" else "Syrid"
        formatted_history += f"{speaker}: {msg}\n"
    
    # Add the new user input and prepare for AI response
    formatted_history += f"User: {user_input}\nSyrid:"
    return formatted_history

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
            insights_raw = query_medical_ai(json.dumps(summary_data) + prompt)
            insights = clean_response(insights_raw)
            
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
        user_input = input_data.get('message', '')
        
        if not user_id or not page_context or not input_data:
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Save the user message to conversation history
        save_message(user_id, "user", user_input, timestamp)
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
                    f"Analyze the following health data from {month_start} to {month_end} and provide {summary_type} insights as a calm, knowledgeable doctor. "
                    "Focus on patterns, frequency, and actionable advice in 500 words or less. "
                    "Include a disclaimer: 'I am not a doctor; please consult one for a professional diagnosis.'\n"
                    f"Data: {json.dumps(summary_data)}"
                )
                insights_raw = query_medical_ai(prompt)
                insights = clean_response(insights_raw)
                
                return jsonify({
                    'user_id': user_id,
                    'insights': insights,
                    'summary_data': summary_data
                })
        
        else:
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
                    prompt = (
                        f"Provide clear, empathetic instructions based on this advice: {reply}\n"
                        "Include a disclaimer: 'I am not a doctor; please consult one for a professional diagnosis.'\n"
                        "Return only the final response, formatted as specified, with no additional tags or comments."
                    )
                    doctor_reply_raw = query_medical_ai(prompt)
                    doctor_reply = clean_response(doctor_reply_raw)
                    save_message(user_id, "assistant", doctor_reply, datetime.now().isoformat())
                    return jsonify({'reply': doctor_reply})
            
            for symptom, details in urgent_symptoms.items():
                if symptom.lower() in user_input.lower():
                    urgent_reply = handle_urgent_symptom(symptom, user_input)
                    if urgent_reply:
                        return jsonify({'reply': urgent_reply})
                    prompt = (
                        f"Provide clear, empathetic instructions based on this urgent advice: {details['response']}\n"
                        "Include a disclaimer: 'I am not a doctor; please consult one for a professional diagnosis.'\n"
                        "Return only the final response, formatted as specified, with no additional tags or comments."
                    )
                    doctor_reply_raw = query_medical_ai(prompt)
                    doctor_reply = clean_response(doctor_reply_raw)
                    save_message(user_id, "assistant", doctor_reply, datetime.now().isoformat())
                    return jsonify({'reply': doctor_reply})
            
            is_educational = any(x in user_input.lower() for x in ["what is", "how does", "explain", "difference between"])
            if is_educational:
                topic = user_input.split()[-1]
                wiki = get_wikipedia_summary(topic)
                prompt = (
                    f"User asked: {user_input}\n"
                    f"Wikipedia information: {wiki}\n"
                    "Provide a clear, concise explanation as a doctor in 500 words or less, using empathetic, non-technical language. "
                    "Ensure medical accuracy and include a disclaimer: 'I am not a doctor; please consult one for a professional diagnosis.'\n"
                    "Return only the final response, formatted as specified, with no additional tags or comments."
                )
                reply_raw = query_medical_ai(prompt)
                reply = clean_response(reply_raw)
                save_message(user_id, "assistant", reply, datetime.now().isoformat())
                return jsonify({'reply': reply})
            
            symptoms = [s for s in known_symptoms if s.lower() in user_input.lower()]
            if symptoms:
                condition, medicine, dosage, diet, confidence = classify_symptoms(', '.join(symptoms))
                if condition == "Unknown":
                    prompt = build_conversation_prompt(user_id, user_input) + (
                        "\nThe symptoms are unclear. Respond empathetically with 1-2 targeted questions to clarify (e.g., duration, severity, additional symptoms). "
                        "Include a disclaimer: 'I am not a doctor; please consult one for a professional diagnosis.'\n"
                        "Return only the final response, formatted as specified, with no additional tags or comments."
                    )
                    reply_raw = query_medical_ai(prompt)
                    reply = clean_response(reply_raw)
                    save_message(user_id, "assistant", reply, datetime.now().isoformat())
                    return jsonify({'reply': reply})
                
                prompt = build_conversation_prompt(user_id, user_input) + (
                    f"\nPatient reports these symptoms. Diagnosis: {condition} (confidence: {confidence:.1%})\n"
                    f"Medication: {medicine}, Dosage: {dosage}\n"
                    f"Diet: {diet}\n"
                    "Provide a structured response with:\n"
                    "- Possible Causes: List 3-5 conditions with brief explanations.\n"
                    "- Urgency Level: Low (self-care), Moderate (see doctor within 24-48 hours), or High (seek immediate care), with a clear reason.\n"
                    "- Next Steps: Specific, actionable advice (e.g., rest, hydrate, doctor visit).\n"
                    "- Disclaimer: 'I am not a doctor; please consult one for a professional diagnosis.'\n"
                    "Use empathetic, clear language in 100-150 words. Return only the final response, formatted as specified, with no additional tags or comments."
                )
                reply_raw = query_medical_ai(prompt)
                reply = clean_response(reply_raw)
                reply += "\n\n⚠️ Monitor your symptoms and consult a doctor if they worsen."
                save_message(user_id, "assistant", reply, datetime.now().isoformat())
                return jsonify({'reply': reply})
            
            prompt = build_conversation_prompt(user_id, user_input) + (
                "\nRespond clearly and politely as a medical assistant. If the input is unclear, ask 1-2 targeted, empathetic questions to clarify (e.g., 'Can you describe your symptoms?' or 'How long have you felt this way?'). "
                "Include a disclaimer: 'I am not a doctor; please consult one for a professional diagnosis.'\n"
                "Return only the final response, formatted as specified, with no additional tags or comments."
            )
            reply_raw = query_medical_ai(prompt)
            reply = clean_response(reply_raw)
            save_message(user_id, "assistant", reply, datetime.now().isoformat())
            return jsonify({'reply': reply})
    
    except Exception as e:
        logging.error(f"Analyze User Input Error: {e}")
        return jsonify({'error': 'Failed to process request'}), 500

if __name__ == '__main__':
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
