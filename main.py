from flask import Flask, request, jsonify, render_template, session
import os
import requests
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from flask_session import Session
from werkzeug.utils import secure_filename
import PyPDF2
import urllib.request

app = Flask(__name__)

app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_COOKIE_NAME'] = 'your_session_cookie_name'
Session(app)


load_dotenv()
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Step 1: Query Perplexity API
def fetch_from_perplexity(fund_name):
    query = f"Find the latest news about {fund_name} mutual fund."
    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            json={
                "model": "sonar",
                "messages": [
                    {"role": "system", "content": "Be precise and concise."},
                    {"role": "user", "content": query}
                ],
                "max_tokens": 500,
                "temperature": 0.2,
                "top_p": 0.9,
                "search_domain_filter": ["news"],
                "return_images": False,
                "return_related_questions": False,
                "search_recency_filter": "week",  # Set to 'week' for recent news
                "top_k": 0,
                "stream": False,
                "presence_penalty": 0,
                "frequency_penalty": 1,
                "response_format": None
            },
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            },
        )
        
        response.raise_for_status()
        data = response.json()

        # Extract the response text
        return data.get("choices", [{}])[0].get("message", {}).get("content", "No response received")
    
    except requests.exceptions.RequestException as e:
        print("Error fetching data from Perplexity:", str(e))
        return ""


# Step 2: Retrieve Relevant Data from RAG
def fetch_from_rag(fund_name):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    documents = []
    raw_texts = []  # List to store raw text data
    
    if 'rag_data' in session:
        for data in session['rag_data']:
            documents.append(Document(page_content=data))
            raw_texts.append(data)  # Collect raw text data

    return raw_texts


# Step 3: Merge and Summarize Data
def summarize_data(perplexity_data, rag_data, fund_name):
    model = ChatOpenAI(model_name="gpt-4-turbo", openai_api_key=OPENAI_API_KEY)
    
    summary_prompt = f"""
    I have gathered two sets of information about the mutual fund '{fund_name}':
    1. From Perplexity AI (News): {perplexity_data}
    2. From my internal knowledge base (RAG): {rag_data}
    
    Please merge the information into a concise summary focusing on the key points from the news and the characteristics of the mutual fund.
    Please response in thai language.
    """
    
    return model.predict(summary_prompt)


@app.route('/', methods=['GET', 'POST'])
def chat():
    if 'history' not in session:
        session['history'] = []

    if request.method == 'POST':
        fund_name = request.form.get('fund_name')
        perplexity_data = fetch_from_perplexity(fund_name)
        rag_data = fetch_from_rag(fund_name)
        summary = summarize_data(perplexity_data, rag_data, fund_name)
        
        # Append the new interaction to the session history
        session['history'].append({
            'fund_name': fund_name,
            'perplexity_data': perplexity_data,
            'rag_data': rag_data,
            'summary': summary
        })
        session.modified = True
        
        return jsonify({
            'perplexity_data': perplexity_data,
            'summary': summary,
            'rag_data': rag_data,
            'history': session['history']
        })
    
    return render_template('./index.html', history=session.get('history', []))

@app.route('/update_rag', methods=['POST'])
def update_rag():
    new_rag_data = request.form.get('rag_data')
    if new_rag_data:
        # Store new RAG data in session
        if 'rag_data' not in session:
            session['rag_data'] = []
        session['rag_data'].append(new_rag_data)
        session.modified = True
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'failure', 'message': 'No RAG data provided'})

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'status': 'failure', 'message': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'failure', 'message': 'No selected file'})
    if file:
        # Ensure the upload directory exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Read PDF and extract text
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ''
                for page in reader.pages:
                    text += page.extract_text()
            
            # Optionally, update RAG data with extracted text
            print("Extracted text from PDF:", text)  # Replace with actual update logic
            return jsonify({'status': 'success', 'text': text})
        except Exception as e:
            print("Error processing PDF:", str(e))
            return jsonify({'status': 'failure', 'message': 'Error processing PDF'})

@app.route('/fetch_url', methods=['POST'])
def fetch_url():
    url = request.form.get('url')
    try:
        response = urllib.request.urlopen(url)
        webContent = response.read().decode('utf-8')
        # Update RAG data with web content
        return jsonify({'status': 'success', 'content': webContent})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/delete_rag_node/<int:index>', methods=['DELETE'])
def delete_rag_node(index):
    if 'rag_data' in session and 0 <= index < len(session['rag_data']):
        session['rag_data'].pop(index)
        session.modified = True
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'failure', 'message': 'Invalid index or no RAG data found'})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session.pop('history', None)  # Remove the 'history' key from the session
    session.modified = True
    return jsonify({'status': 'success'})

if __name__ == "__main__":
    app.run(debug=True)

