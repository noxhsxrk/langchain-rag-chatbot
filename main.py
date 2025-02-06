from flask import Flask, request, jsonify, render_template, session, Response, stream_with_context
import os
import requests
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document
from flask_session import Session
from werkzeug.utils import secure_filename
import PyPDF2
import urllib.request
from elasticsearch import Elasticsearch
import time

app = Flask(__name__)


app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_COOKIE_NAME'] = 'your_session_cookie_name'
app.config['UPLOAD_FOLDER'] = 'uploads'
Session(app)

load_dotenv()
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

ELASTICSEARCH_URL = os.getenv('ELASTICSEARCH_URL')  
ELASTICSEARCH_USERNAME = os.getenv('ELASTICSEARCH_USERNAME')
ELASTICSEARCH_PASSWORD = os.getenv('ELASTICSEARCH_PASSWORD')


es = Elasticsearch(
    ELASTICSEARCH_URL,
    basic_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD),
    verify_certs=False
)

def fetch_from_elasticsearch(index, query):
    try:
        response = es.search(index=index, body=query)
        return response['hits']['hits']
    except Exception as e:
        print("Error fetching data from Elasticsearch:", str(e))
        return []

def fetch_from_perplexity(fund_name):
    query = f"find raw data about {fund_name}"
    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            json={
                "model": "sonar-pro",
                "messages": [
                    {"role": "user", "content": query}
                ],
                "max_tokens": 1000,
                "top_p": 0.9,
                "return_images": False,
                "return_related_questions": False,
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
        citations = data.get("citations", [])
        data = data.get("choices", [{}])[0].get("message", {}).get("content", "No response received")

        return data, citations
    
    except requests.exceptions.RequestException as e:
        print("Error fetching data from Perplexity:", str(e))
        return "", []


def fetch_from_rag():
    documents = []
    raw_texts = []
    
    if 'rag_data' in session:
        for data in session['rag_data']:

            print(f"Data type: {type(data)}, Data content: {data}")

            if not isinstance(data, str):
                data = str(data)

            documents.append(Document(page_content=data))
            raw_texts.append(data)

    return raw_texts

@app.route('/', methods=['GET', 'POST'])
def chat():
    return render_template('./index.html')

@app.route('/update_rag', methods=['POST'])
def update_rag():
    new_rag_data = request.form.get('rag_data')
    if new_rag_data:
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
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ''
                for page in reader.pages:
                    text += page.extract_text()
            
            print("Extracted text from PDF:", text)
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
    session.pop('history', None)
    session.modified = True
    return jsonify({'status': 'success'})

@app.route('/search/fund-information', methods=['POST'])
def search_elasticsearch():
    data = request.get_json()
    index = 'fund-information'
    codes = data.get('fund_codes')

    if not codes:
        return jsonify({'status': 'failure', 'message': 'Fund codes not provided'})

    query = {
        "query": {
            "terms": {
                "code.keyword": codes
            }
        }
    }

    results = fetch_from_elasticsearch(index, query)
    
    if 'rag_data' not in session:
        session['rag_data'] = []
    session['rag_data'].extend(results)
    session.modified = True

    return jsonify({'status': 'success', 'results': results})

@app.route('/fetch_perplexity', methods=['POST'])
def fetch_perplexity_route():
    data = request.get_json()
    fund_name = data.get('fund_name')
    perplexity_data, citations = fetch_from_perplexity(fund_name)
    return jsonify({
        'perplexity_data': perplexity_data,
        'citations': citations
    })

@app.route('/get_summary', methods=['POST'])
def get_summary():
    data = request.get_json()
    citations = data.get('citations')
    model_name = data.get('model', 'gpt-4o-mini')
    rag_data = fetch_from_rag()

    model_mapping = {
        'gpt-4o': 'gpt-4-turbo',
        'gpt-4o-mini': 'gpt-4',
        'gpt-3.5-turbo': 'gpt-3.5-turbo'
    }

    def generate():
        model = ChatOpenAI(
            model_name=model_mapping[model_name],
            openai_api_key=OPENAI_API_KEY,
            streaming=True
        )
        
        summary_prompt = f"""
            I have gathered two sets of information:
            1. From Link citations: {citations}
            2. From my internal knowledge base (RAG): {rag_data}
            Please summarize the information from both sources.
            
            if citations is empty, please use only RAG data to summary.
            if citations is not empty, please use only citations to summary.

            Please response in thai language.
        """
        
        for chunk in model.stream(summary_prompt):
            yield chunk.content
            time.sleep(0.005)

    return Response(stream_with_context(generate()), content_type='text/plain')

if __name__ == "__main__":
    app.run(debug=True)

