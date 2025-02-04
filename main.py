from flask import Flask, request, jsonify, render_template, session
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

# Step 1: Query Perplexity API
def fetch_from_perplexity(fund_name):
    query = f"""Find latest information about mutual funds mentioned in this text: {fund_name}
    If there are multiple funds, provide separate analysis for each fund.
    Focus only on official fund information and recent performance.
    Please response in thai language."""
    
    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            json={
                "model": "sonar-pro",
                "messages": [
                    {"role": "system", "content": "You are a financial analyst. Extract fund codes from user input and provide analysis for each fund separately. Focus on official and verified sources."},
                    {"role": "user", "content": query}
                ],
                "max_tokens": 1000,
                "temperature": 0.2,
                "top_p": 0.9,
                "search_domain_filter": ["news", "finance"],
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


# Step 2: Retrieve Relevant Data from RAG
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


# Step 3: Merge and Summarize Data
def summarize_data(citations, rag_data, fund_name, rag_mode):
    model = ChatOpenAI(model_name="gpt-4-turbo", openai_api_key=OPENAI_API_KEY)
    
    if rag_mode:
        summary_prompt = f"""
        I have gathered two sets of information about the mutual fund '{fund_name}':
        1. From Link citations: {citations}
        2. From my internal knowledge base (RAG): {rag_data}
        
        Please compare the funds using the provided citations and RAG data then tell me the pros and cons of each fund and give me the best fund to invest in.
        And provide the citations for each fund.
        Please response in thai language.
        
        for example:
        Fund 1: // pros and cons
        [citation1, citation2, citation3]
        Fund 2: // pros and cons
        [citation4, citation5, citation6]
        """
    else:
        summary_prompt = f"""
        I have gathered a set of information about the mutual fund '{fund_name}':
        1. From Link citations: {citations}
        
        Please compare the funds using the provided citations and tell me the pros and cons of each fund and give me the best fund to invest in.
        And provide the citations for each fund.
        Please response in thai language.
        
        for example:
        Fund 1: // pros and cons
        [citation1, citation2, citation3]
        Fund 2: // pros and cons
        [citation4, citation5, citation6]
        """
    
    return model.predict(summary_prompt)


@app.route('/', methods=['GET', 'POST'])
def chat():
    if 'history' not in session:
        session['history'] = []

    if request.method == 'POST':
        fund_name = request.form.get('fund_name')
        perplexity_data, citations = fetch_from_perplexity(fund_name)
        rag_data = fetch_from_rag()
        summary = summarize_data(citations, rag_data, fund_name, request.form.get('rag_mode') == 'true')
        
        session['history'].append({
            'fund_name': fund_name,
            'perplexity_data': perplexity_data,
            'rag_data': rag_data,
            'citations': citations,
            'summary': summary
        })
        session.modified = True
        
        return jsonify({
            'perplexity_data': perplexity_data,
            'summary': summary,
            'rag_data': rag_data,
            'citations': citations,
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

if __name__ == "__main__":
    app.run(debug=True)

