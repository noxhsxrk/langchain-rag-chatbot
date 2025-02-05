# LangChain RAG Chatbot

A powerful chatbot implementation using LangChain, Perplexity, and OpenAI that supports Retrieval-Augmented Generation (RAG) with multiple data sources and summarization capabilities.

## Features

- **Multi-source Data Integration**: Add data from various sources including:
  - Plain text
  - PDF documents
  - Elasticsearch indices
- **Citation Generation**: Use Perplexity to find and cite relevant sources
- **Data Summarization**: Leverage OpenAI's models to summarize retrieved information
- **RAG Pipeline**: Implement Retrieval-Augmented Generation for context-aware responses

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/langchain-rag-chatbot.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export PERPLEXITY_API_KEY="your-perplexity-key"
```

## Usage

### Adding Data Sources

1. **PDF Files**:

```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("document.pdf")
documents = loader.load()
```

2. **Elasticsearch**:

```python
from langchain.document_loaders import ElasticsearchLoader

loader = ElasticsearchLoader(
    es_url="http://localhost:9200",
    index_name="my-index"
)
documents = loader.load()
```

### Setting Up RAG Pipeline

```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# Set up QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)
```

### Generating Citations with Perplexity

```python
from langchain.llms import Perplexity

perplexity_llm = Perplexity(api_key="your-perplexity-key")
response = perplexity_llm("What are the latest advancements in AI?")
```

### Summarizing Data with OpenAI

```python
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter

# Split documents
text_splitter = CharacterTextSplitter()
docs = text_splitter.split_documents(documents)

# Create summarization chain
chain = load_summarize_chain(OpenAI(), chain_type="map_reduce")
summary = chain.run(docs)
```

## Configuration

Create a `.env` file with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
PERPLEXITY_API_KEY=your_perplexity_api_key
ELASTICSEARCH_URL=https://localhost:53391
ELASTICSEARCH_USERNAME=elastic
ELASTICSEARCH_PASSWORD=xxxxx
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License
