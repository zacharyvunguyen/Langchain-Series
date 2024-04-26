
# Vector Database with LangChain
![img.png](imgs%2Fimg.png)
This project demonstrates how to ingest data from various sources, break it into manageable chunks, and use different vector databases to retrieve specific information. The code includes examples of document loading, chunking, and vector embedding with the LangChain library.

## Data Ingestion

We use multiple data loaders to ingest documents from different formats and sources:

### Text Document Loader
```python
from langchain_community.document_loaders import TextLoader

# Load a plain text document
loader = TextLoader("speech.txt")
text_document = loader.load()
```
This code snippet loads a text document (`speech.txt`) into memory.

### Web-based Loader
```python
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Load web-based content
from langchain_community.document_loaders import WebBaseLoader
import bs4

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-title", "post-content", "post-header")))
)

text_documents = loader.load()
```
Here, we load content from a specified URL, focusing on specific HTML elements (`post-title`, `post-content`, `post-header`) using BeautifulSoup's `SoupStrainer`.

### PDF Loader
```python
from langchain_community.document_loaders import PyPDFLoader

# Load a PDF file
loader = PyPDFLoader("attention.pdf")
docs = loader.load()
```
This code snippet loads a PDF document (`attention.pdf`).

## Chunking Documents

After loading documents, we need to break them into smaller chunks for easier processing and embedding.
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Split documents into chunks of 1000 characters with 200-character overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)
```
This example uses the `RecursiveCharacterTextSplitter` to split the loaded documents into smaller chunks, allowing for efficient vectorization.

## Vector Embedding and Vector Store

With the documents split into smaller chunks, we can create vector embeddings and store them in vector databases. This allows for fast similarity searches.

### Chroma Vector Store
```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Create vector embeddings and store in Chroma
db = Chroma.from_documents(documents, OpenAIEmbeddings())

# Perform a similarity search
query = "Who are the authors of 'Attention is All You Need'?"
retrieved_results = db.similarity_search(query)

# Output the first result
print(retrieved_results[0].page_content)
```
This code snippet creates vector embeddings using OpenAI and stores them in a Chroma vector store. It then performs a similarity search to find relevant chunks.

### FAISS Vector Store
```python
from langchain_community.vectorstores import FAISS

# Create vector embeddings and store in FAISS
db = FAISS.from_documents(documents, OpenAIEmbeddings())

# Perform a similarity search
query = "Who are the authors of 'Attention is All You Need'?"
retrieved_results = db.similarity_search(query)

# Output the first result
print(retrieved_results[0].page_content)
```
This section demonstrates creating vector embeddings with OpenAI and storing them in a FAISS vector store. It also performs a similarity search for a specific query.

