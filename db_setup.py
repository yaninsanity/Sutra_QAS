import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai
import nltk
from sentence_transformers import SentenceTransformer, util

# Load environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Set environment variables to handle warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Constants
CHROMA_PATH = "chroma"  # Path to store the Chroma DB
DATA_PATH = "data/sutra"  # Path where sutra documents are stored
MODEL_NAME = 'paraphrase-MiniLM-L6-v2'  # Sentence Transformer model for chunking
MAX_CHUNK_SIZE = 400  # Maximum chunk size in characters
CHUNK_OVERLAP = 50  # Overlap to maintain context continuity

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_document_into_chunks(documents)
    save_to_chroma(chunks)

def load_documents():
    """Load documents from the specified directory."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data path {DATA_PATH} not found.")
        
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    
    if not documents:
        raise ValueError(f"No documents found in {DATA_PATH}")
    
    print(f"Loaded {len(documents)} documents from {DATA_PATH}")
    return documents

def split_document_into_chunks(documents: list, max_chunk_size=MAX_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Splits documents into semantically coherent chunks using a sentence transformer model.
    """
    # Initialize sentence transformer model for chunking
    model = SentenceTransformer(MODEL_NAME)
    chunked_documents = []

    for document in documents:
        try:
            # Tokenize document into sentences using NLTK
            sentences = nltk.sent_tokenize(document.page_content)
            # Create semantic chunks
            chunks = create_semantic_chunks(sentences, model, max_chunk_size, chunk_overlap)
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = document.metadata.copy()
                chunk_metadata['chunk_index'] = i
                chunked_documents.append(Document(page_content=chunk, metadata=chunk_metadata))

        except Exception as e:
            print(f"Error processing document {document.metadata.get('source', 'Unknown')}: {str(e)}")

    print(f"Split {len(documents)} documents into {len(chunked_documents)} chunks.")
    return chunked_documents

def create_semantic_chunks(sentences, model, max_chunk_size=MAX_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Creates semantically coherent chunks from sentences.
    """
    sentence_embeddings = model.encode(sentences)
    chunks = []
    current_chunk = []
    current_chunk_size = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # Create a new chunk if the current chunk size exceeds the max_chunk_size
        if current_chunk and (current_chunk_size + sentence_len > max_chunk_size):
            chunks.append(" ".join(current_chunk))
            # Use chunk_overlap to maintain context between chunks
            overlap_sentences = current_chunk[-min(len(current_chunk), chunk_overlap):]
            current_chunk = overlap_sentences + [sentence]
            current_chunk_size = sum(len(s) for s in overlap_sentences) + sentence_len
        else:
            current_chunk.append(sentence)
            current_chunk_size += sentence_len

    # Append the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def save_to_chroma(chunks: list[Document]):
    """
    Saves chunks to Chroma vector store and ensures data is persisted.
    """
    # Clear out the database first to avoid any stale data
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"Cleared old Chroma database at {CHROMA_PATH}")

    # Create a new Chroma DB from the document chunks
    try:
        db = Chroma.from_documents(chunks, OpenAIEmbeddings(), collection_name="knowledge_base", persist_directory=CHROMA_PATH)
        print(f"Successfully saved {len(chunks)} chunks to {CHROMA_PATH}")
    except Exception as e:
        raise RuntimeError(f"Error saving to Chroma: {str(e)}")

if __name__ == "__main__":
    main()
