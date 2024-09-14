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
CHROMA_PATH = "chroma"
DATA_PATH = "data/sutra"
MODEL_NAME = 'paraphrase-MiniLM-L6-v2'
MAX_CHUNK_SIZE = 400  # Maximum chunk size in characters
SIMILARITY_THRESHOLD = 0.85  # Semantic similarity threshold
CHUNK_OVERLAP = 50  # Overlap to maintain context continuity

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_document_into_chunks(documents)
    save_to_chroma(chunks)

def load_documents():
    """Load documents from the specified directory."""
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

def split_document_into_chunks(documents: list[Document], max_chunk_size=MAX_CHUNK_SIZE, similarity_threshold=SIMILARITY_THRESHOLD, chunk_overlap=CHUNK_OVERLAP):
    """Splits documents into semantically coherent chunks."""
    model = SentenceTransformer(MODEL_NAME)
    chunked_documents = []

    for document in documents:
        # Split document into sentences
        sentences = nltk.sent_tokenize(document.page_content)
        # Create semantically coherent chunks
        chunks = create_semantic_chunks(sentences, model, max_chunk_size, similarity_threshold, chunk_overlap)
        for i, chunk in enumerate(chunks):
            chunk_metadata = document.metadata.copy()
            chunk_metadata['chunk_index'] = i
            chunked_documents.append(Document(page_content=chunk, metadata=chunk_metadata))
    
    print(f"Split {len(documents)} documents into {len(chunked_documents)} chunks.")
    return chunked_documents

def create_semantic_chunks(sentences, model, max_chunk_size=MAX_CHUNK_SIZE, similarity_threshold=SIMILARITY_THRESHOLD, chunk_overlap=CHUNK_OVERLAP):
    """Creates semantically coherent chunks from sentences."""
    sentence_embeddings = model.encode(sentences)
    chunks = []
    current_chunk = []
    current_chunk_size = 0

    for i, sentence in enumerate(sentences):
        sentence_len = len(sentence)
        
        if current_chunk and (current_chunk_size + sentence_len > max_chunk_size):
            chunks.append(" ".join(current_chunk))
            # Use chunk_overlap to add continuity between chunks
            overlap_sentences = current_chunk[-(chunk_overlap // len(current_chunk)):]
            current_chunk = overlap_sentences + [sentence]
            current_chunk_size = sum(len(s) for s in overlap_sentences) + sentence_len
        else:
            if current_chunk:
                # Check semantic similarity between the last sentence in the current chunk and the current sentence
                last_sentence_embedding = model.encode([current_chunk[-1]])
                current_sentence_embedding = sentence_embeddings[i].reshape(1, -1)
                similarity = util.pytorch_cos_sim(last_sentence_embedding, current_sentence_embedding)[0][0]
                
                if similarity < similarity_threshold:
                    # If similarity is below threshold, start a new chunk
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_chunk_size = sentence_len
                else:
                    current_chunk.append(sentence)
                    current_chunk_size += sentence_len
            else:
                # Start a new chunk if there is none
                current_chunk.append(sentence)
                current_chunk_size += sentence_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def save_to_chroma(chunks: list[Document]):
    """Saves chunks to Chroma vector store."""
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()
