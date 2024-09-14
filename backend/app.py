from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import logging
import openai
from core import preprocess_query, search_with_fallback

# Load environment variables
def load_env():
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')

# Configure the logger
def configure_logger():
    logger = logging.getLogger('app_logger')
    logger.setLevel(logging.INFO)
    # Create console handler and set level to info
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Add formatter to console handler
    ch.setFormatter(formatter)
    # Add console handler to logger
    logger.addHandler(ch)
    return logger

# Initialize Chroma DB
def init_chroma_db(logger):
    CHROMA_PATH = os.path.join(os.getcwd(), "chroma")  # Ensure path is correct
    embedding_function = OpenAIEmbeddings()

    try:
        logger.info(f"Initializing Chroma DB at path: {CHROMA_PATH}")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function, collection_name="knowledge_base")
        
        # Check if the collection has documents
        collection_size = len(db.get())  # Using get() to fetch documents
        
        if collection_size == 0:
            logger.error("Chroma database is empty. Please ensure data is loaded.")
        else:
            logger.info(f"Chroma database contains {collection_size} documents.")
            
        return db
    except Exception as e:
        logger.error(f"Error initializing Chroma database: {str(e)}")
        return None

# Initialize Flask and SocketIO
def create_app():
    app = Flask(
        __name__,
        template_folder="../frontend/templates",
        static_folder="../frontend/static"
    )
    socketio = SocketIO(app)
    return app, socketio

# Function to process message and interact with Chroma DB and OpenAI
def process_message(db, logger, query_text):
    # Preprocess the query
    processed_query = preprocess_query(query_text)
    logger.info(f"Processed query: {processed_query}")

    # Check if Chroma DB is initialized
    if not db:
        logger.error("Chroma database is not initialized.")
        return None, "Database initialization error"
    
    # Perform similarity search in Chroma DB
    results = search_with_fallback(db, processed_query, logger, top_k=5)
    logger.info(f"Search results: {results}")

    if not results:
        logger.info("No relevant context found in the database.")
        return None, "No relevant context found"

    # Prepare context from the search results
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    logger.info(f"Retrieved context: {context_text[:200]}")  # Log only the first 200 chars for debugging

    # Log document sources for debugging
    sources = [doc.metadata.get("source", "Unknown source") for doc, _ in results]
    logger.info(f"Retrieved document sources: {sources}")
    
    # Return context and sources for further processing
    return context_text, sources

# Prepare the prompt for GPT-4 model and get the response
def generate_gpt_response(logger, context_text, query_text, conversation_history):
    # Update conversation history
    conversation_history.append(f"User: {query_text}")
    
    # Prepare the prompt with conversation history
    history_text = "\n".join(conversation_history)
    PROMPT_TEMPLATE = """
    As a wise and enlightened Buddhist master, provide a comprehensive and insightful interpretation of the following sutra-related question.
    Ensure your explanation thoroughly reflects the provided context, without omitting any key details. 
    If the context does not fully address the question, clearly indicate what is missing and provide a thoughtful explanation based on your deep understanding of Buddhist teachings.

    {conversation_history}

    ---
    Context:
    {context}

    Question: {question}
    Your Enlightened Response:
    """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(conversation_history=history_text, context=context_text, question=query_text)

    # Generate response using the GPT model
    model = ChatOpenAI(model="gpt-4o-mini", max_retries=2)
    response_text = model.predict(prompt)

    # Append model response to conversation history
    conversation_history.append(f"Assistant: {response_text}")
    logger.info("Generated response from GPT model.")

    return response_text

# Flask SocketIO message handling function
def handle_message(data, db, logger, conversation_history):
    query_text = data.get('message')
    
    if not query_text:
        logger.warning("No query provided by the user.")
        return "No query provided", "error"

    logger.info(f"Received query: {query_text}")
    
    context_text, sources = process_message(db, logger, query_text)
    if context_text is None:
        return sources, "error"

    response_text = generate_gpt_response(logger, context_text, query_text, conversation_history)
    return response_text, "assistant", context_text, sources

# Main function to initialize the app and set up routes
def main():
    # Load environment variables and initialize logger
    load_env()
    logger = configure_logger()

    # Initialize the database
    db = init_chroma_db(logger)

    # Create Flask app and SocketIO
    app, socketio = create_app()

    conversation_history = []

    @app.route('/')
    def index():
        return render_template('index.html')

    @socketio.on('send_message')
    def socket_message_handler(data):
        response_text, source, context_text, sources = handle_message(data, db, logger, conversation_history)
        emit('receive_message', {"message": response_text, "source": source, "context": context_text, "sources": sources})

    @socketio.on('reset_conversation')
    def reset_conversation():
        nonlocal conversation_history
        conversation_history = []
        logger.info("Conversation history reset.")
        emit('receive_message', {"message": "Conversation history reset.", "source": "system"})

    socketio.run(app, debug=True, host="0.0.0.0", port=5000)

if __name__ == '__main__':
    main()
