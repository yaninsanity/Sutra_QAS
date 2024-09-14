from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from core import preprocess_query, search_with_fallback
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import logging
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the Flask app to use the frontend folder for templates and static files
app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

socketio = SocketIO(app)

# Initialize Chroma DB
CHROMA_PATH = "../chroma"
embedding_function = OpenAIEmbeddings()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

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

conversation_history = []

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('send_message')
def handle_message(data):
    try:
        query_text = data.get('message')
        
        if not query_text:
            logger.warning("No query provided by the user.")
            emit('receive_message', {"message": "No query provided", "source": "error"})
            return

        logger.info(f"Received query: {query_text}")
        
        # Preprocess the query
        processed_query = preprocess_query(query_text)
        logger.info(f"Processed query: {processed_query}")
        
        # Search the database
        results = search_with_fallback(db, processed_query, top_k=5, initial_threshold=0.7)
        
        if not results:
            logger.info("No relevant context found in the database.")
            emit('receive_message', {"message": "No relevant context found.", "source": "system"})
            return
        
        # Prepare context from the results
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
        logger.info(f"Retrieved context: {context_text}")
        # Log the retrieved documents
        sources = [doc.metadata.get("source", "Unknown source") for doc, _ in results]
        logger.info(f"Retrieved documents: {sources}")
        
        # Update conversation history
        conversation_history.append(f"User: {query_text}")
        
        # Prepare the prompt with conversation history
        history_text = "\n".join(conversation_history)
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(conversation_history=history_text, context=context_text, question=query_text)

        # Generate the response
        model = ChatOpenAI(model="gpt-4o-mini", max_retries=2)
        response_text = model.predict(prompt)

        # Append model response to conversation history
        conversation_history.append(f"Assistant: {response_text}")

        # Send response back to the client
        emit('receive_message', {"message": response_text, "source": "assistant", "context": context_text, "sources": sources})
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        emit('receive_message', {"message": "An internal error occurred", "source": "error"})

@socketio.on('reset_conversation')
def reset_conversation():
    global conversation_history
    conversation_history = []
    logger.info("Conversation history reset.")
    emit('receive_message', {"message": "Conversation history reset.", "source": "system"})

if __name__ == '__main__':
    socketio.run(app, debug=True)
