import os
# from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import openai

# Load environment variables
# load_dotenv()
# openai.api_key = os.getenv('OPENAI_API_KEY')

# CHROMA_PATH = "../chroma"  # Adjusted to point to the correct directory

# PROMPT_TEMPLATE = """
# As a wise and enlightened Buddhist master, provide a comprehensive and insightful interpretation of the following sutra-related question.
# Ensure your explanation thoroughly reflects the provided context, without omitting any key details. 
# If the context does not fully address the question, clearly indicate what is missing and provide a thoughtful explanation based on your deep understanding of Buddhist teachings.
# Context (retrieved from the knowledge base):
# {context}

# ---
# Question: {question}
# Your Enlightened Response:
# """

def preprocess_query(query_text):
    """
    Preprocess the query text to ensure it's optimized for similarity search.
    """
    sentences = sent_tokenize(query_text)
    if len(sentences) <= 1 and len(query_text.split()) <= 10:
        return query_text

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    keywords = vectorizer.get_feature_names_out()

    focused_query = " ".join(keywords)
    return focused_query

def search_with_fallback(db, query_text, top_k, initial_threshold=0.7, fallback_threshold=0.5):
    """
    Perform a similarity search with an initial threshold. If no results are found,
    retry with a lower threshold.
    """
    results = db.similarity_search_with_relevance_scores(query_text, k=top_k)
    if not results or results[0][1] < initial_threshold:
        print(f"Unable to find matching results with a threshold of {initial_threshold}. Trying with a lower threshold...")
        results = db.similarity_search_with_relevance_scores(query_text, k=top_k)
        if not results or results[0][1] < fallback_threshold:
            print(f"Unable to find matching results even with a fallback threshold of {fallback_threshold}.")
            return None
    return results

# def generate_response(query_text, threshold=0.7, top_k=5):
#     """
#     Main function to process the query and generate a response using the RAG approach.
#     """
#     embedding_function = OpenAIEmbeddings()
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

#     processed_query = preprocess_query(query_text)

#     results = search_with_fallback(db, processed_query, top_k, initial_threshold=threshold)
#     if not results:
#         return "No relevant information found."

#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=query_text)

#     model = ChatOpenAI(model="gpt-4o-mini", max_retries=2)
#     response_text = model.predict(prompt)

#     sources = [doc.metadata.get("source", None) for doc, _ in results]
#     return f"Response: {response_text}\nSources: {sources}"
