import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from collections import Counter
import re

import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def preprocess_query(query_text):
    """
    Preprocess the query text to optimize it for similarity search.
    This includes basic text cleaning and keyword extraction using TF-IDF.
    """

    # Clean the query text (remove special characters and multiple spaces)
    cleaned_query = re.sub(r'[^a-zA-Z\s]', '', query_text).strip()

    # Tokenize the query into sentences
    sentences = sent_tokenize(cleaned_query)

    # If the query is short, return it as is to retain full information
    if len(sentences) <= 1 and len(cleaned_query.split()) <= 10:
        return cleaned_query

    # Initialize the TF-IDF vectorizer with built-in stopword handling
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=0.1, ngram_range=(1, 2))

    # Fit TF-IDF to the sentences and extract top features (keywords)
    X = vectorizer.fit_transform(sentences)
    keywords = vectorizer.get_feature_names_out()

    # Combine keywords into a focused query
    focused_query = " ".join(keywords)

    return focused_query


def search_with_fallback(db, query_text, logger, top_k=5, initial_threshold=0.5, fallback_threshold=0.3):
    """
    Perform a similarity search with an initial threshold, and retry with a lower threshold if no results are found.
    """
    try:
        # Perform the initial search
        results = db.similarity_search_with_relevance_scores(query_text, k=top_k)
        logger.info(f"Initial search results: {results}")

        if not results or results[0][1] < initial_threshold:
            logger.warning(f"No results found with the initial threshold. Trying fallback search...")
            # Retry with fallback threshold
            results = db.similarity_search_with_relevance_scores(query_text, k=top_k, threshold=fallback_threshold)
            logger.info(f"Fallback search results: {results}")
        
        if not results:
            logger.warning(f"No results found with fallback threshold either.")
        
        return results
    except Exception as e:
        logger.error(f"Error during search: {e}")
        return []

