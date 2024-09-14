import nltk
import ssl

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')
