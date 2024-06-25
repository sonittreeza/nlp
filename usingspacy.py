import spacy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Initialize the PorterStemmer
stemmer = PorterStemmer()

# Sample text
text = "Hello there! How are you doing today? The weather is great, and Python is awesome. Let's learn NLP."

# Process the text with spaCy
doc = nlp(text)

# Sentence tokenization using spaCy
sentences_spacy = [sent.text for sent in doc.sents]

# Sentence tokenization using NLTK
sentences_nltk = sent_tokenize(text)

# Word tokenization using spaCy
words_spacy = [token.text for token in doc]

# Word tokenization using NLTK
words_nltk = word_tokenize(text)

# Get the list of stop words
stop_words = set(stopwords.words('english'))

# Remove stop words using NLTK
filtered_words_nltk = [word for word in words_nltk if word.lower() not in stop_words]

# Apply stemming using NLTK's PorterStemmer
stemmed_words = [stemmer.stem(token.text) for token in doc]

# Lemmatization using spaCy
lemmatized_words = [token.lemma_ for token in doc]

# Print results
print("Sentences (spaCy):")
print(sentences_spacy)

print("\nSentences (NLTK):")
print(sentences_nltk)

print("\nWords (spaCy):")
print(words_spacy)

print("\nWords (NLTK):")
print(words_nltk)

print("\nFiltered Words (NLTK without stop words):")
print(filtered_words_nltk)

print("\nStemmed Words (NLTK):")
print(stemmed_words)

print("\nLemmatized Words (spaCy):")
print(lemmatized_words)
