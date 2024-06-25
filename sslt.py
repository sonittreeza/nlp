import nltk

# Download the necessary resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

# Sample text
text = "Hello there! How are you doing today? The weather is great, and Python is awesome. Let's learn NLP."

# Sentence tokenization
sentences = sent_tokenize(text)
print("Sentences:")
print(sentences)

# Word tokenization
words = word_tokenize(text)
print("\nWords:")
print(words)

# Initialize the PorterStemmer
stemmer = PorterStemmer()

# Apply stemming to each word
stemmed_words = [stemmer.stem(word) for word in words]
print("\nStemmed Words:")
print(stemmed_words)

# Initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Apply lemmatization to each word
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
print("\nLemmatized Words:")
print(lemmatized_words)

# Get the list of stop words
stop_words = set(stopwords.words('english'))

# Remove stop words
filtered_words = [word for word in words if word.lower() not in stop_words]
print("\nFiltered Words (without stop words):")
print(filtered_words)
