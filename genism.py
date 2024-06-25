import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords
from gensim.corpora import Dictionary
from gensim.models import Word2Vec

# Download the necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Sample text
text = "Hello there! How are you doing today? The weather is great, and Python is awesome. Let's learn NLP."

# Sentence tokenization using nltk
sentences = sent_tokenize(text)
print("Sentences:")
print(sentences)

# Word tokenization using nltk
words = word_tokenize(text)
print("\nWords:")
print(words)

# Remove stop words using gensim
filtered_words = simple_preprocess(remove_stopwords(text))
print("\nFiltered Words (without stop words):")
print(filtered_words)

# Initialize the PorterStemmer
stemmer = PorterStemmer()

# Apply stemming to each word
stemmed_words = [stemmer.stem(word) for word in filtered_words]
print("\nStemmed Words:")
print(stemmed_words)

# Gensim's lemmatization using a Word2Vec model
# Creating sentences tokenized
tokenized_sentences = [simple_preprocess(sent) for sent in sentences]

# Build the Word2Vec model (this is a simple example, in practice you would use a pre-trained model or a larger corpus)
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Lemmatization function using the Word2Vec model
def lemmatize(word):
    return model.wv.most_similar(positive=[word], topn=1)[0][0] if word in model.wv else word

# Apply lemmatization to each word
lemmatized_words = [lemmatize(word) for word in filtered_words]
print("\nLemmatized Words:")
print(lemmatized_words)
