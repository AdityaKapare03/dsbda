import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

document = 'Text analysis is the process of extracting meaningful insights from textual data. It involves techniques like tokenization, stemming and lemmatization.'

tokens = word_tokenize(document)
print('Tokens: ', tokens)

pos_tags = pos_tag(tokens)
print(pos_tags)

stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print('After stop words removal: ', filtered_words)

stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
print('Stemmed Tokens', stemmed_tokens)

lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
print(lemmatized_tokens)

text = " ".join(lemmatized_tokens)
from wordcloud import WordCloud
import matplotlib.pyplot as plt
wordcloud = WordCloud(width = 800, height = 400, background_color = 'white').generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

documents = [
    'Text analysis is the process of extracting meaningful insights from textual data.',
    'It involves techniques like tokenization, stemming and lemmatization.',
    'It provides summary or extract of the text.'
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

print('TF IDF matrix:')
print(tfidf_matrix.toarray())
print('Terms:', feature_names)
