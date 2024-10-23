import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from seaborn import heatmap
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import sessionmaker

# Load the dataset
df = pd.read_csv('trademarks.csv')

# Preprocess the data
df['text'] = df['text'].apply(lambda x: x.lower())
df['text'] = df['text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
df['text'] = df['text'].apply(lambda x: word_tokenize(x))

# Remove stop words
stop_words = set(stopwords.words('english'))
df['text'] = df['text'].apply(lambda x: [word for word in x if word not in stop_words])

# Perform similarity analysis
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(df['text'])
similarity_matrix = cosine_similarity(tfidf, tfidf)

# Train a machine learning model
X_train, X_test, y_train, y_test = train_test_split(similarity_matrix, df['infringement'], test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')

# Visualize the results
plt.hist(predictions, bins=50)
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
plt.title('Distribution of Similarity Scores')
plt.show()

# Create a Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///trademarks.db'
db = SQLAlchemy(app)

# Define a Trademark class
class Trademark(db.Model):
    id = Column(Integer, primary_key=True)
    text = Column(String)
    infringement = Column(Integer)

# Create the database
db.create_all()

# Define a function to predict infringement
def predict_infringement(text):
    # Preprocess the text
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]

    # Convert the text to a TF-IDF vector
    vector = vectorizer.transform([text])

    # Calculate the similarity score
    similarity_score = cosine_similarity(vector, tfidf)

    # Make a prediction
    prediction = model.predict(similarity_score)

    return prediction

# Define a route for predicting infringement
@app.route('/predict', methods=['POST'])
def predict():
    text = request.get_json()['text']
    prediction = predict_infringement(text)
    return jsonify({'prediction': prediction})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)