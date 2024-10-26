import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('ip_litigation_cases.csv')

# Preprocess the data
df['text'] = df['text'].apply(lambda x: word_tokenize(x))
stop_words = set(stopwords.words('english'))
df['text'] = df['text'].apply(lambda x: [word for word in x if word not in stop_words])

# Convert the data to TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(df['text'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tfidf, df['outcome'], test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# Perform network analysis
import networkx as nx
G = nx.Graph()
G.add_nodes_from(df['party'])
G.add_edges_from(df['relationship'])
nx.draw(G, with_labels=True)
plt.show()

# Perform geospatial analysis
import geopandas as gpd
gdf = gpd.GeoDataFrame(df, geometry='location')
gdf.plot()
plt.show()