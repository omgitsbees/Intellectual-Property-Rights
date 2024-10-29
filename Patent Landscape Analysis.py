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
import networkx as nx
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Load the dataset
df = pd.read_csv('patent_data.csv')

# Preprocess the data
df['text'] = df['text'].apply(lambda x: word_tokenize(x))
stop_words = set(stopwords.words('english'))
df['text'] = df['text'].apply(lambda x: [word for word in x if word not in stop_words])

# Convert the data to TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(df['text'])

# Classify the patents into different categories
kmeans = KMeans(n_clusters=5)
kmeans.fit(tfidf)
df['category'] = kmeans.labels_

# Create a patent map
G = nx.Graph()
G.add_nodes_from(df['patent_number'])
G.add_edges_from(df['citations'])
nx.draw(G, with_labels=True)
plt.show()

# Analyze the patent data
print(df['category'].value_counts())
print(df['patent_number'].value_counts())
print(df['citations'].value_counts())

# Perform geospatial analysis
gdf = gpd.GeoDataFrame(df, geometry='location')
gdf.plot()
plt.show()

# Perform PCA to reduce dimensionality
pca = PCA(n_components=2)
pca_tfidf = pca.fit_transform(tfidf)
plt.scatter(pca_tfidf[:, 0], pca_tfidf[:, 1])
plt.show()

# Perform t-SNE to visualize high-dimensional data
tsne = TSNE(n_components=2)
tsne_tfidf = tsne.fit_transform(tfidf)
plt.scatter(tsne_tfidf[:, 0], tsne_tfidf[:, 1])
plt.show()

# Train a random forest classifier to predict patent categories
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(tfidf, df['category'])
predictions = rfc.predict(tfidf)
print(accuracy_score(df['category'], predictions))
print(classification_report(df['category'], predictions))
print(confusion_matrix(df['category'], predictions))

# Perform grid search to optimize hyperparameters
param_grid = {'n_estimators': [10, 50, 100, 200], 'max_depth': [None, 5, 10, 15]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(tfidf, df['category'])
print(grid_search.best_params_)
print(grid_search.best_score_)