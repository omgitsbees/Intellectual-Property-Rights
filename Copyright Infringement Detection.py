import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# Load the datasets
copyrighted_texts = pd.read_csv('copyrighted_texts.csv')
potentially_infringing_texts = pd.read_csv('potentially_infringing_texts.csv')
copyrighted_images = pd.read_csv('copyrighted_images.csv')
potentially_infringing_images = pd.read_csv('potentially_infringing_images.csv')

# Preprocess the text data
copyrighted_texts['text'] = copyrighted_texts['text'].apply(lambda x: word_tokenize(x))
potentially_infringing_texts['text'] = potentially_infringing_texts['text'].apply(lambda x: word_tokenize(x))

# Remove stopwords
stop_words = set(stopwords.words('english'))
copyrighted_texts['text'] = copyrighted_texts['text'].apply(lambda x: [word for word in x if word not in stop_words])
potentially_infringing_texts['text'] = potentially_infringing_texts['text'].apply(lambda x: [word for word in x if word not in stop_words])

# Convert the text data to TF-IDF vectors
vectorizer = TfidfVectorizer()
copyrighted_text_vectors = vectorizer.fit_transform(copyrighted_texts['text'])
potentially_infringing_text_vectors = vectorizer.transform(potentially_infringing_texts['text'])

# Preprocess the image data
copyrighted_image_paths = copyrighted_images['path'].tolist()
potentially_infringing_image_paths = potentially_infringing_images['path'].tolist()

# Define a function to extract features from images
def extract_features(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    features = model.predict(image)
    return features

# Extract features from copyrighted images
copyrighted_image_features = []
for image_path in copyrighted_image_paths:
    features = extract_features(image_path)
    copyrighted_image_features.append(features)

# Extract features from potentially infringing images
potentially_infringing_image_features = []
for image_path in potentially_infringing_image_paths:
    features = extract_features(image_path)
    potentially_infringing_image_features.append(features)

# Calculate the cosine similarity between text vectors
text_similarity_matrix = cosine_similarity(copyrighted_text_vectors, potentially_infringing_text_vectors)

# Calculate the cosine similarity between image features
image_similarity_matrix = cosine_similarity(copyrighted_image_features, potentially_infringing_image_features)

# Detect copyright infringement
infringement_threshold = 0.5
infringing_texts = []
infringing_images = []
for i in range(len(text_similarity_matrix)):
    for j in range(len(text_similarity_matrix[i])):
        if text_similarity_matrix[i][j] > infringement_threshold:
            infringing_texts.append((copyrighted_texts.iloc[i]['text'], potentially_infringing_texts.iloc[j]['text']))
for i in range(len(image_similarity_matrix)):
    for j in range(len(image_similarity_matrix[i])):
        if image_similarity_matrix[i][j] > infringement_threshold:
            infringing_images.append((copyrighted_image_paths[i], potentially_infringing_image_paths[j]))

print(infringing_texts)
print(infringing_images)