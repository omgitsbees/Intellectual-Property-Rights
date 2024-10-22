import pandas as pd
import matplotlib.pyplot as plt
import requests
import numpy as np

# Load the dataset from Kaggle
patents = pd.read_csv('patentsview.csv')

# Clean and preprocess the data
patents = patents.dropna()  # drop rows with missing values

# Convert date column to datetime
patents['date'] = pd.to_datetime(patents['date'])

# Extract top 10 countries by patent count
top_countries = patents['country'].value_counts().head(10)

# Plot bar chart of top countries
plt.bar(top_countries.index, top_countries.values)
plt.xlabel('Country')
plt.ylabel('Number of Patents')
plt.title('Top 10 Countries by Patent Count')
plt.show()

# Time-series analysis of patent filings
filing_counts = patents['date'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
plt.plot(filing_counts.index, filing_counts.values)
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Patent Filings Over Time')
plt.show()

# Geographic analysis of patent holders
country_counts = patents['country'].value_counts()

plt.figure(figsize=(10, 6))
plt.bar(country_counts.index, country_counts.values)
plt.xlabel('Country')
plt.ylabel('Count')
plt.title('Patent Holders by Country')
plt.show()