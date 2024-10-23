Patent Data Analysis

This repository contains a Python script that performs analysis and visualization on a dataset of patents, sourced from Kaggle. The script covers various analyses, including top patent-holding countries, patent filings over time, and geographic distribution of patent holders.
Features

    Top Countries by Patent Count: A bar chart visualizing the top 10 countries with the most patent filings.
    Time-Series Analysis of Patent Filings: A line graph showing the number of patents filed over time.
    Geographic Analysis of Patent Holders: A bar chart depicting the number of patents held by country.

Installation

    Clone the repository:

    bash

git clone https://github.com/your-username/patent-data-analysis.git

Navigate to the project directory:

bash

cd patent-data-analysis

Install the required dependencies:

bash

    pip install pandas matplotlib requests numpy

Dataset

The dataset patentsview.csv should be downloaded from Kaggle and placed in the project directory. The dataset must contain at least the following columns:

    date: The date of the patent filing.
    country: The country of the patent holder.

Usage

To run the analysis, simply execute the Python script:

bash

python patent_analysis.py

Visualizations

    Top 10 Countries by Patent Count: This bar chart shows the number of patents filed by the top 10 countries.
    Patent Filings Over Time: This time-series plot visualizes patent filings over the years.
    Patent Holders by Country: This bar chart displays the distribution of patent holders across all countries.

Code Breakdown

    Data Loading: The script reads a CSV file containing patent data.

    python

patents = pd.read_csv('patentsview.csv')

Data Cleaning: The script removes rows with missing values and converts the date column to a datetime format.

python

patents = patents.dropna()
patents['date'] = pd.to_datetime(patents['date'])

Top Countries by Patent Count: The top 10 countries are identified by the number of patent filings, and a bar chart is plotted.

python

top_countries = patents['country'].value_counts().head(10)
plt.bar(top_countries.index, top_countries.values)

Time-Series Analysis: The number of patent filings is analyzed over time and visualized using a line plot.

python

filing_counts = patents['date'].value_counts().sort_index()
plt.plot(filing_counts.index, filing_counts.values)

Geographic Analysis: The number of patents held by country is visualized using a bar chart.

python

    country_counts = patents['country'].value_counts()
    plt.bar(country_counts.index, country_counts.values)

Requirements

    Python 3.x
    pandas
    matplotlib
    requests
    numpy

Install the required libraries using:

bash

pip install -r requirements.txt

License

This project is licensed under the MIT License.

-----------------------------------------------------------------------------------------------------------------

Trademark Infringement Detection System

This project uses Natural Language Processing (NLP) and machine learning to predict potential trademark infringements based on text similarity. The system processes trademark descriptions, computes similarity scores, trains a classification model, and provides an API for predicting trademark infringement via a Flask web application.
Features

    Data Preprocessing: Clean and tokenize text, remove stopwords, and prepare data for model training.
    Similarity Analysis: Use TF-IDF vectorization and cosine similarity to compare text samples.
    Machine Learning Model: Logistic Regression model to classify trademark infringement based on text similarity.
    Evaluation and Visualization: Evaluate model accuracy and visualize the distribution of similarity scores.
    Flask API: Provides an API endpoint to predict trademark infringement based on input text.

Requirements

    Python 3.x
    Libraries:
        pandas
        numpy
        nltk
        scikit-learn
        matplotlib
        seaborn
        flask
        flask_sqlalchemy
        sqlalchemy

You can install the required libraries using:

bash

pip install -r requirements.txt

Dataset

The dataset (trademarks.csv) contains the following columns:

    text: Description of the trademark
    infringement: Label indicating whether the trademark is an infringement (1 for infringement, 0 for no infringement)

Project Structure

    Data Preprocessing: The text is converted to lowercase, punctuation is removed, and tokens are generated using NLTK. Stopwords are also removed.
    TF-IDF and Cosine Similarity: Text is transformed into TF-IDF vectors, and cosine similarity scores are calculated between the texts.
    Model Training: Logistic Regression is used to train a classifier on the similarity matrix.
    Evaluation: Model accuracy is calculated, and a histogram of the similarity score distribution is generated.
    Flask API: Provides an endpoint /predict for predicting infringement based on the text input.

Setup

    Clone the repository:

    bash

git clone https://github.com/yourusername/trademark-infringement-detector.git

Install the required Python packages:

bash

pip install -r requirements.txt

Load your trademarks.csv dataset into the project directory.

Run the Flask app:

bash

    python app.py

    The API will be available at http://127.0.0.1:5000/predict. You can send POST requests to this endpoint with the text data to receive a prediction.

Example API Usage

Send a POST request to the /predict endpoint with a JSON body:

json

{
  "text": "Example trademark description"
}

The response will contain a prediction:

json

{
  "prediction": [1]
}

Model Evaluation

    Accuracy: The accuracy of the model on the test set is printed during evaluation.
    Confusion Matrix & Classification Report: Add additional evaluation if needed using confusion_matrix and classification_report from scikit-learn.

Visualization

The distribution of similarity scores is visualized using a histogram:

python

plt.hist(predictions, bins=50)
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
plt.title('Distribution of Similarity Scores')
plt.show()

Database Setup

The Flask application uses SQLite to store trademarks. A Trademark model is defined with the following fields:

    id: Unique identifier
    text: Trademark description
    infringement: Infringement label (1 or 0)

Run the following command to create the database:

python

db.create_all()

License

This project is licensed under the MIT License - see the LICENSE file for details.
