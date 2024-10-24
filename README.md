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

----------------------------------------------------------------------------------------------------------------

Copyright Infringement Detection Tool

This tool detects potential copyright infringement in both text and images by comparing copyrighted data with potentially infringing content. It utilizes techniques such as TF-IDF for text vectorization, cosine similarity for comparing text and image data, and VGG16 for extracting image features.
Features

    Text Preprocessing: Tokenization and removal of stopwords from the copyrighted and potentially infringing texts.
    Text Vectorization: Conversion of preprocessed text data into TF-IDF vectors for comparison.
    Image Feature Extraction: Use of the VGG16 model to extract features from images.
    Cosine Similarity Calculation: Determines how similar two pieces of text or images are.
    Detection of Infringement: Text or images with similarity scores above a specified threshold are flagged as potential copyright infringements.

Prerequisites

Before running the tool, ensure you have the following Python libraries installed:

bash

pip install pandas numpy nltk scikit-learn tensorflow pillow opencv-python

Additionally, you need to download the NLTK data for tokenization and stopwords:

bash

import nltk
nltk.download('punkt')
nltk.download('stopwords')

File Structure

You need to have two CSV files for text comparison and two CSV files for image comparison:

    copyrighted_texts.csv: Contains copyrighted text data with a column text.
    potentially_infringing_texts.csv: Contains potentially infringing text data with a column text.
    copyrighted_images.csv: Contains paths to copyrighted images with a column path.
    potentially_infringing_images.csv: Contains paths to potentially infringing images with a column path.

How to Run

    Clone the repository and navigate to the project directory.
    Ensure your CSV files are placed in the same directory.
    Run the script:

bash

python detect_copyright_infringement.py

How It Works

    Text Comparison:
        Loads and preprocesses text data (tokenization and removal of stopwords).
        Converts the text data into TF-IDF vectors.
        Computes cosine similarity between the copyrighted and potentially infringing text vectors.
        Flags texts with a similarity score higher than the set threshold (0.5 by default).

    Image Comparison:
        Loads image paths and processes them using the VGG16 deep learning model to extract features.
        Computes cosine similarity between the extracted image features.
        Flags images with a similarity score higher than the set threshold (0.5 by default).

    Infringement Detection:
        Prints the list of potentially infringing text and image pairs if their similarity exceeds the threshold.

Example Output

bash

[(['copyrighted', 'text'], ['potentially', 'infringing', 'text'])]
[('path/to/copyrighted_image.jpg', 'path/to/potentially_infringing_image.jpg')]

Customization

    Infringement Threshold: By default, the similarity threshold is set to 0.5. You can adjust this by modifying the infringement_threshold variable.

Dependencies

    Pandas: For handling CSV data.
    NumPy: For numerical operations.
    NLTK: For text tokenization and stopword removal.
    Scikit-learn: For TF-IDF vectorization and cosine similarity calculations.
    OpenCV: For reading image data (optional but recommended).
    PIL (Pillow): For image manipulation.
    TensorFlow (Keras): For extracting image features using the VGG16 model.

License

This project is licensed under the MIT License.
Acknowledgments

    The VGG16 model used in this tool is a deep learning model trained on the ImageNet dataset and provided by Keras.

Feel free to contribute to this project by submitting pull requests or reporting issues!
