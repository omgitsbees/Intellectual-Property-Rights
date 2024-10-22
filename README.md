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
