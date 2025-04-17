## 🧬 Project Overview
Customer feedback plays a vital role in shaping product innovation and user experience. In this project, Natural Language Processing (NLP) techniques are applied to an Amazon Alexa product review dataset containing over 3,000 entries. The primary objective is to analyze consumer sentiment and train a machine learning model capable of classifying reviews as positive or negative. By leveraging text cleaning, lemmatization, and Random Forest classification, this project aims to uncover meaningful insights from unstructured text data and demonstrate how NLP can aid businesses in understanding customer feedback at scale.

## 🚀 How to Run
To get this project up and running on your local machine, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/Raxeira/nlp-alexa-reviews.git
   cd nlp-alexa-reviews
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. The amazon_alexa.tsv dataset is already included in the `data/` folder for convenience, so you can download it and use it straightaway. If you encounter any issues with the included file, you can manually download the dataset from [HAM10000 dataset on Kaggle](https://www.kaggle.com/datasets/sid321axn/amazon-alexa-reviews).
4. Launch the Jupyter notebook:
   ```bash
   jupyter notebook alexa-reviews-analysis.ipynb

## 📊 Dataset
The dataset includes customer reviews for Amazon Alexa products such as Echo Dots, Firesticks, and more. It contains the following fields:
- rating: Star rating of the product
- date: Date of review
- variation: Product variation
- verified_reviews: The written review
- feedback: Binary sentiment label (1 = Positive, 0 = Negative)

## 🔍 Exploratory Data Analysis
Various visualizations and statistics were generated to explore the dataset:
- Sentiment distribution of reviews (positive vs. negative)
- Feedback breakdown by product variation and star rating
- Time-based trends in reviews
- WordCloud of frequently used terms in feedback

## 🧹 Data Preprocessing
Essential data preprocessing steps included:
- Handling missing values in the verified_reviews column
- Converting date column to datetime format
- Text cleaning: lowercasing, punctuation removal, stopword removal
- Lemmatization with POS tagging for better context
- One-hot encoding of the variation column
- Vectorization of review text using CountVectorizer with unigrams and bigrams

## 🧠 Modeling
A Random Forest classifier was used to train a binary sentiment prediction model. Key steps:
- Combined one-hot encoded product variations and vectorized review text
- Split dataset using stratified sampling (80/20)
- Excluded rating and date columns to avoid data leakage
- Trained and evaluated the model using accuracy, precision, recall, and AUC

## 📈 Evaluation
The Random Forest model achieved:
- Accuracy: 93.97%
- AUC-ROC Score: 0.90
- Precision & Recall: High for positive class; lower for negative class due to class imbalance
- Confusion Matrix and ROC curve were plotted to assess performance
- WordCloud visualizations revealed key sentiment-bearing words like “love,” “echo,” and “great”
