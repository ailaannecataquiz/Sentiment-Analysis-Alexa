# Sentiment Analysis on Amazon Alexa Reviews

## Overview
This project builds a machine learning model to classify Amazon Alexa customer reviews as **Positive** or **Negative** sentiment.  
It demonstrates an end-to-end NLP pipeline including data cleaning, TF-IDF feature engineering, model training, evaluation, and inference.

## Why This Project
This project shows how businesses can find the sentiment value of customer feedback to help:
- product improvement decisions
- monitor customer experiences
- report trends

## Key Skills Demonstrated
- Data cleaning + preprocessing (punctuation removal, stopword handling, negation preservation)
- Feature engineering using **TF-IDF** (unigrams + bigrams)
- Baseline modeling with **Naive Bayes**
- Improved modeling with **Logistic Regression** (handles class imbalance)
- Model evaluation using confusion matrix + classification report
- Inference on new/unseen reviews

## Dataset
- Source: Amazon Alexa Reviews dataset ('amazon_reviews.csv')
https://www.kaggle.com/datasets/sid321axn/amazon-alexa-reviews/data
- Text column: `verified reviews`
- Target column: `feedback`
    - `1` = Positive
    - `0` = Negative
    * Imbalanced dataset with a majority of positive reviews

## Project Structure
├── Sentiment_Analysis_Notebook.ipynb
├── amazon_reviews.csv
├── requirements.txt
├── README.md
└── .gitignore

## How to Run
1. Clone the repository: 
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

2. Create and activate a virtual environment
Windows (PowerShell):
python -m venv .venv
.venv\Scripts\activate

Mac/Linux:
python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Open the notebook
Run Sentiment_Analysis_Notebook.ipynb in Jupyter Notebook or VS Code

## Modeling Approach
Text Preprocessing
- convert to lowercase
- remove punctuation and special characters
- remove stopwords, keep negation words

Feature Engineering
TF-IDF vectorization: 
- unigrams + bigrams
- minimum document frequency filtering
- limited feature size

## Results
**Naive Bayes** Baseline
Accuracy: 0.92
Macro F1: 0.56
Negative Recall: 0.10

**Logistic Regression** Improved
Accuracy: 0.95
Macro F1: 0.84
Negative Recall: 0.75

## Key Learnings
- Accuracy can be misleading with imbalanced datasets.
- Macro F1 and class specific recall are essential for evaluating performance.
- Decisions during preprocessing stage can significantly change outcomes
- Baseline models are important in measuring improvement

## Final Thoughts
- The final model still shows negative reviews with 67% in precision, which shows a need for improvements. 
- Preprocessing adjustments must be made.
- Error analysis can help in determining what preprocessing measures should be made.
- Reviews can fluctuate in spelling and colloquialism and have to take account for repeated characters and emojis.


## Test/Demo
You can use the predict_sentiment() function to classify new reviews

def sentiment_label(pred):
    return "Positive" if pred == 1 else "Negative"

def predict_sentiment(review_text, model=log_reg):
    cleaned = message_cleaning(review_text)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]
    return sentiment_label(prediction)

# Examples
print(predict_sentiment("I hate the amazon echo, I never want to see it again!"))
print(predict_sentiment("We love our amazon echo, I highly recommend it for everyone :))"))
print(predict_sentiment("Not good, don't buy"))

# Output
Negative
Positive
Negative