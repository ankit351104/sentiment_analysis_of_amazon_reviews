
pip install gradio

from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import pandas as pd
import numpy as np
import random
import gradio as gr

# Mount Google Drive to access your dataset
drive.mount("/content/gdrive")

# Define classes for reviews and review container
class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment, self.probability = self.get_sentiment()

    def get_sentiment(self):
        if self.score <= 2:
            return "NEGATIVE", 1.0 if self.score == 0 else 0.8
        elif self.score == 3:
            return "NEUTRAL", 0.5
        else: #Score of 4 or 5
            return "POSITIVE", 1.0 if self.score == 5 else 0.8

class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews

    def get_text(self):
        return [x.text for x in self.reviews]

    def get_sentiment(self):
        return [x.sentiment for x in self.reviews]

    def get_probability(self):
        return [x.probability for x in self.reviews]

    def evenly_distribute(self):
        negative = list(filter(lambda x: x.sentiment == "NEGATIVE", self.reviews))
        positive = list(filter(lambda x: x.sentiment == "POSITIVE", self.reviews))
        positive_shrunk = positive[:len(negative)]
        self.reviews = negative + positive_shrunk
        random.shuffle(self.reviews)

# Load the dataset
df = pd.read_json('/content/gdrive/My Drive/Datasets/Books_subset3.json', lines=True)

# Create Review objects
reviews = [Review(x, y) for x, y in zip(df['review_body'], df['star_rating'])]

# Split data into train and test sets
train,test = train_test_split(reviews, test_size=0.25, random_state=42)

# Initialize ReviewContainer objects
train_container = ReviewContainer(train)
test_container = ReviewContainer(test)

# Evenly distribute the data
train_container.evenly_distribute()
train_x = train_container.get_text()
train_y = train_container.get_sentiment()

test_container.evenly_distribute()
test_x = test_container.get_text()
test_y = test_container.get_sentiment()

# Vectorize the data
vectorizer = TfidfVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)

# Train SVM model
clf_svm = svm.SVC(kernel='linear', probability=True)
clf_svm.fit(train_x_vectors, train_y)

# Define function for Gradio interface
def predict_sentiment(text):
    # Vectorize the input text
    text_vector = vectorizer.transform([text])

    # Predict sentiment
    sentiment = clf_svm.predict(text_vector)[0]

    # Get probabilities for each class
    probabilities = clf_svm.predict_proba(text_vector)[0]

    # Convert probabilities to percentages
    percentages = [round(prob * 100, 2) for prob in probabilities]

    # Choose the sentiment label based on the predicted class
    if sentiment == "POSITIVE":
        return f"Positive ({percentages[1]}%)"
    elif sentiment == "NEUTRAL":
        return f"Neutral ({percentages[2]}%)"
    else:
        return f"Negative ({percentages[0]}%)"

# Create Gradio interface
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(placeholder="Enter Text", lines=10, label="Enter your text here:"),
    outputs=gr.Textbox(label="Sentiment"),
    title="Sentiment Analysis Developed by Group-12(Ankit, Akshat, Gautam, Pritish) with ♥ from RCC Institute of Information Technology.",
    description="Enter text and predict sentiment"
)

# Launch the interface
iface.launch()

