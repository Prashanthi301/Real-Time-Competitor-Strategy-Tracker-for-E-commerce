# -*- coding: utf-8 -*-
import streamlit as st
import os
import pandas as pd
import plotly.express as px
import requests
from transformers import pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import json

# Set up Streamlit configuration
st.set_page_config(page_title="E-Commerce Competitor Strategy Dashboard", layout="wide")

# Install necessary libraries
os.system("pip install Pillow")

# Display Title
st.title("Welcome to the E-Commerce Competitor Strategy Dashboard")

# Constants
API_KEY = "your_api_key_here"  # Replace with your actual API key
SLACK_WEBHOOK = "https://hooks.slack.com/services/your/webhook/url"  # Replace with your Slack webhook URL

# Dummy product list
products = ["Sony PS4 Slim 1 TB Console", "PS4 GOW HITS", "PS4 Slim 500 GB"]

# Sidebar for product selection
selected_product = st.sidebar.selectbox("Choose a product to analyze:", products, key="product_selector")


# Utility Functions
def truncate_text(text, max_length=512):
    return text[:max_length]


def load_competitor_data():
    """Load competitor data from a CSV file."""
    return pd.read_csv("competitor_data.csv")


def load_reviews_data():
    """Load reviews data from a CSV file."""
    return pd.read_csv("reviews.csv")


def analyze_sentiment(reviews):
    """Analyze customer sentiment for reviews."""
    sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    sentiments = [sentiment_pipeline(review)[0]["label"] for review in reviews]
    return sentiments


def train_predictive_model(data):
    """Train a predictive model for competitor pricing strategy."""
    data["discount"] = data["discount"].str.replace("%", "").astype(float)
    data["price"] = data["price"].astype(float)
    data["Predicted_Discount"] = data["discount"] + (data["price"] * 0.05).round(2)

    X = data[["price", "discount"]]
    y = data["Predicted_Discount"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model


def forecast_discounts_arima(data, future_days=5):
    """Forecast future discounts using ARIMA."""
    data = data.sort_index()
    data["discount"] = pd.to_numeric(data["discount"], errors="coerce").dropna()

    if data.empty or "discount" not in data.columns:
        return pd.DataFrame(columns=["date", "Predicted_Discount"])

    model = ARIMA(data["discount"], order=(2, 1, 0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=future_days)
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days)

    forecast_df = pd.DataFrame({"date": future_dates, "Predicted_Discount": forecast})
    forecast_df.set_index("date", inplace=True)
    return forecast_df


def send_to_slack(message):
    payload = {"text": message}
    requests.post(SLACK_WEBHOOK, data=json.dumps(payload), headers={"Content-Type": "application/json"})


def generate_strategy_recommendation(product, competitor_data, sentiments):
    """Generate strategy recommendations."""
    prompt = f"""
    Based on the data for product '{product}', analyze the competitor data and sentiments.
    Competitor Data: {competitor_data.to_dict()}
    Sentiments: {sentiments}
    Provide actionable recommendations for pricing, promotions, and customer satisfaction.
    """
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json={"messages": [{"role": "user", "content": prompt}], "model": "gpt-3.5-turbo"}
    )
    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No recommendations available.")
    else:
        return "Failed to generate recommendations."


# Load data
competitor_data = load_competitor_data()
reviews_data = load_reviews_data()

# Filter data for the selected product
competitor_data = competitor_data[competitor_data["title"] == selected_product]
product_reviews = reviews_data[reviews_data["title"] == selected_product]

# Display competitor data
st.header(f"Competitor Analysis for {selected_product}")
st.subheader("Competitor Data")
st.table(competitor_data)

# Sentiment analysis
if not product_reviews.empty:
    reviews = product_reviews["review_statements"].apply(truncate_text).tolist()
    sentiments = analyze_sentiment(reviews)

    st.subheader("Customer Sentiment Analysis")
    sentiment_df = pd.DataFrame(sentiments, columns=["Sentiment"])
    fig = px.bar(sentiment_df["Sentiment"].value_counts(), title="Sentiment Analysis Results")
    st.plotly_chart(fig)
else:
    st.write("No reviews available for this product.")

# Forecast discounts
competitor_data["date"] = pd.to_datetime(competitor_data["date"], errors="coerce")
competitor_data.set_index("date", inplace=True)
forecast_df = forecast_discounts_arima(competitor_data)

st.subheader("Competitor Current and Predicted Discounts")
st.table(forecast_df)

# Generate recommendations
recommendations = generate_strategy_recommendation(
    selected_product,
    forecast_df,
    sentiments if "sentiments" in locals() else "No reviews available"
)
st.subheader("Strategic Recommendations")
st.write(recommendations)

# Send to Slack
send_to_slack(recommendations)

