
import streamlit as st
st.set_page_config(page_title="E-Commerce Competitor Strategy Dashboard", layout="wide")
import os
os.system("pip install Pillow")

import json
from datetime import datetime

from PIL import Image

import torch
import pandas as pd
import plotly.express as px
import requests
st.title("Welcome to the E-Commerce Competitor Strategy Dashboard")
from openai import AzureOpenAI
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA

API_KEY = "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" #Groq API Key
SLACK_WEBHOOK = "https://hooks.slack.com/services/your/webhook/url" #Slack webhook url

products = ["Sony PS4 Slim 1 TB Console"
, "PS4 GOW HITS"
, "Ps4 slim 500 gb",
]  # Replace with actual data

selected_product = st.sidebar.selectbox(
    "Choose a product to analyze:", 
    products, 
    key="product_selector"
)


def truncate_text(text, max_length=512):
    return text[:max_length]


def load_competitor_data():
     """Load competitor data from a CSV file."""
     data = pd.read_csv("competitor_data.csv")
     print(data.head())
     return data


def load_reviews_data():
    """Load reviews data from a CSV file."""
    reviews = pd.read_csv("reviews.csv")
    return reviews

from transformers import pipeline
def analyze_sentiment(reviews):
    """Analyze customer sentiment for reviews."""
    sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    sentiments = [sentiment_pipeline(review)[0]["label"] for review in reviews]
    return sentiments


def train_predictive_model(data):
    """Train a predictive model for competitor pricing strategy."""
    data["discount"] = data["discount"].str.replace("%", "").astype(float)
    data["price"] = data["price"].astype(int)
    data["Predicted_Discount"] = data["discount"] + (data["price"] * 0.05).round(2)

    x = data[["price", "discount"]]
    y = data["Predicted_Discount"]
    print(x)
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, train_size=0.8
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model


import numpy as np
import pandas as pd


def forecast_discounts_arima(data, future_days=5):
    """
    Forecast future discounts using ARIMA.
    :param data: DataFrame containing historical discount data (with a datetime index).
    :param future_days: Number of days to forecast.
    :return: DataFrame with historical and forecasted discounts.
    """

    data = data.sort_index()
    print(data.index)

    data["discount"] = pd.to_numeric(data["discount"], errors="coerce")
    data = data.dropna(subset=["discount"])

    discount_series = data["discount"]

    if discount_series.empty:
        # Return an empty DataFrame if discount_series is empty to avoid the error
        return pd.DataFrame(columns=['Date', 'Predicted_Discount']).set_index('Date')

    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            raise ValueError(
                "Index must be datetime or convertible to datetime."
            ) from e

    model = ARIMA(discount_series, order=(2, 1, 0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=future_days)
    future_dates = pd.date_range(
        start=discount_series.index[-1] + pd.Timedelta(days=1), periods=future_days
    )

    forecast_df = pd.DataFrame({"date": future_dates, "Predicted_Discount": forecast})
    forecast_df.set_index("date", inplace=True)

    return forecast_df


def send_to_slack(data):
    payload = {"text": data}
    response = requests.post(
        SLACK_WEBHOOK,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )


def generate_strategy_recommendation(title, competitor_data, sentiment):
    """Generate strategic recommendations using an LLM."""
    date = datetime.now()
    prompt = f"""
    You are a highly skilled business strategist specialilzing in e-commerce. Based on the following details, sug

1. *Product Name*: {title}

2. *Competitor Data*: (including current prices, discounts, and predicted discounts):
{competitor_data}

3. *Sentiment Analysis*:
{sentiment}


4. *Today's Date*: {str(date)}

### Task:
- Analyze the competitor data and identify key pricing trends.
- Leverage sentiment analysis insights to highlight areas where customer satisfaction can be improved.
- Use the discount predictions to suggest how pricing strategies can be optimized over the next 5 days.
- Recommend promotional campaigns or marketing strategies that align with customer sentiments and competitive tr
- Ensure the strategies are actionable, realistic, and geared toward increasing customer satisfaction, driving s

provide your recommendations in a structured format:
1. *pricing strategy*
2. *Promotional Campaign Ideas*
3. *Customer Satisfaction Recommendations*
   """

    messages = [{"role": "user", "content": prompt}]

    data = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "llama3-8b-8192",
        "temperature": 0,
    }

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(data),
        headers=headers,
    )
    print(f"Groq API Response Status Code: {res.status_code}")
    print(f"Groq API Response Content: {res.text}")
    res = res.json()
    if "choices" in res and res["choices"]:
       response = res["choices"][0]["message"]["content"]
    else:
       response= "Error: Unable to generate recommendations. Please check the API key and response." # Provide a default response in case of error
    return response


####----------------------------------------------#############


competitor_data = load_competitor_data()
reviews = load_reviews()

title = competitor_data[competitor_data["title"] == selected_product]
reviews = reviews[review_statements["title"] == selected_product]

st.header(f"Competitor Analysis for {selected_product}")
st.subheader("Competitor Data")
st.table(competitor_data.tail(5))

if not product_reviews.empty:
   product_reviews["reviews"] = product_reviews["reviews"].apply(
      lambda x: truncate_text(x, 512)
   )
   reviews = product_reviews["reviews"].tolist()
   sentiments = analyze_sentiment(reviews)

   st.subheader("Customer Sentiment Analysis")
   sentiment_df = pd.DataFrame(sentiments)
   fig = px.bar(sentiment_df, x="label", title="Sentiment Analysis Results")
   st.plotly_chart(fig)
else:
    st.write("No reviews available for this product.")


# Preprocessing

competitor_data["date"] = pd.to_datetime(competitor_data["date"], errors="coerce")
competitor_data = competitor_data.dropna(subset=["date"])
competitor_data.set_index("date", inplace=True)
competitor_data = competitor_data.sort_index()

competitor_data["discount"] = pd.to_numeric(competitor_data["discount"], errors="coerce")
competitor_data = competitor_data.dropna(subset=["discount"])

# Forecasting Model
competitor_data_with_predictions = forecast_discounts_arima(competitor_data)


st.subheader("Competitor Current and Predicted Discounts")
st.table(competitor_data_with_predictions.tail(10))



recommendations = generate_strategy_recommendation(
    selected_product,
    competitor_data_with_predictions,
    sentiments if not product_reviews.empty else "No reviews available",
)
st.subheader("Strategic Recommendations")
st.write(recommendations)
send_to_slack(recommendations)
selected_product = st.sidebar.selectbox("choose a product to analyze:", products)


competitor_data = load_competitor_data()
reviews = load_reviews_data()

competitor_data = competitor_data[competitor_data["title"] == selected_product]
product_reviews = reviews[review_statements["title"] == selected_product]

st.header(f"Competitor Analysis for {selected_product}")
st.subheader("Competitor Data")
st.table(competitor_data.tail(5))

if not product_reviews.empty:
   product_reviews["reviews"] = product_reviews["reviews"].apply(
      lambda x: truncate_text(x, 512)
   )
   reviews = product_reviews["reviews"].tolist()
   sentiments = analyze_sentiment(reviews)

   st.subheader("Customer Sentiment Analysis")
   sentimentdf = pd.DataFrame(sentiments)
   fig = px.bar(sentiment_df, x="label", title="Sentiment Analysis Results")
   st.plotly_chart(fig)
else:
    st.write("No reviews available for this product.")


# Preprocessing

competitor_data["date"] = pd.to_datetime(competitor_data["date"], errors="coerce")
competitor_data = competitor_data.dropna(subset=["date"])
competitor_data.set_index("date", inplace=True)
competitor_data = competitor_data.sort_index()

competitor_data["discount"] = pd.to_numeric(competitor_data["discount"], errors="coerce")
competitor_data = competitor_data.dropna(subset=["discount"])

# Forecasting Model
competitor_data_with_predictions = forecast_discounts_arima(competitor_data)


st.subheader("Competitor Current and Predicted Discounts")
st.table(competitor_data_with_predictions.tail(10))

if "review_statements" in product_reviews.columns:
    reviews = product_reviews["review_statements"].tolist()
else:
    st.error("The 'review_statements' column is missing from the dataset.")
    reviews = []


recommendations = generate_strategy_recommendation(
    selected_product,
    competitor_data_with_predictions,
    sentiments if not product_reviews.empty else "No reviews available",
)
st.subheader("Strategic Recommendations")
st.write(recommendations)
send_to_slack(recommendations)
