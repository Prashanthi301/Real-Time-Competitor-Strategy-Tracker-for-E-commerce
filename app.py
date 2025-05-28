import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from transformers import pipeline
from datetime import datetime
import json
import numpy as np

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
SLACK_WEBHOOK_API_KEY = st.secrets["SLACK_WEBHOOK_API_KEY"]

def truncate_text(text, max_length=512):
    if isinstance(text, str):
        return text[:max_length]
    elif pd.isna(text):
        return ""
    else:
        return str(text)[:max_length]

def load_competitor_data():
    data = pd.read_csv("competitor_data.csv")
    return data

def load_reviews_data():
    reviews = pd.read_csv("review_data_with_sentiments.csv")
    return reviews

def analyze_sentiment(reviews):
    sentiment_pipeline = pipeline("sentiment-analysis")
    return sentiment_pipeline(reviews)

def forecast_discounts_arima(data, future_days=5):
    data = data.sort_index()
    data['discount'] = pd.to_numeric(data['discount'], errors='coerce')
    data = data.dropna(subset=['discount'])

    discount_series = data['discount']

    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            raise ValueError("Index could not be converted to datetime") from e

    model = ARIMA(discount_series, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=future_days)
    forecast = np.round(forecast).astype(int)

    future_dates = pd.date_range(
        start=discount_series.index[-1] + pd.Timedelta(days=1), 
        periods=future_days
    ).date

    forecast_df = pd.DataFrame({'date': future_dates, 'predicted_discount': forecast})
    forecast_df.set_index('date', inplace=True)
    return forecast_df

def send_to_slack(data):
    payload = {"text": data}
    response = requests.post(
        SLACK_WEBHOOK_API_KEY,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )

def generate_strategy_recommendation(product_name, competitor_data, predicted_discounts, sentiment):
    date = datetime.now()
    prompt = f"""
    You are a highly skilled business strategist specializing in e-commerce.
    Based on the following details, suggest actionable strategies:

    1. **Product Name**: {product_name}

    2. **Competitor Data**:
    {competitor_data}
    **Predicted Discounts**: {predicted_discounts}

    3. **Sentiment Analysis**: {sentiment}

    4. **Today's Date**: {str(date)}

    ### Task:
    - Analyze competitor data and pricing trends
    - Leverage sentiment analysis
    - Use discount predictions
    - Recommend promotional campaigns

    Provide recommendations in this format:
    1. **Pricing Strategy**
    2. **Promotional Campaign Ideas**
    3. **Customer Satisfaction Recommendations**
    """

    message = [{"role":"user","content":prompt}]
    data = {
        "messages": message,
        "model": "llama3-8b-8192",
        "temperature": 0
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROQ_API_KEY}"}

    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(data),
        headers=headers,
    )
    res = res.json()
    return res['choices'][0]['message']['content']

# Streamlit App
st.set_page_config(page_title="E-commerce Competitor Strategy Dashboard", layout="wide")
st.title("E-commerce Competitor Strategy Dashboard")
st.sidebar.header("Select a Product")

products = [
    "GAMEZZ24 Play-Station 4 / PS4 Slim 500 GB Gaming Console",
    "Sony PlayStation®5 Digital Edition (slim) Console Video Game",
    "GAMEZZ24 PS 4 1TB slim/Horizon Zero Dawn CE/GT Sport II/Ratchet &Clank/PS+3M",
    "OWC 1.0 TB External Hard Drive Upgrade for Sony Playstation 4"
]

selected_product = st.sidebar.selectbox("Select a Product to analyze", products)

# Load data
competitor_data = load_competitor_data()
reviews_data = load_reviews_data()

# Filter data for selected product
product_data = competitor_data[competitor_data['title'] == selected_product].copy()
review_statements = reviews_data[reviews_data['title'] == selected_product]['review_statements']

st.header(f"Competitor Analysis for {selected_product}")
st.subheader("Competitor Data")
st.table(product_data.tail(5))

# Price trend chart
fig = px.line(product_data, x='date', y='price', 
              title='Price Trend Over Time', 
              labels={'date':'Date', 'price':'Price (₹)'})
st.plotly_chart(fig)

# Discount trend chart
if 'discount' in product_data.columns:
    fig = px.bar(product_data, x='date', y='discount', 
                 title='Discount Trend Over Time', 
                 labels={'date':'Date', 'discount':'Discount (%)'})
    st.plotly_chart(fig)

# Sentiment Analysis
if not review_statements.empty:
    processed_reviews = review_statements.apply(lambda x: truncate_text(x, 512)).tolist()
    sentiments = analyze_sentiment(processed_reviews)
    
    st.subheader("Customer Sentiment Analysis")
    sentiment_df = pd.DataFrame(sentiments)
    fig = px.bar(sentiment_df, x='label', title='Sentiment Analysis Results')
    st.plotly_chart(fig)
else:
    st.write("No reviews data available for this product.")
    sentiments = "No reviews data available"

# Forecasting
try:
    product_data['date'] = pd.to_datetime(product_data['date'])
    product_data.set_index("date", inplace=True)
    product_data = product_data.sort_index()
    
    if 'discount' in product_data.columns:
        product_data['discount'] = pd.to_numeric(product_data['discount'], errors='coerce')
        product_data = product_data.dropna(subset=['discount'])
        
        if not product_data.empty:
            product_data_with_predictions = forecast_discounts_arima(product_data)
            st.subheader("Competitor Predicted Discounts")
            st.table(product_data_with_predictions)
            
            # Generate recommendations
            recommendations = generate_strategy_recommendation(
                selected_product,
                product_data[['price', 'discount']][-5:],
                product_data_with_predictions,
                sentiments if not review_statements.empty else "No reviews data available"
            )
            
            st.subheader("Strategy Recommendations")
            st.write(recommendations)
            send_to_slack(recommendations)
except Exception as e:
    st.error(f"Error in forecasting: {str(e)}")
