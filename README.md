# Real-Time-Competitor-Strategy-Tracker-for-E-commerce
## Project Overview
This project is a comprehensive E-commerce Competitor Strategy Dashboard designed to assist businesses in analyzing competitor data, predicting trends, and generating actionable strategies to optimize pricing, promotions, and customer satisfaction. The dashboard integrates data scraping, machine learning, and large language models (LLMs) to provide meaningful insights.

## Features
 **1.Data Collection**
- Competitor pricing, discounts, and reviews are scraped daily from e-commerce platforms (e.g., Amazon) using BeautifulSoup.
- The data is stored in two CSV files:
  - competitor_data.csv: Contains product pricing, discounts, and dates.
  - reviews.csv: Contains customer reviews for each product.

**2. Sentiment Analysis**
- Customer reviews are analyzed for sentiment using the Hugging Face Transformers library.
- Sentiment results are visualized in a bar chart using Plotly.

**3. Predictive Modeling**
- A Random Forest Regressor model predicts discounts based on product prices.
- An ARIMA model forecasts future discounts for the next five days.

**4. Strategy Recommendations**
- The dashboard leverages OpenAI's API to generate actionable strategies based on:
  - Competitor data (current and predicted trends).
  -Customer sentiment analysis.

**5. Slack Integration**
- Generated strategy recommendations are sent to a Slack channel for immediate access by stakeholders.

**6. Interactive Dashboard** 
- Built with Streamlit, the dashboard allows users to:
  - Select a product for analysis.
  - View competitor data, predicted discounts, and sentiment analysis.
  - Read detailed strategy recommendations.


