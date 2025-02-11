# -*- coding: utf-8 -*-
"""full.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1EfWhmvwxo4tbkRJDGWRFCj0zOsXLabLq

**Webscrapping**

*packages* *installed*
"""

!pip install bs4

!pip install requests

!pip install pandas

"""**Actual work week 1** - *webscrapping of amazon data*"""

from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import re
import datetime
import random

# Function to extract Product Title
def get_title(soup):

    try:
        # Outer Tag Object
        title = soup.find("span", attrs={"class":"a-size-medium a-color-base a-text-normal"})

        # Inner NavigatableString Object
        title_value = title.text

        # Title as a string value
        title_string = title_value.strip()

    except AttributeError:
        title_string = ""

    return title_string


# Function to extract price
def get_price(soup):
    try:
        price = soup.find("span", attrs={"class": "a-price-whole"})
        return price.text.strip() if price else ""
    except AttributeError:
        return ""


# Function to extract mrp
def get_mrp(soup):
    try:
        mrp = soup.find("span", attrs={"class": "a-price a-text-price"})
        return mrp.text.strip() if mrp else ""
    except AttributeError:
        return ""

# Function to get product discount (without "off")
def get_discount(item):
    try:
        discount = item.find("span", string=re.compile(r"off"))
        if discount:
            # Clean the discount string: remove brackets and the word "off"
            return discount.text.strip()
    except AttributeError:
       return ""


# Function to extract Product Rating
def get_rating(soup):

    try:
        rating = soup.find("i", attrs={'class':'a-declarative'}).string.strip()

    except AttributeError:
        try:
            rating = soup.find("span", attrs={'class':'a-icon-alt'}).string.strip()
        except:
            rating = ""

    return rating

# Function to extract Number of User Reviews
def get_review_count(soup):
    try:
        review_count = soup.find("span", attrs={'id':'acrCustomerReviewText'}).string.strip()

    except AttributeError:
        review_count = ""

    return review_count

#Function to extract review statements
def get_review(soup):
    """
    Extract the product review text from the given soup object.
    """
    try:
        review_tag = soup.find("span", class_="a-size-base review-text")
        if review_tag:
            # Extracting the inner text from the review tag
            return review_tag.text.strip()
    except AttributeError:
        return ""
    return ""

# Function to extract Availability Status
def get_availability(soup):
    try:
        available = soup.find("span", attrs={'class':'a-size-base a-color-price'})
        available = available.find("span").string.strip()

    except AttributeError:
        available = "Not Available"

    return available

# Function to fetch page content
def fetch_page_content(url, headers):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.content, "html.parser")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None

def get_title(soup):
    try:
        # Find the product title
        title = soup.find("span", attrs={"class": "a-size-medium a-color-base a-text-normal"})
        return title.text.strip() if title else ""
    except AttributeError:
        return ""

def get_price(soup):
    try:
        # Find the whole price
        price_whole = soup.find("span", attrs={"class": "a-price-whole"})
        # Find the fractional price, if present
        price_fraction = soup.find("span", attrs={"class": "a-price-fraction"})

        # Combine whole and fractional parts
        if price_whole:
            return f"{price_whole.text.strip()}{price_fraction.text.strip() if price_fraction else ''}"
        return ""
    except AttributeError:
        return ""
def get_mrp(soup):
    try:
        mrp = soup.find("span", attrs={"class": "a-price a-text-price"})
        return mrp.text.strip() if mrp else ""
    except AttributeError:
        return ""
def get_discount(soup):
    try:
        # Look for any mention of "off"
        discount = soup.find("span", string=re.compile(r"off"))
        return discount.text.strip() if discount else ""
    except AttributeError:
        return ""
def get_rating(soup):
    try:
        # Ratings might be stored in a span with 'a-icon-alt' class
        rating = soup.find("span", attrs={"class": "a-icon-alt"})
        return rating.text.strip() if rating else ""
    except AttributeError:
        return ""
def get_review_count(soup):
    try:
        review_count = soup.find("span", attrs={"id": "acrCustomerReviewText"})
        return review_count.text.strip() if review_count else ""
    except AttributeError:
        return ""
def get_review(soup):
    try:
        review_tag = soup.find("span", class_="review-text-content")
        return review_tag.text.strip() if review_tag else ""
    except AttributeError:
        return ""
def get_availability(soup):
    try:
        # Find availability status
        available = soup.find("span", attrs={"class": "a-size-base a-color-success"})
        return available.text.strip() if available else "Not Available"
    except AttributeError:
        return "Not Available"
def fetch_page_content(url, headers):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.content, "html.parser")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None

if __name__ == '__main__':

    HEADERS = ({'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36','Accept-Language': 'en-US, en;q=0.5'})

    # The webpage URL
    URL = 'https://www.amazon.in/s?k=headphones&crid=2ZT1OS98TQZGP&sprefix=headphones%2Caps%2C243&ref=nb_sb_noss_2'
    # HTTP Request
    webpage = requests.get(URL, headers=HEADERS) # Use HEADERS instead of headers

    # Fetch page content
    soup = fetch_page_content(URL, HEADERS) # Changed 'headers' to 'HEADERS'

    # Extract all product links
    if soup:
        product_links = []

        # Search for all product containers (div elements containing product info)
        product_containers = soup.find_all("div", attrs={"data-asin": True})

        for product in product_containers:
            # Get the anchor tag (link) within each product container
            link_tag = product.find("a", attrs={"class": "a-link-normal"})
            if link_tag:
                product_url = "https://www.amazon.in" + link_tag.get("href")
                product_links.append(product_url)

        # Print the extracted links
        if product_links:
            print("Extracted Product Links:")
            for link in product_links:
                print(link)
        else:
            print("No links found.")
    else:
        print("Error fetching or parsing the page.")

    # Store the links
    # Changed 'links' to 'product_links' to iterate over the extracted product URLs
    # links_list = ["https://www.amazon.in" + link.get('href') for link in product_links if link.get('href')]  # This line is incorrect

    # Since product_links already has the full URLs, simply assign it:
    links_list = product_links

    # Remove the unnecessary loop (it also uses undefined 'links'):
    # for link in links:
    #        links_list.append(link.get('href'))

    # Remove this loop as well (it also uses undefined 'links'):
    # for link in links:
    #        links_list.append(link.get('href'))

    # Loop for extracting links from Tag Objects - Removed the original loop as it also uses 'links' with no definition
    #for link in links:
    #        links_list.append(link.get('href')) #Remove this line too

    # The problematic loop has been removed since 'links' is not defined and the links
    # are already stored in 'links_list'

    d = {"title":[], "price":[], "mrp":[], "discount":[], "rating":[], "reviews":[], "review_statements":[], "availability":[], "date":[]}

    for link in links_list:
        product_soup = fetch_page_content(link, HEADERS)
        if product_soup:
            d["title"].append(get_title(product_soup))
            d["price"].append(get_price(product_soup))
            d["mrp"].append(get_mrp(product_soup))
            d["discount"].append(get_discount(product_soup))
            d["rating"].append(get_rating(product_soup))
            d["reviews"].append(get_review_count(product_soup))
            d["review_statements"].append(get_review(product_soup))
            d["availability"].append(get_availability(product_soup))


            current_date = datetime.datetime.now().strftime("%Y-%m-%d")  # Format as YYYY-MM-DD
            d["date"].append(current_date)

        else:
            # Append placeholders if the product page cannot be fetched
            d["title"].append("")
            d["price"].append("")
            d["mrp"].append("")
            d["discount"].append("")
            d["rating"].append("")
            d["reviews"].append("")
            d["review_statements"].append("")
            d["availability"].append("")


            current_date = datetime.datetime.now().strftime("%Y-%m-%d")  # Format as YYYY-MM-DD
            d["date"].append(current_date)

    # Ensure all lists in `d` have the same length
    max_len = max(len(d[key]) for key in d)
    for key in d:
        while len(d[key]) < max_len:
            d[key].append("")


    amazon_df = pd.DataFrame.from_dict(d)
    amazon_df['title'] = amazon_df['title'].replace('', np.nan)
    amazon_df = amazon_df.dropna(subset=['title'])
    amazon_df.to_csv("amazon_data.csv", header=True, index=False)

from bs4 import BeautifulSoup
import requests
import re
import datetime
import pandas as pd
import numpy as np


# Function to extract Product Title
def get_title(soup):
    try:
        title = soup.find("span", attrs={"class": "a-size-medium a-color-base a-text-normal"})
        return title.text.strip() if title else ""
    except AttributeError:
        return ""


# Function to extract price
def get_price(soup):
    try:
        price = soup.find("span", attrs={"class": "a-price-whole"})
        return price.text.strip() if price else ""
    except AttributeError:
        return ""


# Function to extract MRP
def get_mrp(soup):
    try:
        mrp = soup.find("span", attrs={"class": "a-price a-text-price"})
        if mrp:
            mrp_span = mrp.find("span", attrs={"class": "a-offscreen"})
            return mrp_span.text.strip() if mrp_span else ""
    except AttributeError:
        return ""
    return ""


# Function to get product discount (e.g., "20% off")
def get_discount(soup):
    try:
        discount = soup.find("span", string=re.compile(r"off"))
        return discount.text.strip() if discount else ""
    except AttributeError:
        return ""


# Function to extract Product Rating
def get_rating(soup):
    try:
        rating = soup.find("span", attrs={"class": "a-icon-alt"})
        return rating.text.strip() if rating else ""
    except AttributeError:
        return ""


# Function to extract Number of User Reviews
def get_review_count(soup):
    try:
        review_count = soup.find("span", attrs={"class": "a-size-base"}).text
        return review_count.strip() if review_count else ""
    except AttributeError:
        return ""


# Function to extract review statements
def get_review(soup):
    try:
        review = soup.find("span", attrs={"class": "a-size-base review-text"})
        return review.text.strip() if review else ""
    except AttributeError:
        return ""


# Function to extract Availability Status
def get_availability(soup):
    try:
        available = soup.find("div", attrs={"id": "availability"})
        if available:
            return available.text.strip()
    except AttributeError:
        return "Not Available"
    return "Not Available"


# Function to fetch page content
def fetch_page_content(url, headers):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.content, "html.parser")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None
if __name__ == "__main__":
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Accept-Language": "en-US, en;q=0.5",
    }

    URL = "https://www.amazon.in/s?k=headphones&crid=2ZT1OS98TQZGP&sprefix=headphones%2Caps%2C243&ref=nb_sb_noss_2"

    # Fetch main page content
    soup = fetch_page_content(URL, HEADERS)

    if not soup:
        print("Failed to fetch main page content.")
        exit()

    product_links = []
    product_containers = soup.find_all("div", attrs={"data-asin": True})

    # Extract product links
    for product in product_containers:
        link_tag = product.find("a", attrs={"class": "a-link-normal"})
        if link_tag and link_tag.get("href"):
            product_url = "https://www.amazon.in" + link_tag.get("href")
            product_links.append(product_url)

    if not product_links:
        print("No product links found.")
        exit()

    print(f"Extracted {len(product_links)} product links.")

    # Initialize data dictionary
    data = {
        "title": [],
        "price": [],
        "mrp": [],
        "discount": [],
        "rating": [],
        "reviews": [],
        "review_statements": [],
        "availability": [],
        "date": [],
    }

    # Loop through product links to extract data
    for link in product_links:
        print(f"Fetching data for product: {link}")
        product_soup = fetch_page_content(link, HEADERS)
        if product_soup:
            data["title"].append(get_title(product_soup))
            data["price"].append(get_price(product_soup))
            data["mrp"].append(get_mrp(product_soup))
            data["discount"].append(get_discount(product_soup))
            data["rating"].append(get_rating(product_soup))
            data["reviews"].append(get_review_count(product_soup))
            data["review_statements"].append(get_review(product_soup))
            data["availability"].append(get_availability(product_soup))

            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            data["date"].append(current_date)
        else:
            # Append empty values if fetching fails
            for key in data.keys():
                data[key].append("")

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame.from_dict(data)
    df["title"] = df["title"].replace("", np.nan)
    df = df.dropna(subset=["title"])  # Drop rows without a title
    print(f"Number of products fetched: {len(df)}")
    df.to_csv("amazon_products.csv", index=False)
    print("Data saved to amazon_products.csv")

amazon_df = pd.read_csv("amazon_data.csv")

price_data = amazon_df[['date','title','price']]

review_data = amazon_df[['date','title','reviews']]

review_statements = amazon_df[['date','title','review_statements']]

price_data.to_csv("price_data.csv", index=False, encoding='utf-8')
review_data.to_csv("review_data.csv", index=False, encoding='utf-8')
review_statements.to_csv("review_statements.csv", index=False, encoding='utf-8')

"""**Sentiment Analysis**"""

import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
from IPython.display import display  # Import the display function

# Specify the model explicitly
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def get_sentiment(text):
    """Analyze sentiment of the given text using Hugging Face sentiment analysis"""
    try:
        result = sentiment_analyzer(text)
        sentiment = result[0]["label"]
        score = result[0]["score"]
        return sentiment, score
    except Exception as e:
        print(f"Error analyzing sentiment for text: {text}. Error: {e}")
        return "Neutral", 0.0

def preprocess_review_data(file_path):
    """Load and preprocess review data"""
    data = pd.read_csv(file_path)
    data["date"] = pd.to_datetime(data["date"])
    return data

def main(file_path, output_path):
    review_data = preprocess_review_data(file_path)

    # Extract sentiment and scores
    sentiments = review_data["review_statements"].apply(get_sentiment)
    review_data["Sentiment"] = sentiments.apply(lambda x: x[0])
    review_data["SentimentScore"] = sentiments.apply(lambda x: x[1])

    # Save the processed data to a CSV file
    review_data.to_csv(output_path, index=False)

    print(f"Sentiment analysis results saved to {output_path}")

    # Display the DataFrame in the notebook output section
    display(review_data)

if __name__ == "__main__":
    input_file_path = "review_statements.csv"  # Input file with reviews
    output_file_path = "review_data_with_sentiments.csv"  # Output file
    main(input_file_path, output_file_path)

"""**Sentiment score distribution** - *trend lines*"""

import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def get_sentiment(text):
    """Analyze sentiment of the given text using Hugging Face sentiment analysis."""
    try:
        result = sentiment_analyzer(text)
        sentiment = result[0]["label"]
        score = result[0]["score"]
        return sentiment, score
    except Exception as e:
        print(f"Error analyzing sentiment for text: {text}. Error: {e}")
        return "Neutral", 0.0

def preprocess_review_data(file_path):
    """Load and preprocess review data."""
    data = pd.read_csv(file_path)
    data["date"] = pd.to_datetime(data["date"])
    return data

def generate_summary(data):
    """Generate summary statistics for sentiment analysis."""
    sentiment_counts = data["Sentiment"].value_counts()
    sentiment_percentage = (sentiment_counts / len(data)) * 100
    summary = pd.DataFrame({
        "Sentiment": sentiment_counts.index,
        "Count": sentiment_counts.values,
        "Percentage": sentiment_percentage.values
    })
    return summary

def visualize_sentiments(data):
    """Create visualizations for sentiment analysis results."""
    # Sentiment distribution bar chart
    plt.figure(figsize=(8, 5))
    sns.countplot(data=data, x="Sentiment", palette="viridis")
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.show()

    # Sentiment score distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=data, x="SentimentScore", hue="Sentiment", kde=True, palette="viridis")
    plt.title("Sentiment Score Distribution")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.show()

def main(input_file, output_file):
    # Load and preprocess data
    review_data = preprocess_review_data(input_file)

    # Extract sentiment and scores
    sentiments = review_data["review_statements"].apply(get_sentiment)
    review_data["Sentiment"] = sentiments.apply(lambda x: x[0])
    review_data["SentimentScore"] = sentiments.apply(lambda x: x[1])

    # Generate summary statistics
    summary = generate_summary(review_data)
    print("\nSentiment Summary:\n", summary)

    # Save the processed data to a CSV file
    review_data.to_csv(output_file, index=False)
    print(f"\nSentiment analysis results saved to {output_file}")

    # Visualize results
    visualize_sentiments(review_data)

if __name__ == "__main__":
    input_file = "review_statements.csv"  # Input file with reviews
    output_file = "review_data_with_sentiments.csv"  # Output file
    main(input_file, output_file)
