import lime
import torch
import folium
import random
import pickle
import requests
import numpy as np
import pandas as pd
import wikipediaapi
import lime.lime_text
from PIL import Image
import streamlit as st
from io import BytesIO
from datetime import datetime
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.ticker as ticker
from matplotlib.colors import to_hex
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from googleapiclient.discovery import build
import streamlit.components.v1 as components
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertForSequenceClassification

BING_API_KEY = ''
BING_SEARCH_URL = 'https://api.bing.microsoft.com/v7.0/images/search'

# Set page configuration
st.set_page_config(layout="wide", page_title="Firm Reputation Classifier")

# Load the dataset
df = pd.read_csv('~/Documents/Projects/jobreview-absam/Source Codes/Source Code-Datasets/aggregatedFiltered_lemma_Dataset.csv')
reviews_df = pd.read_csv('~/Documents/Projects/jobreview-absam/Source Codes/Source Code-Datasets/glassdoor_reviews_cleaned_copy_v3.csv')

# Load the model and tokenizer
path = '~/Documents/Projects/jobreview-absam/Source Codes/Source Code-Models/'

# BERT
with open(path + 'bert_model.pkl', 'rb') as f:
    bert_model = pickle.load(f)

with open(path + 'bert_tokenizer.pkl', 'rb') as f:
    bert_tokenizer = pickle.load(f)

# Traditional models
traditional_models = {
    'SVM': {
        'SVM Oversampling': (pickle.load(open(path + 'svm_model.pkl', 'rb')), pickle.load(open(path + 'svm_vectorizer.pkl', 'rb'))),
        'SVM Undersampling': (pickle.load(open(path + 'svm_model_ps.pkl', 'rb')), pickle.load(open(path + 'svm_vectorizer_ps.pkl', 'rb')))
    },
    'Random Forest': {
        'RF Oversampling': (pickle.load(open(path + 'rf_model.pkl', 'rb')), pickle.load(open(path + 'rf_vectorizer.pkl', 'rb'))),
        'RF Undersampling': (pickle.load(open(path + 'rf_model_ps.pkl', 'rb')), pickle.load(open(path + 'rf_vectorizer_ps.pkl', 'rb')))
    },
    'Logistic Regression': {
        'LR Oversampling': (pickle.load(open(path + 'lr_model.pkl', 'rb')), pickle.load(open(path + 'lr_vectorizer.pkl', 'rb'))),
        'LR Undersampling': (pickle.load(open(path + 'lr_model_ps.pkl', 'rb')), pickle.load(open(path + 'lr_vectorizer_ps.pkl', 'rb')))
    },
    'Naive Bayes': {
        'NB Oversampling': (pickle.load(open(path + 'nb_model.pkl', 'rb')), pickle.load(open(path + 'nb_vectorizer.pkl', 'rb'))),
        'NB Undersampling': (pickle.load(open(path + 'nb_model_ps.pkl', 'rb')), pickle.load(open(path + 'nb_vectorizer_ps.pkl', 'rb')))
    }
}

# Function to predict sentiment
def predict_sentiment(texts, model, tokenizer):
    if isinstance(model, BertForSequenceClassification):
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1)
        return predicted_labels.cpu().numpy(), probabilities.cpu().numpy()
    else:
        texts = tokenizer.transform(texts)
        predicted_labels = model.predict(texts)
        probabilities = model.predict_proba(texts)
        return predicted_labels, probabilities
    
# Function to get Wikipedia summary
def get_wikipedia_summary(company_name):
    wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent=f'Firm Reputation Classifier/1.0 (https://www.google.com/search?q={company_name})'
    )
    page = wiki_wiki.page(company_name)
    if page.exists():
        # Split the summary into paragraphs and return the first one
        paragraphs = page.summary.split('\n')
        return paragraphs[0] if paragraphs else "No summary available."
    else:
        return "No Wikipedia page found for this company."

# Function to get Company Image
def get_bing_image_url(query, count=10):
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    params = {"q": query, "count": count}
    response = requests.get(BING_SEARCH_URL, headers=headers, params=params)
    search_results = response.json()
    if 'value' in search_results:
        return [img['contentUrl'] for img in search_results['value']]
    else:
        return None

# Function to fetch and resize image
def fetch_and_resize_image(image_url, width=400, height=300):
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        return None

# Function to Color Bar Graph
def color_map(weight):
    """Generate a color based on the weight value."""
    if weight > 0:
        return to_hex([1, 0.5 + weight / 2, 0.1])  # Dynamic orange to yellow
    else:
        # Gradient from dark blue to light blue
        return to_hex([0, 0.5 + weight / 2, 1])  # Dynamic dark blue to light blue

# Function to Plot Weight for Bar Graph
def plot_weight_bar(weight, color):
    """Generate a bar plot for the weight."""
    fig, ax = plt.subplots(figsize=(3, 0.5))
    ax.barh([''], [weight], color=color)
    ax.set_xlim([-1, 1])
    ax.axis('off')
    return fig

# Streamlit interface
st.title("Firm Reputation Classifier")

# Dropdown for firm selection
firm_names = df['firm'].unique()
selected_firms = st.multiselect("Select the firm names:", firm_names)

# Dropdown for model selection
model_type = st.selectbox("Select the type of model:", ['BERT'] + list(traditional_models.keys()))

if model_type == 'BERT':
    model, tokenizer = bert_model, bert_tokenizer
    model_selection_placeholder = None
else:
    model_files = list(traditional_models[model_type].keys())
    selected_model_file = st.selectbox("Select the file for the traditional model:", model_files)
    model, tokenizer = traditional_models[model_type][selected_model_file]

# Use sidebar for vertical tabs
tab = st.sidebar.radio("Select Tab:", ["Company Analysis", "Sentiment Trends & Comparative Analysis", "Review Display",
                                        "Job Titles & Employment Status", "Recommendations, Outlook & CEO Approval", 
                                        "Correlation Analysis", "Quick Recommendation System", "Keyword Co-Occurrence","Keyword-Based Recommendation"])

# Create a dictionary to store results for each firm
if 'results' not in st.session_state:
    st.session_state.results = {}

# Fetch and update results
if selected_firms:
    for selected_firm in selected_firms:
        if selected_firm not in st.session_state.results:
            # Fetch company images
            image_urls = get_bing_image_url(selected_firm)
            images = []
            if image_urls:
                with st.spinner(f"Fetching image for {selected_firm}..."):
                    for image_url in image_urls:
                        img = fetch_and_resize_image(image_url, width=400, height=300)
                        if img:
                            images.append(img)
                            break
            st.session_state.results[selected_firm] = {
                'images': images,
                'summary': get_wikipedia_summary(selected_firm),
                'reviews': df[df['firm'] == selected_firm]['labeled_keywords'].tolist(),
                'sentiment': None,
                'lime_explanation': None
            }

    # Date range filter
    available_dates = pd.to_datetime(reviews_df['date_review']).dt.date.unique()
    available_dates = sorted(available_dates)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.selectbox("Select start date:", available_dates, index=0)
    with col2:
        end_date = st.selectbox("Select end date:", available_dates, index=len(available_dates) - 1)

    if start_date and end_date:
        filtered_reviews_df = reviews_df[(pd.to_datetime(reviews_df['date_review']).dt.date >= start_date) & (pd.to_datetime(reviews_df['date_review']).dt.date <= end_date)]

    if st.button("Predict"):
        for selected_firm in selected_firms:
            firm_reviews = st.session_state.results[selected_firm]['reviews']
            
            if len(firm_reviews) == 0:
                st.write(f"No reviews found for {selected_firm}.")
            else:
                predicted_labels, probabilities = predict_sentiment(firm_reviews, model, tokenizer)
                avg_sentiment = np.mean(predicted_labels)
                sentiment = 'good' if avg_sentiment > 0.5 else 'bad'
                st.session_state.results[selected_firm]['sentiment'] = sentiment

                # LIME explanation
                explainer = lime.lime_text.LimeTextExplainer(class_names=['bad', 'good'])
                exp = explainer.explain_instance(firm_reviews[0], lambda x: predict_sentiment(x, model, tokenizer)[1], num_features=15)
                st.session_state.results[selected_firm]['lime_explanation'] = exp

    if tab == "Company Analysis":
        for selected_firm in selected_firms:
            with st.expander(f"{selected_firm} Analysis", expanded=True):
                # Display Wikipedia summary
                st.subheader("Company Summary")
                st.write(st.session_state.results[selected_firm]['summary'])
                
                # Display images
                if st.session_state.results[selected_firm]['images']:
                    st.subheader("Company Images")
                    for img in st.session_state.results[selected_firm]['images']:
                        st.image(img)
                else:
                    st.write("No images found for this company.")

                # Display Sentiment
                if st.session_state.results[selected_firm]['sentiment']:
                    st.subheader("Predicted Sentiment")
                    st.write(f"The predicted sentiment for {selected_firm} is: {st.session_state.results[selected_firm]['sentiment']}")

                # Display LIME Explanation
                if st.session_state.results[selected_firm]['lime_explanation']:
                    st.subheader("LIME Explanation")
                    exp = st.session_state.results[selected_firm]['lime_explanation']
                    
                    # Display textual explanation
                    components.html(exp.as_html())
                    
                    # Extract and visualize feature weights
                    explanation = exp.as_list()
                    words, weights = zip(*explanation)
                    
                     # Calculate min and max weights
                    min_weight = min(weights)
                    max_weight = max(weights)

                    ## Plot the weights
                    fig, ax = plt.subplots()
                    bars = ax.barh(words, weights, color=[color_map(w) for w in weights])
                    ax.set_xlabel('Weight')
                    ax.set_title('LIME Explanation')
                    plt.gca().invert_yaxis()

                    col1,col2 =st.columns(2)
                    # Display min and max weights
                    with col1:
                        for bar in bars:
                            width = bar.get_width()
                            ax.text(width, bar.get_y() + bar.get_height() / 3, f'{width:.4f}', va='center', ha='right' if width < 0 else 'left')
                        
                        # Display min and max weights
                        plt.text(0, len(words)-1.5, f'Min: {min_weight:.4f}', color='black')
                        plt.text(0, len(words)-1.0, f'Max: {max_weight:.4f}', color='black')
                        
                        st.pyplot(fig)

                    with col2:
                        ## Create a dataframe for the table
                        data = {
                            'Feature': words,
                            'Weight': weights
                        }
                        df = pd.DataFrame(data)
                        
                        # Display the table
                        st.write(df[['Feature', 'Weight']])
                    
                    # Display Keyword Frequency
                    st.subheader("Keyword Frequency")
                    all_keywords = ' '.join(st.session_state.results[selected_firm]['reviews']).split()
                    keyword_counts = Counter(all_keywords)
                    top_keywords = keyword_counts.most_common(10)
                    keyword_df = pd.DataFrame(top_keywords, columns=['Keyword', 'Frequency'])
                    st.table(keyword_df)
        
        # Firm Rankings
        st.subheader("Firm Rankings")

        # Calculate average ratings for each firm
        avg_ratings = reviews_df.groupby('firm')['overall_rating'].mean().reset_index()
        avg_ratings.columns = ['Firm', 'Average Rating']

        # Add a ranking column
        avg_ratings = avg_ratings.sort_values(by='Average Rating', ascending=False).reset_index(drop=True)
        avg_ratings['Ranking'] = avg_ratings.index + 1
        avg_ratings = avg_ratings[['Firm', 'Average Rating']]

        # Highlight selected firms
        def highlight_selected_firms(df, selected_firms):
            # Create a DataFrame with the same shape as `df` filled with empty strings
            styles = pd.DataFrame('', index=df.index, columns=df.columns)
            # Apply the background color to the rows of selected firms
            for i, firm in enumerate(df['Firm']):
                if firm in selected_firms:
                    styles.iloc[i] = 'background-color: yellow'
            return styles

        # Apply the highlighting function
        highlighted_avg_ratings = avg_ratings.style.apply(highlight_selected_firms, selected_firms=selected_firms, axis=None)

        # Display the dataframe with highlighting
        st.dataframe(highlighted_avg_ratings, use_container_width=True)

        st.markdown("""
        <div style="text-align: center; color: rgba(0, 0, 0, 0.5);">
            Analyze individual companies by examining their reviews, extracting insights from sentiment scores, and identifying common themes or issues. 
                    This tab provides a deep dive into each company's performance and user feedback.
        </div>
        """, unsafe_allow_html=True)

    elif tab == "Review Display":
        for selected_firm in selected_firms:
            with st.expander(f"{selected_firm} Reviews", expanded=True):
                st.subheader("Review Rating Statistics")
                firm_reviews = reviews_df[reviews_df['firm'] == selected_firm]
                if not firm_reviews.empty:
                    total_reviews = len(firm_reviews)
                    rating_distribution = firm_reviews['overall_rating'].value_counts().sort_index()
                    st.write(f"Total number of reviews: {total_reviews}")
                    rating_df = pd.DataFrame({'Rating': rating_distribution.index, 'Number of Reviews': rating_distribution.values})
                    st.write("Number of reviews per rating:")
                    st.table(rating_df)
                    
                    st.write("Reviews by Rating:")
                    # Create a container to hold the reviews
                    review_container = st.container()

                    # Loop through each rating and display the reviews
                    for rating in range(1, 6):
                        with review_container:
                            st.subheader(f"Rating: {rating}")
                            rating_reviews = firm_reviews[firm_reviews['overall_rating'] == rating]
                            if not rating_reviews.empty:
                                st.write(f"Number of reviews with rating {rating}: {len(rating_reviews)}")
                                st.dataframe(rating_reviews[['headline', 'overall_rating', 'pros', 'cons']].head(5))
                            else:
                                st.write(f"No reviews found for rating {rating}.")
        
        st.markdown("""
        <div style="text-align: center; color: rgba(0, 0, 0, 0.5);">
            View and interact with raw reviews submitted by users. 
                    This tab allows you to browse through the actual comments and feedback provided for detailed examination.
        </div>
        """, unsafe_allow_html=True)

    elif tab == "Sentiment Trends & Comparative Analysis":
        if selected_firms:
            for selected_firm in selected_firms:
                st.subheader(f"Sentiment Trends for {selected_firm}")
                sentiment_trends_df = filtered_reviews_df[filtered_reviews_df['firm'] == selected_firm].groupby(pd.to_datetime(filtered_reviews_df['date_review']).dt.date)['overall_rating'].mean().reset_index()
                sentiment_trends_df.columns = ['Date', 'Average Rating']
                st.line_chart(sentiment_trends_df.set_index('Date'))
                
            feature_columns = ['senior_mgmt', 'comp_benefits', 'work_life_balance', 'culture_values', 'diversity_inclusion', 'career_opp']
            comparative_df = df[df['firm'].isin(selected_firms)]
            comparative_df = comparative_df.groupby('firm')[feature_columns].mean().reset_index()
            comparative_df['Overall Score'] = comparative_df[feature_columns].mean(axis=1)
            comparative_df = comparative_df.sort_values(by='Overall Score', ascending=False)
            st.subheader("Comparative Analysis")
            st.write("Comparative Analysis of Firms:")
            st.dataframe(comparative_df, use_container_width=True)
        
        st.markdown("""
        <div style="text-align: center; color: rgba(0, 0, 0, 0.5);">
            Explore trends in sentiment over time and compare different companies based on their sentiment scores. 
                    This tab helps in understanding how sentiment evolves and how companies stack up against each other.
        </div>
        """, unsafe_allow_html=True)
    
    elif tab == "Job Titles & Employment Status":
        st.subheader("Job Titles and Employment Status")
        for selected_firm in selected_firms:
            with st.expander(f"{selected_firm} Job Titles and Employment Status", expanded=True):
                firm_reviews = reviews_df[reviews_df['firm'] == selected_firm]
                if not firm_reviews.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        # Display job title frequency
                        job_title_counts = firm_reviews['job_title'].value_counts()
                        job_title_df = pd.DataFrame({'Job Title': job_title_counts.index, 'Frequency': job_title_counts.values})
                        st.write("Job Title Frequency:")
                        st.table(job_title_df.head(10))  # Show top 10 job titles
                    
                    with col2:
                        # Plot job title distribution
                        st.write("Job Title Distribution:")
                        fig, ax = plt.subplots(figsize=(5, 3))
                        job_title_counts.head(10).plot(kind='bar', ax=ax)
                        ax.set_xlabel("Job Title")
                        ax.set_ylabel("Frequency")
                        ax.set_title(f"Top 10 Job Titles in {selected_firm}")
                        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                        
                        # Set y-axis limits (example limits, adjust based on your data)
                        ax.set_ylim(0, job_title_counts.max() + 2)  # Add some padding
                        st.pyplot(fig)

                    col3, col4 = st.columns(2)
                    with col3:
                        # Display employment status frequency
                        employment_status_counts = firm_reviews['current'].value_counts()
                        employment_status_df = pd.DataFrame({'Employment Status': employment_status_counts.index, 
                                                             'Frequency': employment_status_counts.values})
                        st.write("Employment Status Frequency:")
                        st.table(employment_status_df)
                                        
                    with col4:
                        # Plot employment status distribution
                        st.write("Employment Status Distribution:")
                        fig, ax = plt.subplots(figsize=(5, 3))
                        employment_status_counts.plot(kind='bar', ax=ax)
                        ax.set_xlabel("Employment Status")
                        ax.set_ylabel("Frequency")
                        ax.set_title(f"Employment Status in {selected_firm}")
                        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

                        # Set y-axis limits (example limits, adjust based on your data)
                        ax.set_ylim(0, employment_status_counts.max() + 2)  # Add some padding
                        
                        st.pyplot(fig)
                else:
                    st.write(f"No reviews found for {selected_firm}.")

        st.markdown("""
        <div style="text-align: center; color: rgba(0, 0, 0, 0.5);">
            Analyze reviews based on job titles and employment status. 
                    This tab helps identify if there are significant differences in sentiment or feedback 
                    based on employee roles or their status within the company.
        </div>
        """, unsafe_allow_html=True)

    elif tab == "Recommendations, Outlook & CEO Approval":
        for selected_firm in selected_firms:
            with st.expander(f"Displaying recommendations, outlook, and CEO approval for {selected_firm}...", expanded=True):
                with st.spinner(f"Loading data for {selected_firm}..."):
                    firm_reviews_df = filtered_reviews_df[filtered_reviews_df['firm'] == selected_firm]

                    if not firm_reviews_df.empty:
                        recommendations = firm_reviews_df['recommend'].replace({0: 'no', 1: 'yes'}).value_counts()
                        outlook = firm_reviews_df['outlook'].replace({0: 'no', 1: 'yes'}).value_counts()
                        ceo_approval = firm_reviews_df['ceo_approv'].replace({0: 'no', 1: 'yes'}).value_counts()

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write("### Recommendations")
                            st.write(recommendations.to_frame().reset_index().rename(columns={'index': 'Recommend', 'recommend': 'Count'}))
                        with col2:
                            st.write("### Outlook")
                            st.write(outlook.to_frame().reset_index().rename(columns={'index': 'Outlook', 'outlook': 'Count'}))
                        with col3:
                            st.write("### CEO Approval")
                            st.write(ceo_approval.to_frame().reset_index().rename(columns={'index': 'CEO Approval', 'ceo_approv': 'Count'}))

                        fig, ax = plt.subplots(1, 3, figsize=(15, 4))

                        ax[0].bar(recommendations.index, recommendations.values)
                        ax[0].set_title("Recommendations")
                        ax[0].set_xlabel("Recommend")
                        ax[0].set_ylabel("Count")

                        ax[1].bar(outlook.index, outlook.values)
                        ax[1].set_title("Outlook")
                        ax[1].set_xlabel("Outlook")
                        ax[1].set_ylabel("Count")

                        ax[2].bar(ceo_approval.index, ceo_approval.values)
                        ax[2].set_title("CEO Approval")
                        ax[2].set_xlabel("CEO Approval")
                        ax[2].set_ylabel("Count")

                        st.pyplot(fig)
                    else:
                        st.write(f"No data available for {selected_firm} within the selected date range.")

        st.markdown("""
        <div style="text-align: center; color: rgba(0, 0, 0, 0.5);">
            Generate actionable recommendations based on review analysis, explore overall outlook on the company, and evaluate CEO approval ratings. 
                    This tab offers strategic insights and improvement areas.
        </div>
        """, unsafe_allow_html=True)

    elif tab == "Correlation Analysis":
        if selected_firms:
            st.write("Correlation Analysis")
            
            # Example of encoding features and calculating correlations
            feature_columns = ['work_life_balance', 'culture_values', 'diversity_inclusion', 'career_opp', 'comp_benefits', 'senior_mgmt']
            feature_df = df[df['firm'].isin(selected_firms)][feature_columns]
            
            # Replace NaN values with 0 for correlation calculation
            feature_df.fillna(0, inplace=True)
            
            # Calculate correlations
            correlations = feature_df.corr()
            
            st.write("Correlation Matrix:")
            st.dataframe(correlations, use_container_width=True)

            col1, col2,col3= st.columns(3)
            with col2:
                # Plot Correlation Heatmap
                fig, ax = plt.subplots(figsize=(10, 5))
                cax = ax.matshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
                fig.colorbar(cax)
                ax.set_xticks(range(len(correlations.columns)))
                ax.set_yticks(range(len(correlations.columns)))
                ax.set_xticklabels(correlations.columns, rotation=45, ha='left')
                ax.set_yticklabels(correlations.columns)
                ax.set_title("Correlation Heatmap")
                st.pyplot(fig)

        st.markdown("""
        <div style="text-align: center; color: rgba(0, 0, 0, 0.5);">
            Investigate correlations between various factors in the reviews, such as sentiment and keyword frequency. 
                    This tab helps in identifying relationships and patterns that could impact company performance.
        </div>
        """, unsafe_allow_html=True)

    # Recommendation System Tab
    if tab == "Quick Recommendation System":
        st.title("Recommendation System")

        # User-based Recommendations
        st.subheader("User Recommendations")

        # Allow users to select criteria for recommendations
        recommendation_criteria = st.selectbox("Select criteria for recommendations:", ["Job Titles", "Sentiment", "Overall Ratings"])
        
        if recommendation_criteria == "Job Titles":
            st.write("Based on job titles:")
            for selected_firm in selected_firms:
                firm_reviews = reviews_df[reviews_df['firm'] == selected_firm]
                if not firm_reviews.empty:
                    job_title_counts = firm_reviews['job_title'].value_counts()
                    top_job_titles = job_title_counts.head(5)
                    st.write(f"Top job titles for {selected_firm}:")
                    st.write(top_job_titles)
                else:
                    st.write(f"No reviews found for {selected_firm}.")

        elif recommendation_criteria == "Sentiment":
            st.write("Based on sentiment analysis:")
            for selected_firm in selected_firms:
                firm_reviews = st.session_state.results[selected_firm]['reviews']
                if len(firm_reviews) == 0:
                    st.write(f"No reviews found for {selected_firm}.")
                else:
                    predicted_labels, _ = predict_sentiment(firm_reviews, model, tokenizer)
                    avg_sentiment = np.mean(predicted_labels)
                    sentiment = 'good' if avg_sentiment > 0.5 else 'bad'
                    st.write(f"The average sentiment for {selected_firm} is: {sentiment}")

        elif recommendation_criteria == "Overall Ratings":
            st.write("Based on overall ratings:")
            for selected_firm in selected_firms:
                firm_reviews = reviews_df[reviews_df['firm'] == selected_firm]
                if not firm_reviews.empty:
                    avg_rating = firm_reviews['overall_rating'].mean()
                    st.write(f"The average rating for {selected_firm} is: {avg_rating:.2f}")

        # HR-based Recommendations
        st.subheader("HR Recommendations")

        # Provide actionable insights for HR
        hr_insights = st.selectbox("Select HR Insight:", ["Improve Workplace Culture", "Enhance Benefits", "Increase Work-Life Balance"])

        if hr_insights == "Improve Workplace Culture":
            st.write("Recommendations for improving workplace culture:")
            for selected_firm in selected_firms:
                firm_reviews = reviews_df[reviews_df['firm'] == selected_firm]
                if not firm_reviews.empty:
                    culture_feedback = firm_reviews['culture_values'].value_counts()
                    st.write(f"Culture feedback for {selected_firm}:")
                    st.write(culture_feedback)

        elif hr_insights == "Enhance Benefits":
            st.write("Recommendations for enhancing benefits:")
            for selected_firm in selected_firms:
                firm_reviews = reviews_df[reviews_df['firm'] == selected_firm]
                if not firm_reviews.empty:
                    benefits_feedback = firm_reviews['comp_benefits'].value_counts()
                    st.write(f"Benefits feedback for {selected_firm}:")
                    st.write(benefits_feedback)

        elif hr_insights == "Increase Work-Life Balance":
            st.write("Recommendations for improving work-life balance:")
            for selected_firm in selected_firms:
                firm_reviews = reviews_df[reviews_df['firm'] == selected_firm]
                if not firm_reviews.empty:
                    work_life_balance_feedback = firm_reviews['work_life_balance'].value_counts()
                    st.write(f"Work-life balance feedback for {selected_firm}:")
                    st.write(work_life_balance_feedback)

        st.markdown("""
        <div style="text-align: center; color: rgba(0, 0, 0, 0.5);">
            Investigate correlations between various factors in the reviews, such as sentiment and keyword frequency. 
                    This tab helps in identifying relationships and patterns that could impact company performance.
        </div>
        """, unsafe_allow_html=True)
        
    if tab == "Keyword-Based Recommendation":
        st.subheader("Keyword-Based Recommendations")
        if selected_firms:
            for selected_firm in selected_firms:
                with st.expander(f"Recommendations for {selected_firm} based on Keywords", expanded=True):
                    firm_reviews = df[df['firm'] == selected_firm]
                    
                    if not firm_reviews.empty:
                        # Concatenate all labeled_keywords
                        all_keywords = ' '.join(firm_reviews['labeled_keywords'].astype(str))
                        
                        # Count the frequency of each keyword
                        keyword_counts = Counter(all_keywords.split())

                        # Calculate the average frequency
                        avg_frequency = np.mean(list(keyword_counts.values()))
                        
                        # Define the scaling factor
                        scaling_factor = 1.5
                        
                        # Calculate the frequency threshold
                        frequency_threshold = scaling_factor * avg_frequency
                        
                        # Sort by frequency
                        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
                        
                        # Convert to DataFrame
                        keyword_df = pd.DataFrame(sorted_keywords, columns=['Keyword', 'Frequency'])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            # Display the top keywords
                            st.write("Top Keywords:")
                            st.table(keyword_df.head(10))
                        
                        with col2:
                            if selected_firm in st.session_state.results and 'lime_explanation' in st.session_state.results[selected_firm]:
                                exp = st.session_state.results[selected_firm]['lime_explanation']
                                explanation = exp.as_list()
                                
                                # Directly display top 15 LIME weights
                                st.write("Top 15 LIME Weights:")
                                lime_df = pd.DataFrame(explanation, columns=['Keyword', 'Weight'])
                                st.table(lime_df)
                                
                        # Generate recommendations based on keyword frequency and sentiment
                        st.write("Recommendations Based on Keywords:")
                        
                        recommendations = []
                                                
                        for keyword, count in sorted_keywords:
                            if count > frequency_threshold:
                                weight = lime_df[lime_df['Keyword'] == keyword]['Weight'].values[0] if keyword in lime_df['Keyword'].values else 0
                                recommendations.append(f"Keyword '{keyword}' appears frequently ({count} times) with a LIME weight of {weight:.4f}. Consider focusing on this aspect.")
                        
                        # Sentiment-based recommendations
                        sentiment_df = df[['labeled_keywords', 'predicted_sentiment']].copy()
                        sentiment_df['labeled_keywords'] = sentiment_df['labeled_keywords'].apply(lambda x: x if isinstance(x, list) else x.split())
                        sentiment_df = sentiment_df.explode('labeled_keywords')
                        sentiment_counts = sentiment_df.groupby('labeled_keywords')['predicted_sentiment'].mean()
                                                                
                        # Display recommendations
                        if recommendations:
                            for rec in recommendations:
                                st.write(f"- {rec}")
                        else:
                            st.write("No specific recommendations based on keyword analysis.")
                    else:
                        st.write(f"No keywords found for {selected_firm}.")

        st.markdown("""
        <div style="text-align: center; color: rgba(0, 0, 0, 0.5);">
            Generate recommendations based on the analysis of frequently occurring keywords in reviews. 
                    This tab uses keyword frequency and sentiment analysis to suggest focus areas for improvement.
        </div>
        """, unsafe_allow_html=True)

    if tab == "Keyword Co-Occurrence":
        for selected_firm in selected_firms:
            with st.expander(f"{selected_firm} Keyword Co-Occurrence Analysis", expanded=True):
                firm_reviews = df[df['firm'] == selected_firm]['labeled_keywords'].tolist()
                
                # Extract keywords
                all_keywords = set()
                for review in firm_reviews:
                    all_keywords.update(review.split())
                
                # Dropdown for keyword selection
                selected_keywords = st.multiselect("Select keywords for co-occurrence analysis:", list(all_keywords))
                
                if selected_keywords:
                    st.write("Co-Occurrence Recommendations:")
                    
                    # Create co-occurrence pairs
                    co_occurrence_df = pd.DataFrame({'keywords': firm_reviews})
                    co_occurrence_df['keywords'] = co_occurrence_df['keywords'].apply(lambda x: x.split())
                    keyword_pairs = []
                    
                    for keywords in co_occurrence_df['keywords']:
                        for i in range(len(keywords)):
                            for j in range(i + 1, len(keywords)):
                                if keywords[i] in selected_keywords and keywords[j] in selected_keywords:
                                    keyword_pairs.append((keywords[i], keywords[j]))
                    
                    pair_counts = Counter(keyword_pairs)

                    # Calculate average frequency
                    average_frequency = sum(pair_counts.values()) / len(pair_counts)

                    # Set frequency threshold
                    multiplier = 1.5  # Choose a multiplier based on your needs
                    frequency_threshold = average_frequency * multiplier

                    recommendations = []
                    for (kw1, kw2), count in pair_counts.items():
                        if count > frequency_threshold:
                            recommendations.append(f"Keywords '{kw1}' and '{kw2}' frequently appear together by '{count}' times. Try other keywords...\n")
                        else:
                            recommendations.append(f"Keywords '{kw1}' and '{kw2}' do not frequently appear together by '{count}' times. Consider other keywords...\n")
                    
                    st.write("\n".join(recommendations))

        st.markdown("""
        <div style="text-align: center; color: rgba(0, 0, 0, 0.5);">
            Analyze how keywords co-occur in reviews to uncover underlying themes or issues. 
                    This tab helps in understanding which keywords frequently appear together and their significance.
        </div>
        """, unsafe_allow_html=True)