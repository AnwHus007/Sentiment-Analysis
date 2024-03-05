import streamlit as st
from transformers import pipeline
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
# Load the Transformers sentiment analysis model
# transformers_pipeline = pipeline("sentiment-analysis")
# Load the RoBERTa sentiment analysis model
roberta_classifier = pipeline("sentiment-analysis", model="roberta-base")
# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

st.title("Sentiment Analysis")

text_input = st.text_area("Enter your text here:")

if st.button("Analyze"):
    if text_input:
        # Analyze using Transformers
        trans_result = transformers_pipeline(text_input)[0]
        trans_label = trans_result['label']
        trans_score = {
            "label": trans_label,
            "score": round(trans_result['score'], 3)
        }
        # Analyze sentiment using VADER
        vader_scores = vader_analyzer.polarity_scores(text_input)
        
        # Analyze sentiment using RoBERTa
        roberta_result = roberta_classifier(text_input)[0]
        roberta_label = roberta_result['label']
        roberta_score = {
            "label": "NEGATIVE" if roberta_label == "LABEL_1" else "POSITVE",
            "score": round(roberta_result['score'], 3)
        }
        # Display results
        st.subheader("VADER Sentiment Analysis:")
        st.write(f"POSITVE : {vader_scores['pos']}")
        st.write(f"NEUTRAL : {vader_scores['neu']}")
        st.write(f"NEGATIVE : {vader_scores['neg']}")
        st.subheader("RoBERTa Sentiment Analysis:")
        st.write(f"Sentiment : {roberta_score['label']}")
        st.write(f"Score:{roberta_score['score']}")
        # st.subheader("Transformers Sentiment Analysis:")
        # st.write(f"Sentiment : {trans_score['label']}")
        # st.write(f"Score : {trans_score['score']}")
