import streamlit as st
import pandas as pd
from analyzer import SentimentAnalyzer
from visualizer import plot_sentiment_distribution, generate_wordcloud

st.set_page_config(page_title="Sentiment Dashboard", layout="wide")
st.title("🎭 Sentiment Analysis Dashboard")

@st.cache_resource
def load_model():
    return SentimentAnalyzer()

analyzer = load_model()

tab1, tab2 = st.tabs(["Single Analysis", "Batch Processing"])

with tab1:
    text = st.text_area("Enter text to analyze:")
    if st.button("Analyze Text"):
        if text:
            result = analyzer.predict(text)
            st.success(f"Sentiment: {result['label']} (Score: {result['score']:.2f})")

with tab2:
    uploaded_file = st.file_uploader("Upload CSV (must have 'text' column)", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'text' in df.columns:
            if st.button("Process Batch"):
                results = analyzer.batch_predict(df['text'].tolist())
                df['sentiment'] = [r['label'] for r in results]
                df['score'] = [r['score'] for r in results]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_sentiment_distribution(df))
                with col2:
                    st.pyplot(generate_wordcloud(df['text'].tolist()))
                    
                st.dataframe(df)
        else:
            st.error("CSV must contain a 'text' column.")
