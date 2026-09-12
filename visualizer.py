import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def plot_sentiment_distribution(df):
    fig = px.pie(df, names='sentiment', title='Sentiment Distribution',
                 color='sentiment', color_discrete_map={'POSITIVE':'green', 'NEGATIVE':'red'})
    return fig

def generate_wordcloud(texts):
    text = " ".join(texts)
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return fig
