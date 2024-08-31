import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from nltk.sentiment import SentimentIntensityAnalyzer
import yfinance as yf
import nltk
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import ta
import sqlite3
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
import base64

# Set page config at the very beginning
st.set_page_config(page_title="Advanced Market Research Assistant", page_icon="📊", layout="wide")

# Download NLTK data
nltk.download('vader_lexicon')

# Gemini API configuration
GEMINI_API_KEY = "YOUR_GOOGLE_GEMINI_API"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"

# Initialize database
conn = sqlite3.connect('market_research.db')
c = conn.cursor()

# Create tables if they don't exist
c.execute('''CREATE TABLE IF NOT EXISTS history
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              query TEXT,
              result TEXT,
              timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
conn.commit()

# Global variable to store analysis results
analysis_results = {}

def query_gemini(prompt):
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", headers=headers, json=data)
    return response.json()

def analyze_data(data, query):
    prompt = f"Analyze the following market data and answer the query: {query}\n\nData:\n{data}"
    response = query_gemini(prompt)
    result = response['candidates'][0]['content']['parts'][0]['text']
    
    # Save to history
    c.execute("INSERT INTO history (query, result) VALUES (?, ?)", (query, result))
    conn.commit()
    
    return result

def generate_insights(data):
    prompt = f"Generate insights and trends from the following market data:\n\n{data}"
    response = query_gemini(prompt)
    return response['candidates'][0]['content']['parts'][0]['text']

def perform_sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

def forecast_time_series(data, periods=30):
    ma = data.rolling(window=7).mean()
    last_ma = ma.iloc[-1]
    forecast = [last_ma] * periods
    forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=periods)
    return pd.Series(forecast, index=forecast_index)

def perform_kmeans_clustering(df, n_clusters=3):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    return clusters

def perform_pca(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    return pca_result

def get_stock_data(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date)
    return data

def create_network_graph(data):
    G = nx.Graph()
    for i, row in data.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Weight'])
    pos = nx.spring_layout(G)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text',
        marker=dict(showscale=True, colorscale='YlGnBu', reversescale=True, color=[], size=10,
                    colorbar=dict(thickness=15, title='Node Connections', xanchor='left', titleside='right'),
                    line_width=2))
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'{adjacencies[0]} # of connections: {len(adjacencies[1])}')
    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(title='Network Graph', titlefont_size=16, showlegend=False, hovermode='closest',
                                     margin=dict(b=20,l=5,r=5,t=40),
                                     annotations=[dict(text="", showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002)],
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig

def add_technical_indicators(data):
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    data['MACD'] = ta.trend.MACD(data['Close']).macd()
    data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
    return data

def export_to_pdf(content):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.drawString(100, height - 100, "Market Research Report")
    y_position = height - 120
    for section, data in content.items():
        c.drawString(100, y_position, section)
        y_position -= 20
        if isinstance(data, str):
            textobject = c.beginText(100, y_position)
            for line in data.split('\n'):
                textobject.textLine(line)
                y_position -= 15
            c.drawText(textobject)
        elif isinstance(data, plt.Figure):
            img_buffer = BytesIO()
            data.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            img = ImageReader(img_buffer)
            c.drawImage(img, 100, y_position - 200, width=400, height=200)
            y_position -= 220
        y_position -= 20
        
        if y_position < 100:
            c.showPage()
            y_position = height - 100
    
    c.save()
    buffer.seek(0)
    return buffer

st.title("Advanced Market Research Assistant")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Data Analysis", "Market Insights", "Competitor Analysis", "Trend Forecasting", "Stock Analysis", "Network Analysis", "History"])

if page == "Data Analysis":
    st.header("Data Analysis")
    
    uploaded_file = st.file_uploader("Upload your market data CSV", type="csv")

    if uploaded_file is not None:
        data = uploaded_file.getvalue().decode("utf-8")
        df = pd.read_csv(uploaded_file)
        st.write("Data preview:")
        st.write(df.head())
        
        st.subheader("Data Visualization")
        plot_type = st.selectbox("Select plot type", ["Line Chart", "Bar Chart", "Scatter Plot", "Heatmap", "3D Scatter", "Box Plot", "Violin Plot"])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        if plot_type == "Line Chart":
            df.plot(ax=ax)
        elif plot_type == "Bar Chart":
            df.plot(kind='bar', ax=ax)
        elif plot_type == "Scatter Plot":
            x_col = st.selectbox("Select X-axis", df.columns)
            y_col = st.selectbox("Select Y-axis", df.columns)
            df.plot.scatter(x=x_col, y=y_col, ax=ax)
        elif plot_type == "Heatmap":
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        elif plot_type == "3D Scatter":
            fig = px.scatter_3d(df, x=df.columns[0], y=df.columns[1], z=df.columns[2])
            st.plotly_chart(fig)
        elif plot_type == "Box Plot":
            df.boxplot(ax=ax)
        elif plot_type == "Violin Plot":
            df.plot.violin(ax=ax)
        
        st.pyplot(fig)
        analysis_results['Data Visualization'] = fig
        
        st.subheader("Data Insights")
        insights = generate_insights(data)
        st.write(insights)
        analysis_results['Data Insights'] = insights
        
        st.subheader("Custom Query Analysis")
        query = st.text_input("Enter your research question:")
        if query:
            analysis = analyze_data(data, query)
            st.write(analysis)
            analysis_results['Custom Query Analysis'] = analysis
        
        st.subheader("Advanced Analytics")
        if st.button("Perform K-means Clustering"):
            n_clusters = st.slider("Select number of clusters", 2, 10, 3)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            clusters = perform_kmeans_clustering(df[numeric_cols], n_clusters)
            df['Cluster'] = clusters
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(df[df.columns[0]], df[df.columns[1]], c=df['Cluster'], cmap='viridis')
            ax.set_xlabel(df.columns[0])
            ax.set_ylabel(df.columns[1])
            ax.set_title("K-means Clustering")
            st.pyplot(fig)
            analysis_results['K-means Clustering'] = fig
        
        if st.button("Perform PCA"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            pca_result = perform_pca(df[numeric_cols])
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(pca_result[:, 0], pca_result[:, 1])
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("PCA Result")
            st.pyplot(fig)
            analysis_results['PCA'] = fig
        
        if st.button("Train Random Forest"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            target_col = st.selectbox("Select target column", numeric_cols)
            feature_cols = [col for col in numeric_cols if col != target_col]
            rf, mse, r2 = train_random_forest(df[feature_cols], df[target_col])
            st.write(f"Mean Squared Error: {mse}")
            st.write(f"R-squared Score: {r2}")
            analysis_results['Random Forest'] = f"Mean Squared Error: {mse}\nR-squared Score: {r2}"

elif page == "Market Insights":
    st.header("Market Insights")
    
    st.subheader("Trend Analysis")
    trend_query = st.text_input("Enter a market trend to analyze:")
    if trend_query:
        trend_analysis = query_gemini(f"Analyze the current market trends related to {trend_query}")
        st.write(trend_analysis['candidates'][0]['content']['parts'][0]['text'])
        analysis_results['Trend Analysis'] = trend_analysis['candidates'][0]['content']['parts'][0]['text']
        
        st.subheader("Sentiment Analysis")
        sentiment = perform_sentiment_analysis(trend_analysis['candidates'][0]['content']['parts'][0]['text'])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(['Negative', 'Neutral', 'Positive'], [sentiment['neg'], sentiment['neu'], sentiment['pos']])
        ax.set_title("Sentiment Analysis")
        ax.set_ylabel("Score")
        st.pyplot(fig)
        analysis_results['Sentiment Analysis'] = fig
    
    st.subheader("Market Sizing")
    market = st.text_input("Enter a market to estimate its size:")
    if market:
        market_sizing = query_gemini(f"Estimate the market size for {market}")
        st.write(market_sizing['candidates'][0]['content']['parts'][0]['text'])
        analysis_results['Market Sizing'] = market_sizing['candidates'][0]['content']['parts'][0]['text']
    
    st.subheader("Consumer Behavior Analysis")
    behavior_query = st.text_input("Enter a consumer behavior topic:")
    if behavior_query:
        behavior_analysis = query_gemini(f"Analyze consumer behavior related to {behavior_query}")
        st.write(behavior_analysis['candidates'][0]['content']['parts'][0]['text'])
        analysis_results['Consumer Behavior Analysis'] = behavior_analysis['candidates'][0]['content']['parts'][0]['text']

elif page == "Competitor Analysis":
    st.header("Competitor Analysis")
    
    competitor = st.text_input("Enter a competitor name:")
    if competitor:
        st.subheader("Competitive Analysis")
        competitor_analysis = query_gemini(f"Provide a comprehensive competitive analysis for {competitor}")
        st.write(competitor_analysis['candidates'][0]['content']['parts'][0]['text'])
        analysis_results['Competitive Analysis'] = competitor_analysis['candidates'][0]['content']['parts'][0]['text']
        
        st.subheader("SWOT Analysis")
        swot_analysis = query_gemini(f"Perform a detailed SWOT analysis for {competitor}")
        st.write(swot_analysis['candidates'][0]['content']['parts'][0]['text'])
        analysis_results['SWOT Analysis'] = swot_analysis['candidates'][0]['content']['parts'][0]['text']
        
        st.subheader("Market Positioning")
        positioning_analysis = query_gemini(f"Analyze the market positioning of {competitor}")
        st.write(positioning_analysis['candidates'][0]['content']['parts'][0]['text'])
        analysis_results['Market Positioning'] = positioning_analysis['candidates'][0]['content']['parts'][0]['text']
    
    st.subheader("Competitive Landscape")
    industry = st.text_input("Enter an industry for competitive landscape analysis:")
    if industry:
        landscape_analysis = query_gemini(f"Provide a comprehensive competitive landscape analysis for the {industry} industry")
        st.write(landscape_analysis['candidates'][0]['content']['parts'][0]['text'])
        analysis_results['Competitive Landscape'] = landscape_analysis['candidates'][0]['content']['parts'][0]['text']

elif page == "Trend Forecasting":
    st.header("Trend Forecasting")
    
    uploaded_file = st.file_uploader("Upload your time series data CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data preview:")
        st.write(df.head())
        
        time_col = st.selectbox("Select time column", df.columns)
        value_col = st.selectbox("Select value column", df.columns)
        
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
        
        st.subheader("Time Series Plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.index, df[value_col])
        ax.set_title("Time Series Data")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        st.pyplot(fig)
        analysis_results['Time Series Plot'] = fig
        
        st.subheader("Trend Forecast")
        forecast_periods = st.slider("Select forecast periods", 1, 365, 30)
        forecast = forecast_time_series(df[value_col], periods=forecast_periods)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.index, df[value_col], label="Historical Data")
        ax.plot(forecast.index, forecast, label="Forecast")
        ax.set_title("Time Series Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        st.pyplot(fig)
        analysis_results['Forecast Plot'] = fig
        
        st.subheader("Forecast Interpretation")
        forecast_analysis = query_gemini(f"Interpret the following time series forecast for {value_col}:\n\n{forecast.to_string()}")
        st.write(forecast_analysis['candidates'][0]['content']['parts'][0]['text'])
        analysis_results['Forecast Interpretation'] = forecast_analysis['candidates'][0]['content']['parts'][0]['text']

elif page == "Stock Analysis":
    st.header("Stock Analysis")
    
    symbol = st.text_input("Enter stock symbol (e.g., AAPL for Apple):")
    start_date = st.date_input("Start date")
    end_date = st.date_input("End date")
    
    if symbol and start_date and end_date:
        stock_data = get_stock_data(symbol, start_date, end_date)
        
        st.subheader("Stock Price Chart")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(stock_data.index, stock_data['Close'])
        ax.set_title(f"{symbol} Stock Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        st.pyplot(fig)
        analysis_results['Stock Price Chart'] = fig
        
        st.subheader("Stock Price Forecast")
        forecast = forecast_time_series(stock_data['Close'], periods=30)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(stock_data.index, stock_data['Close'], label='Historical')
        ax.plot(forecast.index, forecast, label='Forecast')
        ax.set_title(f"{symbol} Stock Price Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)
        analysis_results['Stock Price Forecast'] = fig
        
        st.subheader("Technical Indicators")
        stock_data = add_technical_indicators(stock_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(stock_data.index, stock_data['Close'], label='Close')
        ax.plot(stock_data.index, stock_data['RSI'], label='RSI')
        ax.plot(stock_data.index, stock_data['MACD'], label='MACD')
        ax.set_title(f"{symbol} Technical Indicators")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        st.pyplot(fig)
        analysis_results['Technical Indicators'] = fig
        
        st.subheader("Stock Analysis")
        stock_analysis = query_gemini(f"Provide a comprehensive analysis of {symbol} stock based on recent market trends and company performance")
        st.write(stock_analysis['candidates'][0]['content']['parts'][0]['text'])
        analysis_results['Stock Analysis'] = stock_analysis['candidates'][0]['content']['parts'][0]['text']
        
        st.subheader("Volatility Analysis")
        stock_data['Returns'] = stock_data['Close'].pct_change()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(stock_data['Returns'], bins=50)
        ax.set_title(f"{symbol} Returns Distribution")
        ax.set_xlabel("Returns")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        analysis_results['Volatility Analysis'] = fig
        
        st.write(f"Annualized Volatility: {stock_data['Returns'].std() * np.sqrt(252):.2%}")

elif page == "Network Analysis":
    st.header("Network Analysis")
    
    uploaded_file = st.file_uploader("Upload your network data CSV (columns: Source, Target, Weight)", type="csv")
    if uploaded_file is not None:
        network_data = pd.read_csv(uploaded_file)
        st.write("Data preview:")
        st.write(network_data.head())
        
        st.subheader("Network Graph")
        network_fig = create_network_graph(network_data)
        st.plotly_chart(network_fig)
        analysis_results['Network Graph'] = network_fig
        
        st.subheader("Network Metrics")
        G = nx.from_pandas_edgelist(network_data, 'Source', 'Target', 'Weight')
        st.write(f"Number of nodes: {G.number_of_nodes()}")
        st.write(f"Number of edges: {G.number_of_edges()}")
        st.write(f"Average degree: {sum(dict(G.degree()).values()) / float(G.number_of_nodes()):.2f}")
        
        st.subheader("Centrality Measures")
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        
        centrality_df = pd.DataFrame({
            'Node': degree_centrality.keys(),
            'Degree Centrality': degree_centrality.values(),
            'Betweenness Centrality': betweenness_centrality.values(),
            'Closeness Centrality': closeness_centrality.values()
        })
        
        st.write(centrality_df)
        analysis_results['Centrality Measures'] = centrality_df.to_string()
        
        st.subheader("Network Analysis")
        network_analysis = query_gemini(f"Analyze the following network data and provide insights:\n\n{network_data.to_string()}")
        st.write(network_analysis['candidates'][0]['content']['parts'][0]['text'])
        analysis_results['Network Analysis'] = network_analysis['candidates'][0]['content']['parts'][0]['text']

elif page == "History":
    st.header("Analysis History")
    
    c.execute("SELECT * FROM history ORDER BY timestamp DESC")
    history = c.fetchall()
    
    for item in history:
        st.subheader(f"Query: {item[1]}")
        st.write(f"Result: {item[2]}")
        st.write(f"Timestamp: {item[3]}")
        st.write("---")

# Add export to PDF option
if st.button("Export to PDF"):
    pdf_buffer = export_to_pdf(analysis_results)
    st.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name="market_research_report.pdf",
        mime="application/pdf"
    )

st.sidebar.info("This Advanced Market Research Assistant uses AI and data analysis techniques to provide comprehensive insights.")
st.sidebar.warning("Note: AI-generated insights should be validated with domain expertise and additional research.")

# Close the database connection when the app is done
conn.close()
