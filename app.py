import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import requests

# Configure page
st.set_page_config(layout="wide", page_title="Multi-Client Portfolio Risk Analysis", page_icon="📊")
st.title("📊 Multi-Client Portfolio Risk Analysis")
st.caption("Evaluating portfolio risk based on news sentiment and corporate events")

# News API integration
@st.cache_data(ttl=3600)  # Refresh every hour
def fetch_news_data():
    try:
        url = "https://service.upstox.com/content/open/v5/news/sub-category/news/list//market-news/stocks?page=1&pageSize=500"
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        data = response.json()
        return data.get('data', [])  # Return empty list if 'data' key doesn't exist
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching news data: {e}")
        return []
    except ValueError as e:
        st.error(f"Error parsing news data: {e}")
        return []

# Process news data to extract relevant information
def process_news_data(news_data):
    processed_news = []
    if not news_data:
        return processed_news
    
    for item in news_data:
        try:
            # Skip if item doesn't have required fields
            if not all(key in item for key in ['title', 'content', 'publishedAt']):
                continue
                
            # Extract stock symbols from the news title/content
            symbols_in_news = []
            title = item.get('title', '').lower()
            content = item.get('content', '').lower()
            
            for stock in stocks:
                stock_lower = stock.lower()
                if stock_lower in title or stock_lower in content:
                    symbols_in_news.append(stock)
            
            if symbols_in_news:
                processed_news.append({
                    'title': item['title'],
                    'content': item['content'],
                    'published_at': item['publishedAt'],
                    'symbols': symbols_in_news,
                    'sentiment': random.uniform(-1, 1)  # Placeholder for actual sentiment analysis
                })
        except Exception as e:
            st.warning(f"Error processing news item: {e}")
            continue
            
    return processed_news

# Define the list of stocks
stocks = [
    "BAJAJHFL", "BAJFINANCE", "M&M", "BHARATFORG", "AXISBANK", "NESTLEIND", "TECHM", "DMART", 
    "KPIGREEN", "RCOM", "TATACOMM", "JKCEMENT", "HCLTECH", "IDEA", "RAYMOND", 
    "RAYMONDLSL", "TORNTPHARM", "APLLTD", "WAAREEENER", "JYOTICNC", "KTKBANK", 
    "JBCHEPHARM", "HAL"
]

# Define sectors for each stock
sector_mapping = {
    'BAJAJHFL': 'Consumer Goods',
    'BAJFINANCE': 'Automotive',
    'M&M': 'Automotive',
    'BHARATFORG': 'Industrial',
    'AXISBANK': 'Financial',
    'NESTLEIND': 'Consumer Goods',
    'TECHM': 'Technology',
    'DMART': 'Retail',
    'KPIGREEN': 'Energy',
    'RCOM': 'Telecom',
    'TATACOMM': 'Telecom',
    'JKCEMENT': 'Construction',
    'HCLTECH': 'Technology',
    'IDEA': 'Telecom',
    'RAYMOND': 'Textiles',
    'RAYMONDLSL': 'Real Estate',
    'TORNTPHARM': 'Pharmaceuticals',
    'APLLTD': 'Pharmaceuticals',
    'WAAREEENER': 'Energy',
    'JYOTICNC': 'Industrial',
    'KTKBANK': 'Financial',
    'JBCHEPHARM': 'Pharmaceuticals',
    'HAL': 'Aerospace & Defense'
}

# Define event types for each stock
event_mapping = {
    'BAJAJHFL': 'agreement',
    'BAJFINANCE': 'financial',
    'M&M': 'financial',
    'BHARATFORG': 'financial',
    'AXISBANK': 'financial',
    'NESTLEIND': 'financial',
    'TECHM': 'financial',
    'DMART': 'investment',
    'KPIGREEN': 'regulatory',
    'RCOM': 'regulatory',
    'TATACOMM': 'financial',
    'JKCEMENT': 'financial',
    'HCLTECH': 'partnership',
    'IDEA': 'expansion',
    'RAYMOND': 'launch',
    'RAYMONDLSL': 'launch',
    'TORNTPHARM': 'agreement',
    'APLLTD': 'regulatory',
    'WAAREEENER': 'agreement',
    'JYOTICNC': 'agreement',
    'KTKBANK': 'leadership',
    'JBCHEPHARM': 'acquisition',
    'HAL': 'financial'
}

# Event risk mapping
event_risk = {
    'financial': 'Low',
    'regulatory': 'High',
    'agreement': 'Medium',
    'partnership': 'Low',
    'expansion': 'Medium',
    'launch': 'High',
    'acquisition': 'High',
    'leadership': 'Medium',
    'investment': 'Medium'
}

# Generate dummy client data
def generate_client_portfolio(client_id):
    portfolio = []
    # Select a random subset of stocks for this client
    num_stocks = random.randint(8, len(stocks))
    client_stocks = random.sample(stocks, num_stocks)
    
    for symbol in client_stocks:
        shares = random.randint(10, 1000)
        avg_cost = random.uniform(50, 5000)
        current_price = avg_cost * random.uniform(0.8, 1.5)  # Simulate price movement
        sector = sector_mapping[symbol]
        event = event_mapping[symbol]
        risk = event_risk[event]
        
        # Generate sentiment (-1 to 1) with sector bias
        if sector in ['Technology', 'Financial']:
            sentiment = random.uniform(0.1, 0.9)
        elif sector in ['Telecom', 'Energy']:
            sentiment = random.uniform(-0.7, 0.3)
        else:
            sentiment = random.uniform(-0.5, 0.5)
        
        portfolio.append({
            "Symbol": symbol,
            "Sector": sector,
            "Event": event,
            "Risk": risk,
            "Sentiment": sentiment,
            "Shares": shares,
            "Avg Cost": avg_cost,
            "Current Price": current_price,
            "Investment": shares * avg_cost,
            "Current Value": shares * current_price
        })
    
    portfolio_df = pd.DataFrame(portfolio)
    portfolio_df["P&L"] = portfolio_df["Current Value"] - portfolio_df["Investment"]
    portfolio_df["P&L %"] = (portfolio_df["P&L"] / portfolio_df["Investment"]) * 100
    portfolio_df["Weight"] = (portfolio_df["Current Value"] / portfolio_df["Current Value"].sum()) * 100
    
    return portfolio_df

# Generate multiple clients
clients = {
    "Client A (Conservative)": generate_client_portfolio(1),
    "Client B (Growth)": generate_client_portfolio(2),
    "Client C (Aggressive)": generate_client_portfolio(3),
    "Client D (Income)": generate_client_portfolio(4),
    "Client E (Balanced)": generate_client_portfolio(5)
}

# Calculate client metrics
client_metrics = []
for client_name, portfolio in clients.items():
    total_value = portfolio["Current Value"].sum()
    total_investment = portfolio["Investment"].sum()
    total_pl = total_value - total_investment
    avg_sentiment = (portfolio["Sentiment"] * portfolio["Weight"]).sum() / 100
    
    # Calculate risk score (High=3, Medium=2, Low=1)
    risk_score = (portfolio["Risk"].map({"High": 3, "Medium": 2, "Low": 1}) * portfolio["Weight"]).sum() / 100
    
    client_metrics.append({
        "Client": client_name,
        "Portfolio Value": total_value,
        "Investment": total_investment,
        "P&L": total_pl,
        "Avg Sentiment": avg_sentiment,
        "Risk Score": risk_score,
        "High Risk Exposure": portfolio[portfolio["Risk"] == "High"]["Weight"].sum()
    })

client_metrics_df = pd.DataFrame(client_metrics)

# Risk Analysis
st.header("Client Portfolio Risk Analysis")

# Client risk overview
st.subheader("Client Risk Exposure")
col1, col2 = st.columns(2)

with col1:
    fig = px.bar(client_metrics_df, x="Client", y="Risk Score", 
                 color="Risk Score", color_continuous_scale="Viridis",
                 title="Overall Portfolio Risk Score")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(client_metrics_df, x="Avg Sentiment", y="High Risk Exposure",
                     size="Portfolio Value", color="Client",
                     hover_name="Client", 
                     title="Sentiment vs High Risk Exposure",
                     labels={"Avg Sentiment": "Average News Sentiment", 
                             "High Risk Exposure": "High Risk Allocation (%)"})
    st.plotly_chart(fig, use_container_width=True)

# Identify client at most risk
most_at_risk = client_metrics_df.loc[client_metrics_df["Risk Score"].idxmax()]
st.warning(f"🚨 Client at Highest Risk: **{most_at_risk['Client']}** "
           f"(Risk Score: {most_at_risk['Risk Score']:.1f}, "
           f"High Risk Exposure: {most_at_risk['High Risk Exposure']:.1f}%)")

# Client selector
selected_client = st.selectbox("Select Client for Detailed Analysis", list(clients.keys()))
portfolio = clients[selected_client]

# Display client portfolio
st.subheader(f"{selected_client} Portfolio Details")

# Portfolio metrics
total_value = portfolio["Current Value"].sum()
total_investment = portfolio["Investment"].sum()
total_pl = total_value - total_investment
total_pl_pct = (total_pl / total_investment) * 100
avg_sentiment = (portfolio["Sentiment"] * portfolio["Weight"]).sum() / 100
high_risk_exposure = portfolio[portfolio["Risk"] == "High"]["Weight"].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Portfolio Value", f"₹{total_value:,.0f}")
col2.metric("Profit & Loss", f"₹{total_pl:,.0f}", f"{total_pl_pct:.2f}%")
col3.metric("Avg News Sentiment", f"{avg_sentiment:.2f}", 
            "Positive" if avg_sentiment > 0 else "Negative")
col4.metric("High Risk Exposure", f"{high_risk_exposure:.1f}%")

# Portfolio analysis tabs
tab1, tab2, tab3, tab4 = st.tabs(["Holdings Analysis", "Risk Exposure", "Recommendations", "What-If Scenario"])

with tab1:
    st.subheader("Portfolio Composition")
    col1, col2 = st.columns(2)
    
    with col1:
        # Sector exposure
        sector_exposure = portfolio.groupby("Sector")["Current Value"].sum().reset_index()
        sector_exposure["Weight"] = (sector_exposure["Current Value"] / total_value) * 100
        fig = px.pie(sector_exposure, values="Current Value", names="Sector", 
                     title="Sector Allocation")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Event type distribution
        event_exposure = portfolio.groupby("Event")["Current Value"].sum().reset_index()
        event_exposure["Weight"] = (event_exposure["Current Value"] / total_value) * 100
        fig = px.bar(event_exposure, x="Event", y="Weight", color="Event",
                     title="Event Type Exposure")
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed holdings
    st.subheader("Detailed Holdings")
    portfolio_display = portfolio.copy()
    portfolio_display["Current Price"] = portfolio_display["Current Price"].apply(lambda x: f"₹{x:,.2f}")
    portfolio_display["Investment"] = portfolio_display["Investment"].apply(lambda x: f"₹{x:,.0f}")
    portfolio_display["Current Value"] = portfolio_display["Current Value"].apply(lambda x: f"₹{x:,.0f}")
    portfolio_display["P&L"] = portfolio_display["P&L"].apply(lambda x: f"₹{x:,.0f}")
    portfolio_display["P&L %"] = portfolio_display["P&L %"].apply(lambda x: f"{x:.2f}%")
    portfolio_display["Weight"] = portfolio_display["Weight"].apply(lambda x: f"{x:.1f}%")
    portfolio_display["Sentiment"] = portfolio_display["Sentiment"].apply(lambda x: f"{x:.2f}")
    
    # Add recommendation based on sentiment and risk
    def generate_recommendation(row):
        if row["Risk"] == "High" and row["Sentiment"] < 0:
            return "STRONG SELL"
        elif row["Risk"] == "High" and row["Sentiment"] < 0.3:
            return "SELL"
        elif row["Sentiment"] > 0.5 and row["Risk"] == "Low":
            return "STRONG BUY"
        elif row["Sentiment"] > 0.3 and row["Risk"] in ["Low", "Medium"]:
            return "BUY"
        elif row["Sentiment"] < -0.3:
            return "REDUCE"
        else:
            return "HOLD"
    
    portfolio_display["Recommendation"] = portfolio.apply(generate_recommendation, axis=1)
    
    # Color coding for recommendations
    def color_recommendation(val):
        if "BUY" in val:
            return 'color: green; font-weight: bold'
        elif "SELL" in val:
            return 'color: red; font-weight: bold'
        elif "REDUCE" in val:
            return 'color: orange; font-weight: bold'
        return ''
    
    st.dataframe(
        portfolio_display[["Symbol", "Sector", "Event", "Risk", "Sentiment", 
                           "Weight", "P&L %", "Recommendation"]].rename(columns={
                               "Symbol": "Symbol",
                               "Sector": "Sector",
                               "Event": "Recent Event",
                               "Risk": "Risk Level",
                               "Sentiment": "News Sentiment",
                               "Weight": "Portfolio Weight",
                               "P&L %": "Return %",
                               "Recommendation": "Recommendation"
                           }).style.applymap(color_recommendation, subset=["Recommendation"]),
        height=400,
        use_container_width=True
    )

with tab2:
    st.subheader("Risk Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution
        risk_exposure = portfolio.groupby("Risk")["Current Value"].sum().reset_index()
        risk_exposure["Weight"] = (risk_exposure["Current Value"] / total_value) * 100
        fig = px.pie(risk_exposure, values="Current Value", names="Risk", 
                     title="Risk Level Distribution",
                     color="Risk", 
                     color_discrete_map={"High": "#ef5350", "Medium": "#ffca28", "Low": "#66bb6a"})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk vs Return
        fig = px.scatter(portfolio, x="P&L %", y="Risk", 
                         color="Sector", size="Current Value",
                         hover_name="Symbol", 
                         title="Risk vs Return Analysis",
                         labels={"P&L %": "Return %", "Risk": "Risk Level"})
        st.plotly_chart(fig, use_container_width=True)
    
    # High risk stocks
    st.subheader("High Risk Stocks")
    high_risk = portfolio[portfolio["Risk"] == "High"]
    if not high_risk.empty:
        fig = go.Figure()
        for _, row in high_risk.iterrows():
            fig.add_trace(go.Bar(
                x=[row["Symbol"]],
                y=[row["Weight"]],
                name=row["Symbol"],
                text=[f"Sentiment: {row['Sentiment']:.2f}<br>Event: {row['Event']}"],
                hovertemplate="%{x}<br>Weight: %{y:.1f}%<br>%{text}"
            ))
        
        fig.update_layout(barmode="stack", title="High Risk Exposure by Stock",
                          yaxis_title="Portfolio Weight (%)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No high-risk stocks in this portfolio")

with tab3:
    st.subheader("Portfolio Recommendations")
    
    # Overall portfolio recommendation
    if high_risk_exposure > 30:
        st.error("⚠️ **High Risk Alert:** This portfolio has significant exposure to high-risk stocks. Consider reducing exposure to regulatory and launch event stocks.")
    elif avg_sentiment < -0.1:
        st.warning("⚠️ **Negative Sentiment Alert:** Overall portfolio sentiment is negative. Review holdings with poor sentiment.")
    else:
        st.success("✅ **Portfolio Status:** Well-balanced with moderate risk exposure")
    
    # Generate specific recommendations
    recommendations = []
    for _, row in portfolio.iterrows():
        if row["Risk"] == "High" and row["Sentiment"] < 0:
            rec = {
                "Symbol": row["Symbol"],
                "Action": "Reduce",
                "Amount": f"Sell {min(100, int(row['Shares'] * 0.5))} shares",
                "Reason": f"High risk ({row['Risk']}) with negative sentiment ({row['Sentiment']:.2f})"
            }
            recommendations.append(rec)
        elif row["Sentiment"] > 0.5 and row["Risk"] == "Low":
            rec = {
                "Symbol": row["Symbol"],
                "Action": "Increase",
                "Amount": f"Add {int(row['Shares'] * 0.2)} shares",
                "Reason": f"Strong positive sentiment ({row['Sentiment']:.2f}) with low risk"
            }
            recommendations.append(rec)
        elif row["Sentiment"] < -0.3:
            rec = {
                "Symbol": row["Symbol"],
                "Action": "Reduce",
                "Amount": f"Sell {int(row['Shares'] * 0.3)} shares",
                "Reason": f"Strong negative sentiment ({row['Sentiment']:.2f})"
            }
            recommendations.append(rec)
    
    if recommendations:
        st.subheader("Specific Recommendations")
        rec_df = pd.DataFrame(recommendations)
        st.dataframe(rec_df, height=200, use_container_width=True)
        
        # Visualize recommendations
        fig = px.bar(rec_df, x="Symbol", y="Action", color="Action",
                     color_discrete_map={"Increase": "#4caf50", "Reduce": "#f44336"},
                     title="Recommended Actions",
                     hover_data=["Reason"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No specific recommendations at this time. Portfolio appears well-positioned.")
    
    # Sector rotation advice
    st.subheader("Sector Rotation Advice")
    
    # Find underperforming sectors
    sector_performance = portfolio.groupby("Sector")["P&L %"].mean().reset_index()
    underperforming = sector_performance[sector_performance["P&L %"] < 0]
    
    if not underperforming.empty:
        st.warning(f"Consider reducing exposure to underperforming sectors: {', '.join(underperforming['Sector'])}")
    
    # Find strong sectors
    strong_sectors = sector_performance[sector_performance["P&L %"] > 10]
    if not strong_sectors.empty:
        st.success(f"Consider increasing exposure to strong performing sectors: {', '.join(strong_sectors['Sector'])}")
    
    # Visualize sector performance
    fig = px.bar(sector_performance, x="Sector", y="P&L %", 
                 color="P&L %", title="Sector Performance",
                 color_continuous_scale="Viridis")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("What-If Scenario Based on Latest News")
    
    # Fetch and process news data
    news_data = fetch_news_data()
    if news_data:
        processed_news = process_news_data(news_data)
        
        # Filter news relevant to this portfolio
        portfolio_symbols = portfolio['Symbol'].unique()
        relevant_news = [news for news in processed_news if any(symbol in news['symbols'] for symbol in portfolio_symbols)]
        
        if relevant_news:
            st.success(f"Found {len(relevant_news)} recent news items affecting this portfolio")
            
            # Display news headlines
            st.subheader("Recent News Affecting Portfolio")
            for news in relevant_news[:5]:  # Show top 5 news items
                with st.expander(f"{news['title']} (Published: {news['published_at']})"):
                    st.write(news['content'][:500] + "...")  # Show first 500 chars
                    
                    # Show affected stocks
                    affected = [s for s in news['symbols'] if s in portfolio_symbols]
                    st.write(f"**Affected stocks in portfolio:** {', '.join(affected)}")
                    st.write(f"**News sentiment:** {news['sentiment']:.2f}")
            
            # Create what-if scenario
            st.subheader("Create What-If Scenario")
            
            # Select news item to simulate
            news_options = [f"{n['title']} ({n['published_at']})" for n in relevant_news]
            selected_news = st.selectbox("Select news item to simulate impact", news_options)
            selected_news_data = relevant_news[news_options.index(selected_news)]
            
            # Get affected stocks in portfolio
            affected_stocks = [s for s in selected_news_data['symbols'] if s in portfolio_symbols]
            
            # Impact severity
            impact = st.slider("Estimated impact severity", 0.1, 2.0, 1.0, 0.1,
                             help="How strongly will this news affect stock prices? 1.0 = average impact")
            
            # Simulate impact
            if st.button("Simulate Impact on Portfolio"):
                simulated_portfolio = portfolio.copy()
                
                # Adjust prices based on news sentiment and impact
                for symbol in affected_stocks:
                    sentiment_factor = selected_news_data['sentiment'] * impact
                    current_row = simulated_portfolio[simulated_portfolio['Symbol'] == symbol].iloc[0]
                    
                    # Calculate new price (simple simulation)
                    new_price = current_row['Current Price'] * (1 + (sentiment_factor * 0.1))
                    simulated_portfolio.loc[simulated_portfolio['Symbol'] == symbol, 'Current Price'] = new_price
                
                # Recalculate portfolio values
                simulated_portfolio['Current Value'] = simulated_portfolio['Shares'] * simulated_portfolio['Current Price']
                simulated_portfolio['P&L'] = simulated_portfolio['Current Value'] - simulated_portfolio['Investment']
                simulated_portfolio['P&L %'] = (simulated_portfolio['P&L'] / simulated_portfolio['Investment']) * 100
                
                # Calculate new totals
                new_total_value = simulated_portfolio['Current Value'].sum()
                value_change = new_total_value - total_value
                value_change_pct = (value_change / total_value) * 100
                
                # Display results
                st.subheader("Simulation Results")
                col1, col2 = st.columns(2)
                col1.metric("Original Portfolio Value", f"₹{total_value:,.0f}")
                col2.metric("Simulated Portfolio Value", f"₹{new_total_value:,.0f}", 
                           f"{value_change:,.0f} ({value_change_pct:.2f}%)")
                
                # Show affected stocks
                st.subheader("Most Impacted Stocks")
                affected = simulated_portfolio[simulated_portfolio['Symbol'].isin(affected_stocks)]
                affected = affected.sort_values('P&L %', ascending=False)
                
                fig = px.bar(affected, x='Symbol', y='P&L %', 
                            title='Projected Impact on Holdings',
                            color='P&L %',
                            color_continuous_scale='RdYlGn')
                st.plotly_chart(fig, use_container_width=True)
                
                # Show before/after comparison
                comparison = pd.merge(
                    portfolio[['Symbol', 'Current Value', 'P&L %']],
                    simulated_portfolio[['Symbol', 'Current Value', 'P&L %']],
                    on='Symbol',
                    suffixes=(' Before', ' After')
                )
                
                # Calculate changes
                comparison['Value Change'] = comparison['Current Value After'] - comparison['Current Value Before']
                comparison['P&L Change'] = comparison['P&L % After'] - comparison['P&L % Before']
                
                st.dataframe(
                    comparison.sort_values('Value Change', ascending=False).style.format({
                        'Current Value Before': '₹{:,.0f}',
                        'Current Value After': '₹{:,.0f}',
                        'Value Change': '₹{:,.0f}',
                        'P&L % Before': '{:.2f}%',
                        'P&L % After': '{:.2f}%',
                        'P&L Change': '{:.2f}%'
                    }),
                    use_container_width=True
                )
        else:
            st.info("No recent news found that affects this portfolio's holdings")
    else:
        st.error("Could not fetch news data. Please try again later.")

# News-based risk timeline
st.header("Event Risk Timeline")
st.info("Recent corporate events impacting portfolio holdings")

# Generate dummy events
events = []
for symbol in portfolio["Symbol"].sample(5):
    event_date = datetime.now() - timedelta(days=random.randint(1, 30))
    events.append({
        "Stock": symbol,
        "Event": event_mapping[symbol],
        "Date": event_date,
        "Risk": event_risk[event_mapping[symbol]],
        "Sentiment": random.uniform(-0.8, 0.8),
        "Impact": random.choice(["High", "Medium", "Low"])
    })

events_df = pd.DataFrame(events)

# Plot timeline
fig = px.timeline(events_df, x_start="Date", x_end=events_df["Date"] + timedelta(hours=12), 
                 y="Stock", color="Risk",
                 title="Recent Corporate Events Impacting Portfolio",
                 hover_data=["Event", "Sentiment", "Impact"],
                 color_discrete_map={"High": "#ef5350", "Medium": "#ffca28", "Low": "#66bb6a"})
fig.update_yaxes(autorange="reversed")
st.plotly_chart(fig, use_container_width=True)

# Display event details
st.subheader("Event Impact Analysis")
for _, event in events_df.iterrows():
    with st.expander(f"{event['Stock']}: {event['Event'].title()} Event ({event['Date'].strftime('%b %d')})"):
        st.metric("Risk Level", event["Risk"], 
                  "High Impact" if event["Impact"] == "High" else "Moderate Impact" if event["Impact"] == "Medium" else "Low Impact")
        st.metric("News Sentiment", f"{event['Sentiment']:.2f}", 
                  "Positive" if event['Sentiment'] > 0 else "Negative")
        
        # Event impact description
        if event["Sentiment"] > 0.5:
            st.success("✅ Positive event with strong market reception")
        elif event["Sentiment"] < -0.3:
            st.error("❌ Negative event with significant market concerns")
        else:
            st.info("ℹ️ Neutral event with limited market impact")
        
        # Recommendation based on event
        if event["Risk"] == "High" and event["Sentiment"] < 0:
            st.markdown("**Recommendation:** Consider reducing position immediately")
        elif event["Risk"] == "High" and event["Sentiment"] < 0.3:
            st.markdown("**Recommendation:** Monitor closely and set stop-loss")
        elif event["Sentiment"] > 0.5:
            st.markdown("**Recommendation:** Potential buying opportunity")
        else:
            st.markdown("**Recommendation:** Maintain current position")
