import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import requests
from functools import lru_cache
import yfinance as yf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler, StandardScaler

# Page Configuration
st.set_page_config(
    page_title="Indian Market Investment Analysis",
    page_icon="📈",
    layout="wide"
)

# CSS Styling
st.markdown("""
<style>
    .stApp {max-width: 1200px; margin: 0 auto;}
    .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
    .chart-container {background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

# API Configuration
ALPHA_VANTAGE_API_KEY = "UR API KEY"
BASE_URL = "https://www.alphavantage.co/query"
MFAPI_BASE_URL = "https://api.mfapi.in/mf"

# Fetch Functions
@lru_cache(maxsize=128)
def fetch_alpha_vantage_data(symbol, function="TIME_SERIES_DAILY", outputsize="full"):
    try:
        url = f"{BASE_URL}?function={function}&symbol={symbol}&outputsize={outputsize}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "Time Series (Daily)" not in data:
            return None, f"Invalid response for {symbol}"
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
        df = df.rename(columns={
            '1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'
        })
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        return df.sort_index(), None
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=3600)
def fetch_mfapi_data(scheme_code):
    try:
        url = f"{MFAPI_BASE_URL}/{scheme_code}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "SUCCESS":
            return None, f"Error in MFAPI response: {data.get('status')}"
        df = pd.DataFrame(data["data"])
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
        df["nav"] = df["nav"].astype(float)
        df = df.sort_values("date")
        latest_nav = df["nav"].iloc[-1]
        returns_1y = ((latest_nav / df["nav"].iloc[-252] - 1) * 100) if len(df) > 252 else 0
        returns_3y = ((latest_nav / df["nav"].iloc[-756] - 1) * 100) if len(df) > 756 else 0
        returns_5y = ((latest_nav / df["nav"].iloc[-1260] - 1) * 100) if len(df) > 1260 else 0
        fund_size = data["meta"]["fund_house"]
        expense_ratio = "N/A"
        return df, fund_size, returns_1y, returns_3y, returns_5y, expense_ratio, None
    except Exception as e:
        return None, None, 0, 0, 0, "N/A", str(e)

@st.cache_data(ttl=3600)
def get_stock_data(symbol, exchange="BSE"):
    suffix = "BSE" if exchange == "BSE" else "NSE"
    formatted_symbol = f"{symbol}.{suffix}"
    df, error = fetch_alpha_vantage_data(formatted_symbol)
    if df is not None:
        return df, "Close"
    try:
        stock = yf.Ticker(formatted_symbol)
        hist = stock.history(period="2y")
        if not hist.empty:
            return hist, "Close"
    except Exception as e:
        st.warning(f"Using yfinance fallback failed for {formatted_symbol}: {str(e)}")
    return None, "Close"

# IndianPortfolioBuilder Class
class IndianPortfolioBuilder:
    def __init__(self):
        self.asset_classes = ['Equity', 'Debt', 'Gold', 'Cash']
        self.sample_assets = {
            'Equity': {'NIFTY 50': {'return': 0.12, 'volatility': 0.18, 'symbol': '^NSEI'}},
            'Debt': {'HDFC Corporate Bond Fund': {'return': 0.07, 'volatility': 0.03, 'scheme_code': '119021'}},
            'Gold': {'Gold ETF': {'return': 0.08, 'volatility': 0.12, 'symbol': 'GOLDBEES.NS'}},
            'Cash': {'Liquid Fund': {'return': 0.04, 'volatility': 0.01, 'symbol': None}}
        }
        self.risk_free_rate = 0.06

    def create_allocation_ui(self):
        st.subheader("Portfolio Allocation")
        allocations = {}
        remaining = 100
        
        cols = st.columns(len(self.asset_classes))
        for i, asset in enumerate(self.asset_classes[:-1]):
            with cols[i]:
                value = st.number_input(
                    f"{asset} (%)",
                    min_value=0,
                    max_value=remaining,
                    value=min(25, remaining),
                    key=f"alloc_{asset}"
                )
                allocations[asset] = value
                remaining -= value
        
        allocations['Cash'] = remaining
        with cols[-1]:
            st.metric("Cash", f"{remaining}%")
        
        st.subheader("Customize Assumptions")
        custom_returns = {}
        custom_volatility = {}
        for asset in self.asset_classes:
            with st.expander(f"{asset} Assumptions"):
                sample_asset = list(self.sample_assets[asset].keys())[0]
                default_return = self.sample_assets[asset][sample_asset]['return'] * 100
                default_vol = self.sample_assets[asset][sample_asset]['volatility'] * 100
                custom_returns[asset] = st.number_input(
                    f"Expected Annual Return for {asset} (%)", 
                    value=default_return, 
                    key=f"ret_{asset}"
                ) / 100  # type: ignore
                custom_volatility[asset] = st.number_input(
                    f"Expected Volatility for {asset} (%)", 
                    value=default_vol, 
                    key=f"vol_{asset}"
                ) / 100  # type: ignore
        return allocations, custom_returns, custom_volatility

    def calculate_portfolio_metrics(self, allocations, custom_returns, custom_volatility):
        weights = np.array([allocations[asset] / 100 for asset in self.asset_classes])
        returns = np.array([custom_returns[asset] for asset in self.asset_classes])
        volatilities = np.array([custom_volatility[asset] for asset in self.asset_classes])
        portfolio_return = np.sum(weights * returns)
        portfolio_volatility = np.sqrt(np.sum(weights**2 * volatilities**2))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def suggest_optimized_allocation(self, custom_returns, custom_volatility):
        n_assets = len(self.asset_classes)
        base_weight = 100 / n_assets
        optimized = {}
        total = 0
        for i, asset in enumerate(self.asset_classes[:-1]):
            sharpe_contrib = (custom_returns[asset] - self.risk_free_rate) / custom_volatility[asset]
            weight = min(max(base_weight * (1 + sharpe_contrib / 2), 0), 100 - total)
            optimized[asset] = weight
            total += weight
        optimized['Cash'] = 100 - total
        return optimized

# Plotting Functions
def plot_allocation_pie(allocations, title="Portfolio Allocation"):
    fig = px.pie(
        values=list(allocations.values()),
        names=list(allocations.keys()),
        title=title,
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    return fig

def plot_performance_simulation(allocations, custom_returns, custom_volatility, years=5):
    months = years * 12
    monthly_returns = {asset: (1 + custom_returns[asset]) ** (1/12) - 1 for asset in allocations}
    monthly_vols = {asset: custom_volatility[asset] / np.sqrt(12) for asset in allocations}
    initial_investment = 1000000
    portfolio_values = [initial_investment]
    for _ in range(months):
        period_return = 0
        for asset in allocations:
            weight = allocations[asset] / 100
            period_return += weight * np.random.normal(monthly_returns[asset], monthly_vols[asset])
        portfolio_values.append(portfolio_values[-1] * (1 + period_return))
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=portfolio_values, mode='lines', name='Portfolio Value'))
    fig.update_layout(title="Portfolio Value Simulation", xaxis_title="Months", yaxis_title="Value (₹)")
    return fig

# Tax Calculator Class
class TaxCalculator:
    TAX_SLABS = {
        'old': [(250000, 0), (500000, 0.05), (750000, 0.10), (1000000, 0.15), 
                (1250000, 0.20), (1500000, 0.25), (float('inf'), 0.30)],
        'new': [(300000, 0), (600000, 0.05), (900000, 0.10), (1200000, 0.15), 
                (1500000, 0.20), (float('inf'), 0.30)]
    }
    
    def calculate_tax(self, income, regime='old'):
        tax = 0
        prev_slab = 0
        for slab, rate in self.TAX_SLABS[regime]:
            if income <= prev_slab:
                break
            taxable = min(income, slab) - prev_slab
            tax += taxable * rate
            prev_slab = slab
        cess = tax * 0.04
        return tax + cess, tax

# FL Model Definition
class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super(AttentionBlock, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        self.fc = nn.Linear(dim, dim)
        
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = self.fc(out)
        return out + x

class SuperModel(nn.Module):
    def __init__(self, input_dim):
        super(SuperModel, self).__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.attention = AttentionBlock(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(0.3)
        self.skip1 = nn.Linear(input_dim, 128)
        self.skip2 = nn.Linear(128, 256)
        self.skip3 = nn.Linear(256, 128)
        
    def forward(self, x):
        x = self.input_bn(x)
        identity = self.skip1(x)
        x = F.silu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x += identity
        
        identity = self.skip2(x)
        x = F.silu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x += identity
        
        x = self.attention(x)
        identity = self.skip3(x)
        x = F.silu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x += identity
        
        x = self.fc4(x)
        return x

# Preprocessing for Prediction
def super_preprocess(features):
    feature_scaler = RobustScaler()
    features_scaled = feature_scaler.fit_transform(features)
    features_poly = np.hstack([features_scaled, np.power(features_scaled, 2)])
    ema_features = np.copy(features_scaled)
    alpha = 0.3
    for i in range(1, len(features_scaled)):
        ema_features[i] = alpha * features_scaled[i] + (1 - alpha) * ema_features[i-1]
    features_poly = np.hstack([features_poly, ema_features])
    return torch.FloatTensor(features_poly)

# Load Trained Model
@st.cache_resource
def load_model():
    numeric_columns = [
        'sortino', 'alpha', 'sd', 'beta', 'sharpe', 'returns_1yr', 'returns_3yr', 'returns_5yr',
        'category', 'sub_category', 'amc_name', 'fund_manager', 'risk_return_score'
    ]
    input_dim = len(numeric_columns) * 3  # Original + quadratic + EMA
    model = SuperModel(input_dim=input_dim)
    try:
        model.load_state_dict(torch.load('mutual_fund_model.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Model file 'mutual_fund_model.pth' not found. Please ensure it exists in the working directory.")
        return None

# Main Function
def main():
    st.title("Indian Market Investment Analysis")
    
    page = st.sidebar.selectbox(
        "Select Analysis",
        ["Portfolio Builder", "Stock Analysis", "Mutual Fund Explorer", "SIP Calculator", "Tax Calculator"]
    )
    
    if page == "Portfolio Builder":
        portfolio = IndianPortfolioBuilder()
        allocations, custom_returns, custom_volatility = portfolio.create_allocation_ui()
        st.subheader("Portfolio Metrics")
        port_return, port_volatility, sharpe_ratio = portfolio.calculate_portfolio_metrics(
            allocations, custom_returns, custom_volatility
        )
        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Annual Return", f"{port_return*100:.2f}%")
        col2.metric("Portfolio Volatility", f"{port_volatility*100:.2f}%")
        col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        st.subheader("Optimized Allocation Suggestion")
        optimized_alloc = portfolio.suggest_optimized_allocation(custom_returns, custom_volatility)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_allocation_pie(allocations, "Current Allocation"), use_container_width=True)
        with col2:
            st.plotly_chart(plot_allocation_pie(optimized_alloc, "Optimized Allocation"), use_container_width=True)
        st.subheader("Portfolio Performance Simulation (5 Years)")
        st.plotly_chart(plot_performance_simulation(allocations, custom_returns, custom_volatility), use_container_width=True)
        st.subheader("Tax Implications")
        annual_gain = st.number_input("Expected Annual Portfolio Gain (₹)", value=100000, step=10000)
        ltcg_tax = max(0, annual_gain - 100000) * 0.1
        st.write(f"Long Term Capital Gains Tax (10% beyond ₹1L): ₹{ltcg_tax:,.2f}")
        if st.button("Export Portfolio"):
            df = pd.DataFrame({
                "Asset Class": list(allocations.keys()),
                "Allocation (%)": list(allocations.values()),
                "Expected Return (%)": [custom_returns[asset]*100 for asset in allocations],
                "Volatility (%)": [custom_volatility[asset]*100 for asset in allocations]
            })
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv, "portfolio.csv", "text/csv")

    elif page == "Stock Analysis":
        st.header("Stock Analysis")
        exchange = st.selectbox("Exchange", ["BSE"])
        
        stock_options = {
            "Adani Enterprises": {"NSE": "ADANIENT.NS", "BSE": "ADANIENT.BSE"},
            "Adani Ports": {"NSE": "ADANIPORTS.NS", "BSE": "ADANIPORTS.BSE"},
            "Apollo Hospitals": {"NSE": "APOLLOHOSP.NS", "BSE": "APOLLOHOSP.BSE"},
            "Asian Paints": {"NSE": "ASIANPAINT.NS", "BSE": "ASIANPAINT.BSE"},
            "Axis Bank": {"NSE": "AXISBANK.NS", "BSE": "AXISBANK.BSE"},
            "Bajaj Auto": {"NSE": "BAJAJ-AUTO.NS", "BSE": "BAJAJ-AUTO.BSE"},
            "Bajaj Finance": {"NSE": "BAJFINANCE.NS", "BSE": "BAJFINANCE.BSE"},
            "Bajaj Finserv": {"NSE": "BAJAJFINSV.NS", "BSE": "BAJAJFINSV.BSE"},
            "Bharti Airtel": {"NSE": "BHARTIARTL.NS", "BSE": "BHARTIARTL.BSE"},
            "BPCL": {"NSE": "BPCL.NS", "BSE": "BPCL.BSE"},
            "Britannia Industries": {"NSE": "BRITANNIA.NS", "BSE": "BRITANNIA.BSE"},
            "Cipla": {"NSE": "CIPLA.NS", "BSE": "CIPLA.BSE"},
            "Coal India": {"NSE": "COALINDIA.NS", "BSE": "COALINDIA.BSE"},
            "Divis Laboratories": {"NSE": "DIVISLAB.NS", "BSE": "DIVISLAB.BSE"},
            "Dr. Reddy's Laboratories": {"NSE": "DRREDDY.NS", "BSE": "DRREDDY.BSE"},
            "Eicher Motors": {"NSE": "EICHERMOT.NS", "BSE": "EICHERMOT.BSE"},
            "Grasim Industries": {"NSE": "GRASIM.NS", "BSE": "GRASIM.BSE"},
            "HCL Technologies": {"NSE": "HCLTECH.NS", "BSE": "HCLTECH.BSE"},
            "HDFC Bank": {"NSE": "HDFCBANK.NS", "BSE": "HDFCBANK.BSE"},
            "HDFC Life Insurance": {"NSE": "HDFCLIFE.NS", "BSE": "HDFCLIFE.BSE"},
            "Hero MotoCorp": {"NSE": "HEROMOTOCO.NS", "BSE": "HEROMOTOCO.BSE"},
            "Hindalco Industries": {"NSE": "HINDALCO.NS", "BSE": "HINDALCO.BSE"},
            "Hindustan Unilever": {"NSE": "HINDUNILVR.NS", "BSE": "HINDUNILVR.BSE"},
            "ICICI Bank": {"NSE": "ICICIBANK.NS", "BSE": "ICICIBANK.BSE"},
            "IndusInd Bank": {"NSE": "INDUSINDBK.NS", "BSE": "INDUSINDBK.BSE"},
            "Infosys": {"NSE": "INFY.NS", "BSE": "INFY.BSE"},
            "ITC": {"NSE": "ITC.NS", "BSE": "ITC.BSE"},
            "JSW Steel": {"NSE": "JSWSTEEL.NS", "BSE": "JSWSTEEL.BSE"},
            "Kotak Mahindra Bank": {"NSE": "KOTAKBANK.NS", "BSE": "KOTAKBANK.BSE"},
            "Larsen & Toubro": {"NSE": "LT.NS", "BSE": "LT.BSE"},
            "LTIMindtree": {"NSE": "LTIM.NS", "BSE": "LTIM.BSE"},
            "Mahindra & Mahindra": {"NSE": "M&M.NS", "BSE": "M&M.BSE"},
            "Maruti Suzuki": {"NSE": "MARUTI.NS", "BSE": "MARUTI.BSE"},
            "Nestle India": {"NSE": "NESTLEIND.NS", "BSE": "NESTLEIND.BSE"},
            "NTPC": {"NSE": "NTPC.NS", "BSE": "NTPC.BSE"},
            "ONGC": {"NSE": "ONGC.NS", "BSE": "ONGC.BSE"},
            "Power Grid Corporation": {"NSE": "POWERGRID.NS", "BSE": "POWERGRID.BSE"},
            "Reliance Industries": {"NSE": "RELIANCE.NS", "BSE": "RELIANCE.BSE"},  # Corrected "Roger Federer" typo
            "SBI Life Insurance": {"NSE": "SBILIFE.NS", "BSE": "SBILIFE.BSE"},
            "State Bank of India": {"NSE": "SBIN.NS", "BSE": "SBIN.BSE"},
            "Sun Pharmaceutical": {"NSE": "SUNPHARMA.NS", "BSE": "SUNPHARMA.BSE"},
            "Tata Consultancy Services": {"NSE": "TCS.NS", "BSE": "TCS.BSE"},
            "Tata Consumer Products": {"NSE": "TATACONSUM.NS", "BSE": "TATACONSUM.BSE"},
            "Tata Motors": {"NSE": "TATAMOTORS.NS", "BSE": "TATAMOTORS.BSE"},
            "Tata Steel": {"NSE": "TATASTEEL.NS", "BSE": "TATASTEEL.BSE"},
            "Tech Mahindra": {"NSE": "TECHM.NS", "BSE": "TECHM.BSE"},
            "Titan Company": {"NSE": "TITAN.NS", "BSE": "TITAN.BSE"},
            "UltraTech Cement": {"NSE": "ULTRACEMCO.NS", "BSE": "ULTRACEMCO.BSE"},
            "UPL": {"NSE": "UPL.NS", "BSE": "UPL.BSE"},
            "Wipro": {"NSE": "WIPRO.NS", "BSE": "WIPRO.BSE"}
        }
        
        selected_stock = st.selectbox(
            f"Select a Stock ({exchange})",
            options=["Custom"] + sorted(list(stock_options.keys())),
            help="Choose a stock from the Nifty 50 index or select 'Custom' to enter your own symbol."
        )
        
        if selected_stock == "Custom":
            symbol = st.text_input(
                f"Enter {exchange} Symbol (e.g., {'RELIANCE.NS' if exchange == 'NSE' else 'RELIANCE.BSE'})",
                value="RELIANCE"
            )
        else:
            symbol = stock_options[selected_stock][exchange] # type: ignore
            st.write(f"Using symbol: `{symbol}`")

        df, price_col = get_stock_data(symbol.split('.')[0], exchange) # type: ignore
        if df is not None and not df.empty:
            latest = df[price_col].iloc[-1]
            change = (latest - df[price_col].iloc[0]) / df[price_col].iloc[0] * 100
            col1, col2 = st.columns(2)
            col1.metric("Latest Price", f"₹{latest:.2f}")
            col2.metric("Change %", f"{change:.2f}%")
            fig = px.line(df, y=price_col, title=f"{symbol} Performance")
            fig.update_layout(xaxis_title="Date", yaxis_title="Price (₹)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Failed to fetch stock data. Please check the symbol or try another stock.")

    elif page == "Mutual Fund Explorer":
        st.header("Mutual Fund Explorer")
        fund_options = {
            "Equity": {
                "HDFC Top 100 Fund - Direct Growth": "118934",
                "ICICI Pru Bluechip Fund - Direct Growth": "120586",
                "SBI Small Cap Fund - Direct Growth": "125494"
            },
            "Debt": {
                "HDFC Corporate Bond Fund - Direct Growth": "119021",
                "ICICI Pru Short Term Fund - Direct Growth": "120604"
            },
            "Hybrid": {
                "ICICI Pru Balanced Advantage Fund - Direct Growth": "120704"
            }
        }
        category = st.selectbox("Select Category", list(fund_options.keys()))
        fund_name = st.selectbox("Select Fund", list(fund_options[category].keys())) # type: ignore
        scheme_code = fund_options[category][fund_name] # type: ignore
        fund_data, fund_size, returns_1y, returns_3y, returns_5y, expense_ratio, error = fetch_mfapi_data(scheme_code) # type: ignore
        
        if fund_data is not None and not fund_data.empty:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Fund House", fund_size)
            col2.metric("1Y Return", f"{returns_1y:.2f}%")
            col3.metric("3Y Return", f"{returns_3y:.2f}%")
            col4.metric("5Y Return", f"{returns_5y:.2f}%")
            st.metric("Expense Ratio", expense_ratio)
            fig = px.line(fund_data, x="date", y="nav", title=f"{fund_name} NAV Trend")
            fig.update_layout(xaxis_title="Date", yaxis_title="NAV (₹)")
            st.plotly_chart(fig, use_container_width=True)
            
            # AI-Based Recommendation Section
            st.subheader("AI-Based 1-Year Return Prediction")
            model = load_model()
            if model is not None:
                features = {
                    'sortino': st.number_input("Sortino Ratio", value=1.5, step=0.1),
                    'alpha': st.number_input("Alpha", value=2.0, step=0.1),
                    'sd': st.number_input("Standard Deviation", value=15.0, step=0.5),
                    'beta': st.number_input("Beta", value=1.0, step=0.1),
                    'sharpe': st.number_input("Sharpe Ratio", value=0.8, step=0.1),
                    'returns_1yr': returns_1y / 100,
                    'returns_3yr': returns_3y / 100,
                    'returns_5yr': returns_5y / 100,
                    'category': float(list(fund_options.keys()).index(category)), # type: ignore
                    'sub_category': 0.0,
                    'amc_name': float(fund_size.lower().find("icici") != -1), # type: ignore
                    'fund_manager': 0.0,
                    'risk_return_score': (returns_1y + returns_3y + returns_5y) / 3
                }
                feature_array = np.array([list(features.values())])
                preprocessed_features = super_preprocess(feature_array)
                
                with torch.no_grad():
                    pred_scaled = model(preprocessed_features)
                    target_scaler = StandardScaler()
                    sample_returns = np.array([returns_1y, returns_3y, returns_5y]).reshape(-1, 1)
                    target_scaler.fit(sample_returns)
                    pred_unscaled = target_scaler.inverse_transform(pred_scaled.numpy().reshape(-1, 1))[0][0]
                
                st.metric("Predicted 1-Year Return", f"{pred_unscaled:.2f}%")
                if pred_unscaled > returns_1y:
                    st.success("The AI model predicts a higher return than the past year, suggesting potential growth.")
                elif pred_unscaled < returns_1y:
                    st.warning("The AI model predicts a lower return than the past year, suggesting caution.")
                else:
                    st.info("The AI model predicts a return similar to the past year.")

    elif page == "SIP Calculator":
        st.header("SIP Calculator")
        col1, col2 = st.columns(2)
        with col1:
            monthly = st.number_input("Monthly SIP (₹)", value=10000, step=1000)
            years = st.slider("Years", 1, 30, 10)
        with col2:
            returns = st.slider("Expected Return (%)", 5, 20, 12) / 100
            risk = st.slider("Risk Level", 1, 5, 3)
        months = years * 12
        monthly_rate = (1 + returns) ** (1/12) - 1
        values = [0]
        for _ in range(months):
            values.append((values[-1] + monthly) * (1 + np.random.normal(monthly_rate, risk * 0.01)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=values, mode='lines', name='Projection'))
        fig.update_layout(title="SIP Growth", xaxis_title="Months", yaxis_title="Value (₹)")
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Tax Calculator":
        st.header("Tax Calculator")
        tax_calc = TaxCalculator()
        regime = st.radio("Tax Regime", ['old', 'new'])
        income = st.number_input("Annual Income (₹)", value=500000, step=50000)
        if regime == 'old':
            deductions = st.number_input("Deductions (₹)", value=150000, step=10000)
            taxable = max(0, income - deductions)
        else:
            taxable = income
        total_tax, base_tax = tax_calc.calculate_tax(taxable, regime) # type: ignore
        col1, col2 = st.columns(2)
        col1.metric("Taxable Income", f"₹{taxable:,}")
        col2.metric("Total Tax", f"₹{total_tax:,.2f}")

if __name__ == "__main__":
    main()
