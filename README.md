# MutualFundAI

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Welcome to **MutualFundAI**, a portfolio project that harnesses deep learning and federated learning to predict mutual fund returns in the Indian market. This project combines financial data analysis with cutting-edge AI techniques and presents the results through an interactive Streamlit dashboard. Whether you're an investor, a data scientist, or a developer, this project offers a glimpse into the power of AI-driven financial insights.

## Project Overview

MutualFundAI aims to predict mutual fund performance (specifically 1-year returns) using a custom deep learning model with attention mechanisms, trained via a federated learning approach. It integrates real-time and historical financial data from APIs like Alpha Vantage, MFAPI, and Yahoo Finance. The project also includes a user-friendly web interface built with Streamlit, featuring tools for portfolio management, stock analysis, mutual fund exploration, SIP calculations, and tax planning.

### Why This Project?
- **Real-World Application**: Demonstrates how AI can enhance investment decisions in the Indian financial market.
- **Technical Depth**: Showcases advanced machine learning (deep learning with attention, federated learning) and full-stack development (Streamlit).
- **Portfolio Showcase**: Highlights skills in Python, PyTorch, data preprocessing, API integration, and UI design.

### Key Features
1. **Deep Learning Model**:
   - Custom `SuperModel` with attention blocks, skip connections, and dropout for robust predictions.
   - Predicts 1-year mutual fund returns based on features like Sortino, Sharpe, and historical returns.
2. **Federated Learning**:
   - Uses FedProx aggregation to train across multiple "clients" (simulated datasets), mimicking distributed environments.
   - Ensures scalability and privacy-preserving training.
3. **Financial Data Integration**:
   - Fetches stock data (e.g., Nifty 50) and mutual fund NAVs from APIs.
   - Supports BSE stocks and popular Indian mutual funds.
4. **Interactive Dashboard**:
   - **Portfolio Builder**: Allocate assets, simulate performance, and optimize allocations.
   - **Stock Analysis**: Visualize price trends for Indian stocks.
   - **Mutual Fund Explorer**: Analyze NAV trends and get AI-predicted returns.
   - **SIP Calculator**: Project Systematic Investment Plan growth.
   - **Tax Calculator**: Compute tax liability under old/new regimes.
5. **Advanced Preprocessing**:
   - Robust scaling, polynomial features, and exponential moving averages (EMA) for enhanced model input.

## Installation

### Prerequisites
- **Python**: Version 3.8 or higher (for compatibility with dependencies).
- **Git**: To clone the repository.
- **Virtual Environment**: Recommended to isolate dependencies.
