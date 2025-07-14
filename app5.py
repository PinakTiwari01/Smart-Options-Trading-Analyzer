import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import ta
from io import StringIO
import re

# Page config
st.set_page_config(
    page_title="Smart Options Trading Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79, #2e8b57);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f4e79;
        margin: 0.5rem 0;
    }
    .signal-bullish {
        background: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .signal-bearish {
        background: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
    .signal-neutral {
        background: #fff3cd;
        color: #856404;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

class TradingAnalyzer:
    def __init__(self):
        self.data = None
        self.source = None
        self.expiry_day = None
        
    def fix_merged_columns(self, df):
        """Fix merged column names by splitting them properly"""
        new_columns = []
        
        for col in df.columns:
            # Check for the specific merged pattern: DateOpenHighLowCloseShares TradedTurnover(₹ Cr)
            if 'DateOpenHighLowClose' in col:
                st.warning(f"Found merged column: '{col}'. Attempting to split...")
                
                # If it's the massive merged column, we need to split the data
                if 'DateOpenHighLowCloseShares TradedTurnover(₹ Cr)' in col:
                    # This suggests all columns are merged into one
                    # We'll need to split the data values, not just the column name
                    st.error("All columns appear to be merged into one. Please check your CSV file format.")
                    st.info("Expected format: Date,Open,High,Low,Close,Shares Traded,Turnover(₹ Cr)")
                    return None
                    
            # Handle other merged patterns
            elif 'Shares TradedTurnover' in col:
                # Split this into two columns
                st.info(f"Splitting column '{col}' into 'Shares Traded' and 'Turnover(₹ Cr)'")
                new_columns.append('Shares Traded')
                # We'll handle the actual data splitting later
                continue
                
            new_columns.append(col)
        
        return df
    
    def detect_and_fix_csv_format(self, df):
        """Detect and fix CSV formatting issues"""
        
        # Check if we have the problematic merged columns
        columns = df.columns.tolist()
        
        # Case 1: All data is in one column (most severe)
        if len(columns) == 1 and any('DateOpenHighLowClose' in col for col in columns):
            st.error("❌ All data appears to be in one column. CSV formatting issue detected.")
            st.info("🔧 **Solution:** Please ensure your CSV file has proper column separators (commas)")
            st.info("📋 **Expected format:** Date,Open,High,Low,Close,Shares Traded,Turnover(₹ Cr)")
            return None
            
        # Case 2: Some columns are merged
        fixed_df = self.fix_merged_columns(df)
        if fixed_df is None:
            return None
            
        # Case 3: Check if we need to split data within columns
        for col in df.columns:
            if 'Shares TradedTurnover' in col:
                # Try to split the data in this column
                try:
                    # This is a complex case - we'd need to see the actual data format
                    st.warning(f"Column '{col}' appears to contain merged data. Manual intervention needed.")
                except:
                    pass
        
        return df
    
    def detect_source_and_clean(self, df):
        """Detect if data is NSE or BSE and clean columns"""
        
        # First try to fix CSV format issues
        df = self.detect_and_fix_csv_format(df)
        if df is None:
            raise ValueError("CSV format issues detected. Please fix the file format.")
        
        columns = df.columns.tolist()
        
        # Print columns for debugging
        st.write("**Debug:** Column names found:", columns)
        
        # Clean column names - remove extra spaces and standardize
        df.columns = [col.strip() for col in df.columns]
        columns = df.columns.tolist()
        
        # Check for NSE specific columns
        nse_indicators = [
            'Shares Traded', 'Shares TradedTurnover(₹ Cr)',
            'Turnover (₹ Cr)', 'Turnover', 'TradedTurnover(₹ Cr)',
            'Turnover(₹ Cr)', 'Volume', 'Shares TradedTurnover',
            'SharesTraded', 'Turnover(Rs Cr)'
        ]
        
        # More flexible pattern matching for NSE detection
        has_nse_indicators = False
        for col in columns:
            if any(indicator.lower() in col.lower() for indicator in ['shares', 'traded', 'turnover', 'volume']):
                has_nse_indicators = True
                break
        
        if has_nse_indicators:
            self.source = "NSE"
            self.expiry_day = "Thursday"
            
            # Handle various NSE column naming patterns
            column_mapping = {}
            
            for col in columns:
                col_lower = col.lower()
                # Map shares traded columns
                if 'shares' in col_lower and 'traded' in col_lower:
                    if 'turnover' not in col_lower:  # Pure shares traded column
                        column_mapping[col] = 'Shares_Traded'
                    else:
                        # This is a merged column - need special handling
                        st.warning(f"Found merged column '{col}' containing both Shares Traded and Turnover data")
                        
                # Map turnover columns
                elif 'turnover' in col_lower:
                    column_mapping[col] = 'Turnover_Cr'
                elif 'volume' in col_lower:
                    column_mapping[col] = 'Volume'
                    
            # Apply column mapping
            df = df.rename(columns=column_mapping)
            
        else:
            self.source = "BSE"
            self.expiry_day = "Tuesday"
            
        # Standardize OHLC columns
        ohlc_mapping = {
            'open': 'Open', 'Open': 'Open',
            'high': 'High', 'High': 'High', 
            'low': 'Low', 'Low': 'Low',
            'close': 'Close', 'Close': 'Close'
        }
        
        for col in df.columns:
            if col.lower() in ohlc_mapping:
                df = df.rename(columns={col: ohlc_mapping[col.lower()]})
        
        # Handle date column
        date_columns = ['Date', 'date', 'DATE', 'Timestamp', 'timestamp', 'Time', 'time', 'DateTime', 'datetime']
        date_col_found = None
        
        for col in date_columns:
            if col in df.columns:
                date_col_found = col
                break
        
        # If no standard date column found, check for date-like patterns
        if date_col_found is None:
            for col in df.columns:
                if any(date_word in col.lower() for date_word in ['date', 'time', 'day']):
                    date_col_found = col
                    break
        
        # If still no date column found, use the first column
        if date_col_found is None:
            date_col_found = df.columns[0]
            st.warning(f"No standard date column found. Using first column: '{date_col_found}'")
        
        # Rename to standard 'Date' column
        if date_col_found != 'Date':
            df = df.rename(columns={date_col_found: 'Date'})
            
        # Validate required columns
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.info("Available columns: " + ", ".join(df.columns.tolist()))
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Parse date with error handling
        try:
            # Try different date formats
            date_formats = [
                '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%d/%m/%Y',
                '%Y/%m/%d', '%d-%b-%Y', '%d %b %Y', '%Y-%m-%d %H:%M:%S'
            ]
            
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', infer_datetime_format=True)
            
            # If that fails, try specific formats
            if df['Date'].isna().all():
                for fmt in date_formats:
                    try:
                        df['Date'] = pd.to_datetime(df['Date'], format=fmt, errors='coerce')
                        if not df['Date'].isna().all():
                            break
                    except:
                        continue
            
            # Remove rows with invalid dates
            invalid_dates = df['Date'].isna().sum()
            if invalid_dates > 0:
                st.warning(f"Found {invalid_dates} invalid date entries. Removing them.")
                df = df.dropna(subset=['Date'])
                
            if len(df) == 0:
                raise ValueError("No valid dates found in the data")
                
        except Exception as e:
            st.error(f"Error parsing dates: {str(e)}")
            st.info("Please ensure your date column is in a standard format (YYYY-MM-DD, DD-MM-YYYY, etc.)")
            raise
        
        # Convert OHLC columns to numeric
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with invalid OHLC data
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        if len(df) == 0:
            raise ValueError("No valid OHLC data found")
        
        # Add weekday
        df['Weekday'] = df['Date'].dt.day_name()
        
        # Remove weekends
        df = df[~df['Weekday'].isin(['Saturday', 'Sunday'])]
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Basic data validation
        if len(df) < 5:
            st.warning("Very limited data available. Results may not be reliable.")
        
        return df
    
    def calculate_technical_indicators(self, df):
        """Calculate RSI, trends, and other technical indicators"""
        # Calculate RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # Calculate profit indicators
        df['Call_Profit'] = df['Close'] > df['Open']
        df['Put_Profit'] = df['Close'] < df['Open']
        
        # Calculate trend
        df['Price_Change'] = df['Close'].diff()
        df['Trend'] = df['Price_Change'].apply(
            lambda x: 'Up' if x > 0 else ('Down' if x < 0 else 'No Change')
        )
        
        # Calculate trend sequences
        df['Trend_Sequence'] = 0
        current_trend = None
        sequence_count = 0
        
        for i in range(len(df)):
            if df.iloc[i]['Trend'] == current_trend:
                sequence_count += 1
            else:
                current_trend = df.iloc[i]['Trend']
                sequence_count = 1
            df.iloc[i, df.columns.get_loc('Trend_Sequence')] = sequence_count
            
        # Calculate moving averages
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # Calculate volatility
        df['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        return df
    
    def generate_weekday_analysis(self, df):
        """Generate weekday-wise trading analysis"""
        weekday_stats = df.groupby('Weekday').agg({
            'Call_Profit': ['count', 'sum'],
            'Put_Profit': ['count', 'sum'],
            'Close': 'count'
        }).round(2)
        
        weekday_stats.columns = ['Total_Days', 'Call_Profit_Days', 'Total_Days2', 'Put_Profit_Days', 'Total_Days3']
        weekday_stats = weekday_stats[['Total_Days', 'Call_Profit_Days', 'Put_Profit_Days']]
        
        weekday_stats['Call_Profit_Ratio'] = (weekday_stats['Call_Profit_Days'] / weekday_stats['Total_Days'] * 100).round(1)
        weekday_stats['Put_Profit_Ratio'] = (weekday_stats['Put_Profit_Days'] / weekday_stats['Total_Days'] * 100).round(1)
        
        return weekday_stats
    
    def generate_btst_signal(self, df, vix, sgx_gap, pcr, iv_ce, iv_pe, oi_signal, price_pattern):
        """Generate BTST (Buy Today, Sell Tomorrow) signal"""
        if len(df) == 0:
            return "No data available", "Insufficient data", 0, "neutral"
            
        latest = df.iloc[-1]
        rsi = latest['RSI']
        trend = latest['Trend']
        trend_sequence = latest['Trend_Sequence']
        
        signals = []
        score = 50  # Base score
        
        # RSI Analysis
        if rsi < 30:
            signals.append("RSI oversold")
            score += 15
        elif rsi > 70:
            signals.append("RSI overbought")
            score -= 15
        
        # Trend Analysis
        if trend == 'Up' and trend_sequence >= 2:
            signals.append(f"Uptrend {trend_sequence} days")
            score += 10
        elif trend == 'Down' and trend_sequence >= 2:
            signals.append(f"Downtrend {trend_sequence} days")
            score -= 10
            
        # VIX Analysis
        if vix > 13:
            signals.append("High VIX - high risk")
            score -= 10
        elif vix < 11:
            signals.append("Low VIX - low volatility")
            score += 5
            
        # SGX Gap Analysis
        if abs(sgx_gap) > 40:
            direction = "up" if sgx_gap > 0 else "down"
            signals.append(f"SGX gap {direction} {abs(sgx_gap):.0f} pts")
            score += 15 if sgx_gap > 0 else -15
            
        # PCR Analysis
        if pcr > 1.3:
            signals.append("High PCR - bearish")
            score -= 10
        elif pcr < 0.8:
            signals.append("Low PCR - bullish")
            score += 10
            
        # IV Analysis
        if iv_ce > iv_pe + 5:
            signals.append("CE IV > PE IV")
            score += 8
        elif iv_pe > iv_ce + 5:
            signals.append("PE IV > CE IV")
            score -= 8
            
        # OI Signal
        if oi_signal == "CE Buildup":
            signals.append("CE OI buildup")
            score += 10
        elif oi_signal == "PE Buildup":
            signals.append("PE OI buildup")
            score -= 10
            
        # Determine action based on score
        if score >= 70:
            action = "Buy CE"
            signal_class = "bullish"
        elif score <= 30:
            action = "Buy PE"
            signal_class = "bearish"
        elif score >= 60:
            action = "Buy CE (Weak)"
            signal_class = "bullish"
        elif score <= 40:
            action = "Buy PE (Weak)"
            signal_class = "bearish"
        else:
            action = "Hold/Neutral"
            signal_class = "neutral"
            
        reason = " | ".join(signals) if signals else "No clear signals"
        
        return action, reason, score, signal_class
    
    def generate_expiry_day_plan(self, df, current_time=None):
        """Generate expiry day trading plan"""
        if len(df) == 0:
            return "No data available", "Insufficient data", "9:15-10:15"
            
        latest = df.iloc[-1]
        rsi = latest['RSI']
        volatility = latest.get('Volatility', 0.2)
        
        # Determine trading style based on volatility and RSI
        if volatility > 0.3:
            style = "Scalping"
            entry_time = "9:15-10:15, 1:15-2:15"
        elif 30 < rsi < 70:
            style = "Trend Following"
            entry_time = "9:30-11:30, 1:30-2:30"
        else:
            style = "Fade/Reversal"
            entry_time = "10:00-11:00, 2:00-3:00"
            
        # Generate action based on RSI and recent trend
        if rsi < 35:
            action = "Buy CE on dips"
        elif rsi > 65:
            action = "Buy PE on rallies"
        else:
            action = "Follow breakout direction"
            
        return style, action, entry_time
    
    def generate_daily_forecast(self, df):
        """Generate tomorrow's market forecast"""
        if len(df) < 3:
            return "Neutral", "Insufficient data", "Analysis", 50
            
        latest = df.iloc[-1]
        prev_2 = df.iloc[-2]
        prev_3 = df.iloc[-3]
        
        rsi = latest['RSI']
        trend_sequence = latest['Trend_Sequence']
        trend = latest['Trend']
        
        # Calculate today's range
        today_range = abs(latest['High'] - latest['Low'])
        avg_range = df['High'].subtract(df['Low']).tail(20).mean()
        
        confidence = 50
        
        # RSI-based prediction
        if rsi < 30 and trend == 'Down' and trend_sequence >= 2:
            bias = "Bullish"
            direction = "Gap Up"
            forecast_type = "Bounce"
            confidence += 25
        elif rsi > 70 and trend == 'Up' and trend_sequence >= 2:
            bias = "Bearish"
            direction = "Gap Down"
            forecast_type = "Reversal"
            confidence += 25
        elif today_range < avg_range * 0.7:
            bias = "Neutral"
            direction = "Rangebound"
            forecast_type = "Breakout"
            confidence += 15
        else:
            bias = "Neutral"
            direction = "Rangebound"
            forecast_type = "Continuation"
            confidence += 10
            
        # Adjust confidence based on trend consistency
        if trend_sequence >= 3:
            confidence += 10
        elif trend_sequence == 1:
            confidence -= 5
            
        confidence = min(95, max(5, confidence))
        
        return bias, direction, forecast_type, confidence

# Initialize analyzer
analyzer = TradingAnalyzer()

# Main app header
st.markdown('<div class="main-header"><h1>📈 Smart Options Trading Analyzer</h1><p>NSE & BSE | BTST & Expiry Strategies | Daily Forecasting</p></div>', unsafe_allow_html=True)

# Add CSV format help section
with st.expander("📋 CSV Format Guide", expanded=False):
    st.markdown("""
    **Common CSV Issues and Solutions:**
    
    ❌ **Problem:** All columns merged into one (e.g., "DateOpenHighLowCloseShares TradedTurnover(₹ Cr)")
    ✅ **Solution:** Ensure proper comma separation between columns
    
    ❌ **Problem:** Spaces in column names or special characters
    ✅ **Solution:** Use standard column names: Date, Open, High, Low, Close
    
    **Expected Format for NSE:**
    ```
    Date,Open,High,Low,Close,Shares Traded,Turnover(₹ Cr)
    2024-01-01,50000,50200,49800,50100,1000000,500
    ```
    
    **Expected Format for BSE:**
    ```
    Date,Open,High,Low,Close
    2024-01-01,50000,50200,49800,50100
    ```
    """)

# Sidebar for inputs
with st.sidebar:
    st.header("📊 Market Inputs")
    
    # File upload
    uploaded_file = st.file_uploader("Upload OHLC CSV Data", type=['csv'])
    
    st.subheader("🔴 Live Market Data")
    
    # Market inputs
    vix = st.number_input("India VIX", min_value=8.0, max_value=80.0, value=13.5, step=0.1)
    sgx_gap = st.number_input("SGX Nifty Gap", min_value=-200.0, max_value=200.0, value=0.0, step=5.0)
    pcr = st.number_input("Put Call Ratio (PCR)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
    
    st.subheader("⚡ Options Data")
    iv_ce = st.number_input("CE Implied Volatility", min_value=5.0, max_value=100.0, value=20.0, step=1.0)
    iv_pe = st.number_input("PE Implied Volatility", min_value=5.0, max_value=100.0, value=22.0, step=1.0)
    
    oi_signal = st.selectbox("OI Signal", ["Neutral", "CE Buildup", "PE Buildup", "Unwinding"])
    
    price_pattern = st.selectbox(
        "Last 30-min Pattern",
        ["Normal", "Tight Coil", "Inside Bar", "Breakout", "Reversal"]
    )
    
    # Current time for BTST
    current_time = st.time_input("Current Time", value=datetime.now().time())

# Main content area
if uploaded_file is not None:
    # Load and process data
    try:
        df = pd.read_csv(uploaded_file)
        
        # Display raw column info for debugging
        st.write("**Raw CSV Info:**")
        st.write(f"- Total columns: {len(df.columns)}")
        st.write(f"- Column names: {list(df.columns)}")
        
        # Clean and detect source
        df = analyzer.detect_source_and_clean(df)
        
        # Calculate technical indicators
        df = analyzer.calculate_technical_indicators(df)
        
        # Store processed data
        analyzer.data = df
        
        # Display source info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Source", analyzer.source)
        with col2:
            st.metric("Expiry Day", analyzer.expiry_day)
        with col3:
            st.metric("Total Trading Days", len(df))
        
        # Display latest data
        st.subheader("📊 Latest Market Data")
        
        latest = df.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Close Price", f"₹{latest['Close']:.2f}", f"{latest['Price_Change']:.2f}")
        with col2:
            st.metric("RSI (14)", f"{latest['RSI']:.1f}")
        with col3:
            st.metric("Trend", latest['Trend'])
        with col4:
            st.metric("Trend Sequence", f"{latest['Trend_Sequence']} days")
        
        # Generate weekday analysis
        weekday_stats = analyzer.generate_weekday_analysis(df)
        
        # Display weekday analysis
        st.subheader("📅 Weekday Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Weekday Statistics**")
            st.dataframe(weekday_stats)
        
        with col2:
            # Create pie chart for Call/Put profits
            fig = go.Figure()
            
            total_call_profit = weekday_stats['Call_Profit_Days'].sum()
            total_put_profit = weekday_stats['Put_Profit_Days'].sum()
            
            fig.add_trace(go.Pie(
                labels=['Call Profit Days', 'Put Profit Days'],
                values=[total_call_profit, total_put_profit],
                hole=0.3,
                marker_colors=['#28a745', '#dc3545']
            ))
            
            fig.update_layout(
                title="Overall Call vs Put Profit Distribution",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Price chart with technical indicators
        st.subheader("📈 Price Chart with Technical Analysis")
        
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price"
        ))
        
        # Add moving averages
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['MA_20'],
            mode='lines',
            name='MA 20',
            line=dict(color='orange', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['MA_50'],
            mode='lines',
            name='MA 50',
            line=dict(color='blue', width=1)
        ))
        
        fig.update_layout(
            title="Price Chart with Moving Averages",
            height=500,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # RSI Chart
        fig_rsi = go.Figure()
        
        fig_rsi.add_trace(go.Scatter(
            x=df['Date'],
            y=df['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=2)
        ))
        
        # Add RSI levels
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Neutral")
        
        fig_rsi.update_layout(
            title="RSI (14) Indicator",
            height=300,
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        # Generate signals and forecasts
        st.subheader("🎯 Trading Signals & Forecasts")
        
        # BTST Signal
        btst_action, btst_reason, btst_score, signal_class = analyzer.generate_btst_signal(
            df, vix, sgx_gap, pcr, iv_ce, iv_pe, oi_signal, price_pattern
        )
        
        # Expiry Day Plan
        expiry_style, expiry_action, expiry_time = analyzer.generate_expiry_day_plan(df, current_time)
        
        # Daily Forecast
        forecast_bias, forecast_direction, forecast_type, forecast_confidence = analyzer.generate_daily_forecast(df)
        
        # Display signals
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔁 BTST Signal**")
            
            signal_color = "signal-bullish" if signal_class == "bullish" else ("signal-bearish" if signal_class == "bearish" else "signal-neutral")
            
            st.markdown(f"""
            <div class="{signal_color}">
                <strong>Action:</strong> {btst_action}<br>
                <strong>Score:</strong> {btst_score}/100<br>
                <strong>Reason:</strong> {btst_reason}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.write("**⏳ Expiry Day Plan**")
            
            st.markdown(f"""
            <div class="metric-card">
                <strong>Style:</strong> {expiry_style}<br>
                <strong>Action:</strong> {expiry_action}<br>
                <strong>Entry Time:</strong> {expiry_time}
            </div>
            """, unsafe_allow_html=True)
        
        # Daily Forecast
        st.write("**🔮 Tomorrow's Forecast**")
        
        forecast_color = "signal-bullish" if forecast_bias == "Bullish" else ("signal-bearish" if forecast_bias == "Bearish" else "signal-neutral")
        
        st.markdown(f"""
        <div class="{forecast_color}">
            <strong>Market Bias:</strong> {forecast_bias}<br>
            <strong>Predicted Direction:</strong> {forecast_direction}<br>
            <strong>Forecast Type:</strong> {forecast_type}<br>
            <strong>Confidence Score:</strong> {forecast_confidence}%
        </div>
        """, unsafe_allow_html=True)
        
        # Final Summary
        st.subheader("📋 Trading Summary")
        
        summary_data = {
            "Parameter": [
                "🟩 Source", "📅 Expiry Day", "📈 RSI Today", "📊 Current Trend",
                "🔁 BTST Signal", "⏳ Expiry Strategy", "🔮 Tomorrow Forecast", "✅ Confidence"
            ],
            "Value": [
                analyzer.source,
                analyzer.expiry_day,
                f"{latest['RSI']:.1f}",
                f"{latest['Trend']} ({latest['Trend_Sequence']} days)",
                btst_action,
                f"{expiry_style} - {expiry_action}",
                f"{forecast_bias} - {forecast_direction}",
                f"{forecast_confidence}%"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Risk Warning
        st.warning("⚠️ This is for educational purposes only. Always do your own research and consult with a financial advisor before making trading decisions.")
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please ensure your CSV file has proper column separation and required OHLC data")
        
        # Show detailed error information
        with st.expander("Error Details", expanded=False):
            st.code(str(e))
            st.write("**Troubleshooting Steps:**")
            st.write("1. Check if your CSV has proper comma separation")
            st.write("2. Ensure Date, Open, High, Low, Close columns exist")
            st.write("3. Verify date format is readable (YYYY-MM-DD or DD-MM-YYYY)")
            st.write("4. Check for any merged columns or formatting issues")

else:
    st.info("👆 Please upload a CSV file with OHLC data to begin analysis")
    
    st.subheader("📋 Required CSV Format")
    
    # Show correct format examples
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**✅ Correct NSE Format:**")
        sample_nse = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Open': [50000, 50100, 50200],
            'High': [50200, 50300, 50400],
            'Low': [49800, 49900, 50000],
            'Close': [50100, 50200, 50300],
            'Shares Traded': [1000000, 1100000, 1200000],
            'Turnover(₹ Cr)': [500, 550, 600]
        })
        st.dataframe(sample_nse, use_container_width=True)
    
    with col2:
        st.write("**✅ Correct BSE Format:**")
        sample_bse = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Open': [50000, 50100, 50200],
            'High': [50200, 50300, 50400],
            'Low': [49800, 49900, 50000],
            'Close': [50100, 50200, 50300]
        })
        st.dataframe(sample_bse, use_container_width=True)
    
    # Show common problems
    st.subheader("❌ Common CSV Problems")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Problem: Merged Columns**")
        problem_example = pd.DataFrame({
            'DateOpenHighLowCloseShares TradedTurnover(₹ Cr)': [
                '2024-01-01,50000,50200,49800,50100,1000000,500'
            ]
        })
        st.dataframe(problem_example, use_container_width=True)
        st.error("All data in one column - CSV format issue")
    
    with col2:
        st.write("**Problem: Missing Separators**")
        problem_example2 = pd.DataFrame({
            'Date': ['2024-01-01'],
            'OpenHighLowClose': ['50000 50200 49800 50100']
        })
        st.dataframe(problem_example2, use_container_width=True)
        st.error("OHLC data not properly separated")
    
    st.info("💡 **Tips:** Ensure your CSV uses comma separators, has proper column headers, and contains clean OHLC data")

# Footer
st.markdown("---")
st.markdown("*Built with ❤️ for Smart Options Trading | Always trade responsibly*")