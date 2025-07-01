import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def fetch_stock_data(self, symbol, period="1y"):
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def create_features(self, data):
        """Create technical indicators as features"""
        df = data.copy()
        
        # Technical indicators
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'] = self.calculate_macd(df['Close'])
        df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
        
        # Price features
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Price_Volume'] = df['Close'] * df['Volume']
        
        # Lag features
        for i in range(1, 6):
            df[f'Close_Lag_{i}'] = df['Close'].shift(i)
            df[f'Volume_Lag_{i}'] = df['Volume'].shift(i)
        
        return df
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        if len(prices) < window:
            return prices * 0  # Return series of zeros with same index
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        # Avoid division by zero
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral RSI value
    
    def calculate_macd(self, prices, fast=12, slow=26):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def prepare_training_data(self, data, target_days=1):
        """Prepare data for training"""
        df = self.create_features(data)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Define features
        feature_columns = [
            'Open', 'High', 'Low', 'Volume', 'SMA_10', 'SMA_30', 
            'RSI', 'MACD', 'Volume_MA', 'Price_Change', 'High_Low_Ratio',
            'Price_Volume'
        ] + [f'Close_Lag_{i}' for i in range(1, 6)] + [f'Volume_Lag_{i}' for i in range(1, 6)]
        
        # Create target (future price)
        target_col = f'Target_{target_days}d'
        df[target_col] = df['Close'].shift(-target_days)
        
        # Remove last rows where target is NaN
        df = df.dropna()
        
        X = df[feature_columns]
        y = df[target_col]
        
        return X, y, df
    
    def train_model(self, symbol, period="2y"):
        """Train the prediction model"""
        print(f"Training model for {symbol}...")
        
        # Fetch data
        data = self.fetch_stock_data(symbol, period)
        if data is None or len(data) < 100:
            return False, "Insufficient data for training"
        
        # Prepare training data
        X, y, df = self.prepare_training_data(data)
        
        if len(X) < 50:
            return False, "Insufficient processed data for training"
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.is_trained = True
        self.feature_columns = X.columns.tolist()
        
        return True, {
            'mse': mse,
            'r2': r2,
            'accuracy': max(0, r2 * 100)  # Convert to percentage
        }
    
    def predict_price(self, symbol, days_ahead=1):
        """Predict future stock price"""
        if not self.is_trained:
            train_success, train_result = self.train_model(symbol)
            if not train_success:
                return None, train_result
        
        # Fetch recent data
        data = self.fetch_stock_data(symbol, period="3mo")
        if data is None:
            return None, "Could not fetch recent data"
        
        # Prepare features
        df = self.create_features(data)
        df = df.dropna()
        
        if len(df) == 0:
            return None, "No valid data for prediction"
        
        # Get latest features
        if len(df) < 1:
            return None, "Insufficient data for feature extraction"
            
        latest_features = df[self.feature_columns].iloc[-1:].values
        latest_features_scaled = self.scaler.transform(latest_features)
        
        # Make prediction
        predicted_price = self.model.predict(latest_features_scaled)[0]
        current_price = data['Close'].iloc[-1]
        
        # Calculate prediction confidence and direction
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        direction = "UP" if price_change > 0 else "DOWN"
        confidence = min(95, abs(price_change_pct) * 10)  # Simple confidence metric
        
        return {
            'current_price': round(current_price, 2),
            'predicted_price': round(predicted_price, 2),
            'price_change': round(price_change, 2),
            'price_change_pct': round(price_change_pct, 2),
            'direction': direction,
            'confidence': round(confidence, 1)
        }, None
    
    def get_stock_info(self, symbol):
        """Get basic stock information"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            return {
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A')
            }
        except (KeyError, ValueError) as e:
            print(f"Error getting stock info for {symbol}: {e}")
            return {
                'name': symbol,
                'sector': 'N/A',
                'industry': 'N/A',
                'market_cap': 'N/A',
                'pe_ratio': 'N/A'
            }