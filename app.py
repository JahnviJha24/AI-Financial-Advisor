from flask import Flask, render_template, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
import pandas as pd
import json
from datetime import datetime, timedelta
import sys
import os

#change
# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from models.stock_predictor import StockPredictor
from models.sentiment_analyzer import SentimentAnalyzer

app = Flask(__name__)

# Initialize models
predictor = StockPredictor()
sentiment_analyzer = SentimentAnalyzer()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_stock():
    """Main API endpoint for stock analysis"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper().strip()
        
        if not symbol:
            return jsonify({'error': 'Please provide a stock symbol'}), 400
        
        # Get basic stock info
        stock_info = predictor.get_stock_info(symbol)
        
        # Get stock price prediction
        prediction_result, prediction_error = predictor.predict_price(symbol)
        
        if prediction_error:
            return jsonify({'error': f'Prediction error: {prediction_error}'}), 400
        
        # Get sentiment analysis
        sentiment_data = sentiment_analyzer.get_news_sentiment(symbol, stock_info['name'])
        
        # Get historical data for charts
        historical_data = get_historical_data(symbol)
        
        response = {
            'symbol': symbol,
            'stock_info': stock_info,
            'prediction': prediction_result,
            'sentiment': sentiment_data,
            'historical_data': historical_data,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/historical/<symbol>')
def get_historical_data(symbol, period='6mo'):
    """Get historical stock data for charts"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        # Convert to format suitable for Chart.js
        chart_data = {
            'labels': [date.strftime('%Y-%m-%d') for date in data.index],
            'prices': data['Close'].round(2).tolist(),
            'volumes': data['Volume'].tolist()
        }
        
        return chart_data
        
    except Exception as e:
        print(f"Historical data error: {e}")
        return {'labels': [], 'prices': [], 'volumes': []}

@app.route('/api/popular-stocks')
def get_popular_stocks():
    """Get list of popular stocks for suggestions"""
    popular_stocks = [
        {'symbol': 'AAPL', 'name': 'Apple Inc.'},
        {'symbol': 'GOOGL', 'name': 'Alphabet Inc.'},
        {'symbol': 'MSFT', 'name': 'Microsoft Corporation'},
        {'symbol': 'AMZN', 'name': 'Amazon.com Inc.'},
        {'symbol': 'TSLA', 'name': 'Tesla Inc.'},
        {'symbol': 'META', 'name': 'Meta Platforms Inc.'},
        {'symbol': 'NVDA', 'name': 'NVIDIA Corporation'},
        {'symbol': 'NFLX', 'name': 'Netflix Inc.'},
        {'symbol': 'DIS', 'name': 'The Walt Disney Company'},
        {'symbol': 'UBER', 'name': 'Uber Technologies Inc.'}
    ]
    
    return jsonify(popular_stocks)

@app.route('/api/validate-symbol/<symbol>')
def validate_symbol(symbol):
    """Validate if a stock symbol exists"""
    try:
        stock = yf.Ticker(symbol.upper())
        info = stock.info
        
        # Check if we got valid data
        if 'symbol' in info and info.get('symbol'):
            return jsonify({
                'valid': True,
                'name': info.get('longName', symbol.upper()),
                'symbol': symbol.upper()
            })
        else:
            return jsonify({'valid': False})
            
    except Exception as e:
        print(f"Symbol validation error: {e}")
        return jsonify({'valid': False})

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Create __init__.py in models directory
    init_file = os.path.join('models', '__init__.py')
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write('')
    
    print("🚀 Starting AI Financial Advisor...")
    
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get('PORT', 5000))
    
    # For local development
    if os.environ.get('ENVIRONMENT') == 'development':
        print(f"📊 Access your app at: http://127.0.0.1:{port}")
        app.run(debug=True, host='127.0.0.1', port=port)
    else:
        # For production deployment (Render)
        print(f"📊 Starting production server on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False)