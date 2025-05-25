# Trading Strategy Tester

A web application for testing trading strategies using historical data from Upstox. The application consists of a FastAPI backend and a Streamlit frontend.

## Features

- Test trading strategies with historical data
- Visualize strategy performance with interactive charts
- Analyze trade distributions and metrics
- Support for multiple timeframes and symbols
- Real-time strategy testing with customizable parameters

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Upstox API credentials:
Create a `.env` file in the root directory and add your Upstox API credentials:
```
UPSTOX_API_KEY=your_api_key
UPSTOX_API_SECRET=your_api_secret
```

## Running the Application

1. Start the FastAPI backend:
```bash
uvicorn api:app --reload
```

2. In a new terminal, start the Streamlit frontend:
```bash
streamlit run app.py
```

3. Open your browser and navigate to:
```
http://localhost:8501
```

## Usage

1. Select a trading symbol from the dropdown menu
2. Choose the timeframe for testing
3. Set the testing period in months
4. Configure strategy parameters:
   - Short and long window for moving averages
   - Investing amount
   - Leverage
   - Volume
   - Commission
5. Click "Test Strategy" to run the backtest
6. View the results in the main panel:
   - Strategy metrics and statistics
   - Equity curve
   - Trade distribution analysis
   - Trade history

## Project Structure

```
project_root/
├── api.py                      # FastAPI backend implementation
├── app.py                      # Streamlit frontend implementation
├── strategy_tester.py          # Core strategy testing logic
├── strategies/                 # Trading strategies package
│   ├── __init__.py            # Package initialization
│   └── moving_average_strategy.py  # Moving average crossover strategy
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 