# Shanmugaa - Algorithmic Trading Project

This project implements an algorithmic trading system with the following key components:

## Project Structure

```
src/
├── broker_module/
│   └── upstox/
│       ├── data/
│       │   └── CandleData.py
│       ├── utils/
│       │   └── InstrumentKeyFinder.py
│       └── __init__.py
└── strategy_module/
    ├── strategies/
    │   └── moving_average_strategy.py
    ├── utils/
    │   ├── backward_testing.py
    │   ├── strategy_tester.py
    │   └── __init__.py
    ├── results/
    ├── core_engine.py
    └── strategy_cli.py
```

## Features

- Historical data fetching and management
- Strategy implementation and testing
- Backtesting framework
- Performance optimization
- Results analysis and reporting

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure your Upstox API credentials
4. Run the strategy tester:
   ```bash
   python src/strategy_module/strategy_cli.py
   ```

## Development

- Follow PEP 8 style guide
- Write unit tests for new features
- Update documentation as needed
- Use meaningful commit messages

## License

MIT License
