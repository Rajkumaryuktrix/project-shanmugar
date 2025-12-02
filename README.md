# 🚀 Autonomous AI Trader - MVP → Scale

A sophisticated algorithmic trading platform built with Python, featuring real-time market data processing, strategy backtesting, and automated trading capabilities with seamless migration paths from MVP to enterprise scale.

## 🎯 Project Overview

This project implements a **two-track architecture** that allows you to:
- **Ship Fast**: Deploy the MVP quickly with proven, lightweight technologies
- **Scale Seamlessly**: Migrate to enterprise-grade solutions without API changes
- **Maintain Compliance**: Built-in audit trails, data quality, and observability from day one

## 🏗️ Architecture Principles

- **OSS-first**: Reduce vendor lock-in
- **Thin adapters**: Wrap components behind interfaces for easy swapping
- **Two-track**: MVP now, scale later with clear migration paths
- **Compliance-ready**: Data quality, observability, audit trail from day one

## 🚀 Features

### Core Capabilities
- **Real-time Market Data Processing**
  - Live data streaming from multiple brokers
  - Historical data management and caching
  - Efficient candle data processing

- **Strategy Framework**
  - Modular strategy implementation
  - Moving average crossover strategy
  - Customizable strategy parameters
  - Strategy performance analytics

- **Backtesting Engine**
  - Historical data simulation
  - Performance metrics calculation
  - Risk analysis and reporting
  - Strategy optimization tools

- **LLM Integration**
  - AI-powered trading decisions
  - Natural language strategy description
  - Automated market analysis

### MVP Stack (Option A)
- **Frontend**: Streamlit
- **API**: FastAPI
- **Database**: Supabase (Postgres)
- **Event Bus**: Redis Streams
- **LLM**: Groq/OpenAI
- **Backtesting**: VectorBT
- **Orchestration**: Airflow

### Scale Stack (Option B)
- **Frontend**: Next.js + Streamlit (ops)
- **API**: FastAPI + Go/Rust (hot paths)
- **Database**: Postgres + Keycloak
- **Event Bus**: Kafka/Redpanda
- **LLM**: vLLM (local models)
- **Backtesting**: NautilusTrader
- **Orchestration**: Dagster/Prefect

## 📋 Prerequisites

- Python 3.8 or higher
- Trading account with API access
- Sufficient system memory (8GB+ recommended)
- Docker and Docker Compose (for local development)

## 🛠️ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/autonomous-ai-trader.git
cd autonomous-ai-trader
```

### 2. Setup Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Unix/MacOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit with your configuration
# UPSTOX_API_KEY=your_api_key
# UPSTOX_API_SECRET=your_api_secret
```

### 4. Run MVP Version
```bash
# Start the application
python src/cli/main.py

# Or run specific components
python src/cli/main.py backtest --strategy moving_average
```

## 📁 Project Structure

```
autonomous-ai-trader/
├── 📁 src/                        # Source code
│   ├── 📁 adapters/              # Adapter layer for easy migrations
│   ├── 📁 strategies/            # Trading strategies
│   ├── 📁 backtesting/           # Backtesting engine
│   ├── 📁 llm/                   # LLM integration
│   └── 📁 api/                   # FastAPI application
├── 📁 infrastructure/             # Infrastructure configurations
│   ├── 📁 mvp/                   # MVP infrastructure
│   └── 📁 scale/                 # Scale infrastructure
├── 📁 migrations/                 # Migration scripts and guides
└── 📁 docs/                       # Comprehensive documentation
```

## 🔄 Migration Paths

### When to Migrate
- **Kafka**: >2M events/day or >48h replay required
- **Postgres+Keycloak**: SSO, data residency, or RLS complexity
- **NautilusTrader**: Need tick-level fills, order book sim, latency modeling
- **vLLM**: LLM spend >15% infra or privacy requirements

### Migration Process
1. **Zero API Changes**: All migrations happen behind adapters
2. **Gradual Rollout**: Migrate component by component
3. **Easy Rollback**: Built-in rollback capabilities
4. **Validation**: Automated compatibility checks

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run with coverage
pytest --cov=src --cov-report=html
```

## 📊 Performance Analysis

The system generates comprehensive performance reports including:
- Total returns and Sharpe ratio
- Maximum drawdown and win rate
- Risk-adjusted metrics
- Strategy comparison analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📝 Development Guidelines

- Follow PEP 8 style guide
- Write unit tests for new features
- Update documentation
- Use meaningful commit messages
- Maintain backward compatibility

## 🔒 Security

- Never commit API keys or sensitive credentials
- Use environment variables for configuration
- Implement proper error handling
- Follow security best practices

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/autonomous-ai-trader/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/autonomous-ai-trader/discussions)
- **Documentation**: [Project Wiki](https://github.com/yourusername/autonomous-ai-trader/wiki)

## 🔄 Updates

Stay updated with the latest changes:
- Follow the repository
- Check the [CHANGELOG.md](CHANGELOG.md)
- Review [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)

## 🎯 Roadmap

- [ ] MVP Release (Q4 2025)
- [ ] Scale Migration Tools (Q1 2026)
- [ ] Enterprise Features (Q3 2026)
- [ ] Multi-Broker Support (Q1 2026)

---

**Built with ❤️ for the algorithmic trading community**

*This project follows the [Cursor Agent Elite Protocol](https://github.com/cursor-ai/cursor-agent-elite-protocol) for professional-grade development standards.*
