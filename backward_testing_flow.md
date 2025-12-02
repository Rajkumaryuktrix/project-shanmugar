# Backward Testing Engine - Code Flow Visualization

## Main Class Structure and Data Flow

```mermaid
classDiagram
    class CandleData {
        +timestamp: np.ndarray
        +open: np.ndarray
        +high: np.ndarray
        +low: np.ndarray
        +close: np.ndarray
        +volume: np.ndarray
        +oi: np.ndarray
        +signal: np.ndarray
        +sl: Optional[np.ndarray]
        +tp: Optional[np.ndarray]
        +from_list(candles, signals, sl_levels, tp_levels) classmethod
    }

    class BacktestingEngine {
        +data: CandleData
        +initial_balance: float
        +volume: float
        +max_position_size: float
        +risk_per_trade: float
        +leverage: float
        +slippage_value: float
        +min_position_size_ratio: float
        +min_fill_ratio: float
        +slippage_model: str
        +position_sizing_method: str
        +segment: str
        +max_open_positions: int
        +allow_partial_fills: bool
        +parallel_opening: bool
        +use_sltp: bool
        +force_signal_exits: bool
        +strict_sltp: bool
        +tax_calculator: IndianTradeCostCalculator
        +trades: List
        +balance_history: np.ndarray
        +equity_curve: np.ndarray
        +drawdown_curve: np.ndarray
        +__init__(candle_data, **kwargs)
        +_calculate_pnl_vectorized(entry_prices, exit_prices, volumes, position_types, commissions) static
        +calculate_position_size(price, volume)
        +apply_slippage(price, volume, direction)
        +run_backtest() Dict
    }

    class IndianTradeCostCalculator {
        +Trade(segment, buy_value, sell_value)
        +calc_charges(trade)
    }

    CandleData --> BacktestingEngine : provides data
    BacktestingEngine --> IndianTradeCostCalculator : uses for cost calculation
```

## Main Execution Flow

```mermaid
flowchart TD
    A[Start: Create CandleData] --> B[Initialize BacktestingEngine]
    B --> C[Validate Parameters]
    C --> D[Setup Tax Calculator]
    D --> E[Initialize Arrays & Variables]
    
    E --> F[Run Backtest]
    F --> G[Process Each Candle]
    
    G --> H{Check Balance Sufficient?}
    H -->|No| I[Close All Positions]
    H -->|Yes| J[Process Open Positions]
    
    J --> K{Check Exit Conditions}
    K -->|Signal Exit| L[Close by Signal]
    K -->|SL/TP Exit| M[Close by SL/TP]
    K -->|No Exit| N[Check Entry Signals]
    
    L --> O[Calculate PnL & Update Balance]
    M --> O
    O --> N
    
    N --> P{New Entry Signal?}
    P -->|Yes| Q[Calculate Position Size]
    P -->|No| R[Next Candle]
    
    Q --> S{Sufficient Margin?}
    S -->|Yes| T[Create Position]
    S -->|No| R
    
    T --> U[Apply Slippage]
    U --> V[Calculate Commission]
    V --> W[Add to Open Positions]
    W --> R
    
    R --> X{More Candles?}
    X -->|Yes| G
    X -->|No| Y[Close Remaining Positions]
    
    Y --> Z[Calculate Final Results]
    Z --> AA[Return Results Dictionary]
    
    I --> AA
```

## Position Management Flow

```mermaid
flowchart TD
    A[Entry Signal Detected] --> B[Determine Position Type]
    B --> C[Apply Slippage Model]
    
    C --> D{Position Sizing Method}
    D -->|Fixed| E[Use Fixed Volume]
    D -->|Risk Based| F[Calculate Risk-Based Size]
    
    E --> G[Check Min Position Size]
    F --> G
    
    G --> H{Sufficient Size?}
    H -->|No| I[Skip Trade]
    H -->|Yes| J[Calculate Required Margin]
    
    J --> K{Sufficient Balance?}
    K -->|No| L[Skip Trade]
    K -->|Yes| M[Calculate Commission]
    
    M --> N[Create Position Record]
    N --> O[Add to Open Positions]
    O --> P[Update Balance]
```

## Exit Condition Logic

```mermaid
flowchart TD
    A[Process Each Open Position] --> B{Force Signal Exits?}
    
    B -->|Yes| C{Signal Exit Present?}
    B -->|No| D[Check All Exit Types]
    
    C -->|Yes| E[Close by Signal]
    C -->|No| F[Skip Position]
    
    D --> G{Signal Exit?}
    D --> H{SL/TP Enabled?}
    
    G -->|Yes| E
    G -->|No| I{SL/TP Exit?}
    
    H -->|Yes| I
    H -->|No| F
    
    I --> J{Long Position?}
    I --> K{Short Position?}
    
    J --> L{Price <= SL or >= TP?}
    K --> M{Price >= SL or <= TP?}
    
    L -->|Yes| N[Close by SL/TP]
    L -->|No| F
    M -->|Yes| N
    M -->|No| F
    
    E --> O[Calculate PnL]
    N --> O
    O --> P[Update Balance]
    P --> Q[Add to Trade History]
```

## Data Processing Pipeline

```mermaid
flowchart LR
    A[Raw Candle Data] --> B[Data Validation]
    B --> C[Convert to Numpy Arrays]
    C --> D[Extract Timestamps]
    D --> E[Convert Price Data]
    E --> F[Process Signals]
    F --> G[Process SL/TP Levels]
    G --> H[CandleData Object]
    
    H --> I[BacktestingEngine]
    I --> J[Vectorized Operations]
    J --> K[Trade Execution]
    K --> L[Results Collection]
```

## Key Methods and Their Purposes

```mermaid
graph TB
    subgraph "Data Management"
        A[CandleData.from_list] --> B[Data Validation & Conversion]
        B --> C[Numpy Array Creation]
    end
    
    subgraph "Position Management"
        D[calculate_position_size] --> E[Risk-Based or Fixed Sizing]
        F[apply_slippage] --> G[Price Adjustment]
    end
    
    subgraph "Trade Execution"
        H[_calculate_pnl_vectorized] --> I[Vectorized PnL Calculation]
        J[run_backtest] --> K[Main Execution Loop]
    end
    
    subgraph "Cost Calculation"
        L[IndianTradeCostCalculator] --> M[Commission & Tax Calculation]
    end
    
    C --> K
    E --> K
    G --> K
    I --> K
    M --> K
```

## Error Handling and Validation Flow

```mermaid
flowchart TD
    A[Input Validation] --> B{Valid Parameters?}
    B -->|No| C[Raise ValueError]
    B -->|Yes| D[Engine Initialization]
    
    D --> E{SL/TP Data Present?}
    E -->|No| F{Strict SLTP Mode?}
    E -->|Yes| G[Continue with SL/TP]
    
    F -->|Yes| H[Raise ValueError]
    F -->|No| I[Continue without SL/TP]
    
    G --> J[Main Backtest Loop]
    I --> J
    
    J --> K{Balance Sufficient?}
    K -->|No| L[Close Positions & Stop]
    K -->|Yes| M[Continue Processing]
    
    M --> N{Valid Trade Conditions?}
    N -->|No| O[Skip Trade]
    N -->|Yes| P[Execute Trade]
    
    P --> Q{Exception in Trade?}
    Q -->|Yes| R[Log Error & Continue]
    Q -->|No| S[Update Results]
```

## Results Structure

```mermaid
graph LR
    A[Backtest Results] --> B[Trades List]
    A --> C[Balance History]
    A --> D[Equity Curve]
    A --> E[Drawdown Curve]
    A --> F[Strategy Config]
    
    B --> G[Trade Records]
    G --> H[Entry/Exit Times]
    G --> I[Prices & PnL]
    G --> J[Commission Details]
    
    F --> K[Engine Parameters]
    F --> L[Execution Statistics]
``` 