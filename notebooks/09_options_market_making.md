# Step 10: Options Market Making

Options market making means quoting two-sided markets:

```text
bid = price where the market maker buys
ask = price where the market maker sells
```

The market maker earns spread when order flow is balanced, but takes inventory,
Gamma, Vega, Theta, and adverse-selection risk.

## 1. Fair Value

The first quote anchor is model fair value:

```text
fair value = Black-Scholes price
```

In production, fair value would use the desk volatility surface, dividends,
rates, borrow assumptions, and model adjustments. Our first engine uses the
Black-Scholes model from Step 3.

## 2. Bid/Ask Spread

A simple quote is:

```text
bid = fair value - spread / 2
ask = fair value + spread / 2
```

But real spreads widen when risk rises.

In this project, spread increases with:

- Base spread
- Inventory pressure
- Gamma exposure
- Vega exposure
- Adverse-selection buffer

## 3. Inventory Management

Inventory is the market maker's current option position.

Positive inventory:

```text
long options
```

Negative inventory:

```text
short options
```

If the market maker is already long options, the engine skews the quote lower
to encourage reducing inventory. If the market maker is short options, the
quote skews higher to encourage buying inventory back.

## 4. Gamma Exposure

Gamma exposure is:

```text
option inventory * option gamma
```

High Gamma means Delta changes quickly. A short-Gamma market maker may need to
rebalance aggressively during large spot moves.

## 5. Theta Decay

Theta exposure is:

```text
option inventory * option theta
```

Long options usually have negative Theta. Short options usually collect Theta,
but this is compensation for carrying convexity and volatility risk.

## 6. Vega Exposure

Vega exposure is:

```text
option inventory * option vega
```

If the market maker is long Vega, rising implied volatility helps. If short
Vega, volatility spikes hurt.

## 7. Adverse Selection

Adverse selection means the other side may know something or may be trading
when the market is about to move.

Examples:

- A customer buys calls just before news.
- A trader lifts offers before volatility reprices higher.
- Someone sells options right before a volatility collapse.

Market makers charge wider spreads when adverse-selection risk is high.

## 8. Code Entry Point

The implementation lives in:

```text
analytics/market_making.py
```

Main classes:

```text
MarketMakingEngine
MarketMakingConfig
InventoryState
OptionQuote
```

Basic usage:

```python
from analytics.market_making import InventoryState, MarketMakingEngine
from utils.instruments import EuropeanOption, OptionType
from utils.market import MarketEnvironment

option = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
market = MarketEnvironment(spot=100.0, risk_free_rate=0.05, volatility=0.2)

engine = MarketMakingEngine(option=option, market=market)
inventory = InventoryState(option_position=0.0, cash=0.0)

quote = engine.quote(inventory)

print(quote.bid)
print(quote.ask)
print(quote.gamma_exposure)
print(quote.theta_decay)

inventory = engine.execute_trade(
    inventory=inventory,
    side="sell",
    quantity=1.0,
    quote=quote,
)
```

## 9. Professional Interpretation

Market making is not just "buy low, sell high." The spread compensates for:

- Inventory risk
- Hedging costs
- Gamma losses during large moves
- Volatility repricing
- Stale quote risk
- Adverse selection

The quote is a control system. It changes prices and sizes to manage risk while
still providing liquidity.

