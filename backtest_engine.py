"""
Production-grade backtesting engine for options strategies.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from scipy.stats import norm

@dataclass
class Position:
    """Represents a single option position."""
    option_symbol: str
    call_put: str  # 'C' or 'P'
    strike: float
    expiration: pd.Timestamp
    quantity: int  # Positive for long, negative for short
    entry_price: float
    entry_date: pd.Timestamp
    underlying_price_entry: float
    delta: float
    gamma: float
    theta: float
    vega: float
    
    exit_price: Optional[float] = None
    exit_date: Optional[pd.Timestamp] = None
    realized_pnl: Optional[float] = None
    
    def mark_to_market(self, current_price: float) -> float:
        """Compute unrealized P&L."""
        return self.quantity * (current_price - self.entry_price)
    
    def close_position(self, exit_price: float, exit_date: pd.Timestamp):
        """Close position and compute realized P&L."""
        self.exit_price = exit_price
        self.exit_date = exit_date
        self.realized_pnl = self.quantity * (exit_price - self.entry_price)


@dataclass
class Trade:
    """Represents a complete strategy trade (can include multiple legs)."""
    trade_id: str
    entry_date: pd.Timestamp
    positions: List[Position]
    strategy_name: str
    
    exit_date: Optional[pd.Timestamp] = None
    total_cost: float = 0.0
    total_pnl: float = 0.0
    is_closed: bool = False
    
    def __post_init__(self):
        """Calculate initial trade cost."""
        self.total_cost = sum(p.quantity * p.entry_price for p in self.positions)
    
    def current_value(self, current_prices: Dict[str, float]) -> float:
        """Compute current value of all positions."""
        total = 0.0
        for pos in self.positions:
            if pos.option_symbol in current_prices:
                total += pos.quantity * current_prices[pos.option_symbol]
        return total
    
    def close_trade(self, exit_prices: Dict[str, float], exit_date: pd.Timestamp):
        """Close all positions in the trade."""
        self.exit_date = exit_date
        
        for pos in self.positions:
            if pos.option_symbol in exit_prices:
                pos.close_position(exit_prices[pos.option_symbol], exit_date)
        
        self.total_pnl = sum(p.realized_pnl for p in self.positions if p.realized_pnl)
        self.is_closed = True


class BacktestEngine:
    """
    Options strategy backtesting engine with realistic transaction costs.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_per_contract: float = 0.65,
        slippage_bps: float = 5.0,  # 5 basis points
        min_spread_pct: float = 0.01  # Minimum 1% spread
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission_per_contract = commission_per_contract
        self.slippage_bps = slippage_bps
        self.min_spread_pct = min_spread_pct
        
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.transaction_costs: List[float] = []
    
    def apply_transaction_costs(
        self,
        quantity: int,
        bid: float,
        ask: float,
        is_entry: bool = True
    ) -> Tuple[float, float]:
        """
        Apply realistic transaction costs.
        
        Args:
            quantity: Number of contracts (positive for long)
            bid: Bid price
            ask: Ask price
            is_entry: True if entering position, False if exiting
        
        Returns:
            (execution_price, total_cost)
        """
        # Determine if we're buying or selling
        is_buying = (quantity > 0 and is_entry) or (quantity < 0 and not is_entry)
        
        # Base price (take ask when buying, bid when selling)
        if is_buying:
            base_price = ask
        else:
            base_price = bid
        
        # Add slippage
        slippage = base_price * (self.slippage_bps / 10000.0)
        if is_buying:
            execution_price = base_price + slippage
        else:
            execution_price = base_price - slippage
        
        # Ensure minimum spread
        spread = ask - bid
        min_spread = base_price * self.min_spread_pct
        if spread < min_spread:
            if is_buying:
                execution_price += (min_spread - spread) / 2
            else:
                execution_price -= (min_spread - spread) / 2
        
        # Commission
        commission = abs(quantity) * self.commission_per_contract
        
        # Total cost (negative for credits received)
        total_cost = abs(quantity) * execution_price + commission
        
        return execution_price, total_cost
    
    def open_trade(
        self,
        trade_date: pd.Timestamp,
        strategy_name: str,
        legs: List[Dict]
    ) -> Optional[Trade]:
        """
        Open a multi-leg strategy trade.
        
        Args:
            trade_date: Entry date
            strategy_name: Name of strategy
            legs: List of dictionaries with keys:
                  'option_symbol', 'call_put', 'strike', 'expiration',
                  'quantity', 'bid', 'ask', 'underlying_price',
                  'delta', 'gamma', 'theta', 'vega'
        
        Returns:
            Trade object if successful, None if insufficient capital
        """
        positions = []
        total_cost = 0.0
        
        # Create positions and calculate costs
        for leg in legs:
            exec_price, leg_cost = self.apply_transaction_costs(
                leg['quantity'],
                leg['bid'],
                leg['ask'],
                is_entry=True
            )
            
            # Adjust for short positions (receive credit)
            if leg['quantity'] < 0:
                leg_cost = -leg_cost
            
            total_cost += leg_cost
            
            position = Position(
                option_symbol=leg['option_symbol'],
                call_put=leg['call_put'],
                strike=leg['strike'],
                expiration=pd.Timestamp(leg['expiration']),
                quantity=leg['quantity'],
                entry_price=exec_price,
                entry_date=trade_date,
                underlying_price_entry=leg['underlying_price'],
                delta=leg['delta'],
                gamma=leg.get('gamma', 0.0),
                theta=leg.get('theta', 0.0),
                vega=leg.get('vega', 0.0)
            )
            positions.append(position)
        
        # Check capital requirements (for debit trades)
        if total_cost > 0 and total_cost > self.capital:
            print(f"Insufficient capital: need {total_cost:.2f}, have {self.capital:.2f}")
            return None
        
        # Create trade
        trade_id = f"{strategy_name}_{trade_date.strftime('%Y%m%d_%H%M%S')}"
        trade = Trade(
            trade_id=trade_id,
            entry_date=trade_date,
            positions=positions,
            strategy_name=strategy_name
        )
        
        # Update capital
        self.capital -= total_cost
        self.transaction_costs.append(total_cost)
        
        # Record trade
        self.trades.append(trade)
        self.equity_curve.append((trade_date, self.capital))
        
        return trade
    
    def close_trade(
        self,
        trade: Trade,
        exit_date: pd.Timestamp,
        exit_prices: Dict[str, Tuple[float, float]]  # {symbol: (bid, ask)}
    ):
        """
        Close an existing trade.
        
        Args:
            trade: Trade to close
            exit_date: Exit date
            exit_prices: Dictionary mapping option_symbol to (bid, ask) tuple
        """
        if trade.is_closed:
            return
        
        total_proceeds = 0.0
        execution_prices = {}
        
        # Close each position
        for pos in trade.positions:
            if pos.option_symbol not in exit_prices:
                print(f"Warning: No exit price for {pos.option_symbol}")
                continue
            
            bid, ask = exit_prices[pos.option_symbol]
            
            # Reverse the position (if long, we sell; if short, we buy back)
            exit_quantity = -pos.quantity
            exec_price, leg_proceeds = self.apply_transaction_costs(
                exit_quantity,
                bid,
                ask,
                is_entry=False
            )
            
            # For closing, if we're selling (closing long), we receive proceeds
            # If buying back (closing short), we pay
            if pos.quantity > 0:  # Closing long
                total_proceeds += leg_proceeds
            else:  # Closing short
                total_proceeds -= leg_proceeds
            
            execution_prices[pos.option_symbol] = exec_price
        
        # Close trade
        trade.close_trade(execution_prices, exit_date)
        
        # Update capital
        self.capital += total_proceeds
        self.equity_curve.append((exit_date, self.capital))
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics.
        """
        if len(self.equity_curve) < 2:
            return {}
        
        # Convert equity curve to DataFrame
        df = pd.DataFrame(self.equity_curve, columns=['date', 'equity'])
        df = df.set_index('date').sort_index()
        
        # Returns
        df['returns'] = df['equity'].pct_change()
        
        # Remove NaN
        returns = df['returns'].dropna()
        
        if len(returns) == 0:
            return {}
        
        # Annualization factor (assume daily data)
        n_periods_per_year = 252
        
        # Total return
        total_return = (df['equity'].iloc[-1] / self.initial_capital) - 1
        
        # CAGR
        n_years = (df.index[-1] - df.index[0]).days / 365.25
        if n_years > 0:
            cagr = (1 + total_return) ** (1 / n_years) - 1
        else:
            cagr = 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(n_periods_per_year)
        
        # Sharpe ratio (assuming 0 risk-free rate)
        sharpe = (returns.mean() * n_periods_per_year) / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(n_periods_per_year)
        sortino = (returns.mean() * n_periods_per_year) / downside_std if downside_std > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        closed_trades = [t for t in self.trades if t.is_closed]
        if closed_trades:
            winning_trades = sum(1 for t in closed_trades if t.total_pnl > 0)
            win_rate = winning_trades / len(closed_trades)
        else:
            win_rate = 0
        
        # Profit factor
        gross_profit = sum(t.total_pnl for t in closed_trades if t.total_pnl > 0)
        gross_loss = abs(sum(t.total_pnl for t in closed_trades if t.total_pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Total transaction costs
        total_costs = sum(self.transaction_costs)
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'n_trades': len(closed_trades),
            'total_transaction_costs': total_costs,
            'final_capital': df['equity'].iloc[-1]
        }


# Example: Risk Reversal Strategy
class RiskReversalStrategy:
    """
    Long call / Short put strategy (bullish risk reversal).
    """
    
    @staticmethod
    def find_25_delta_options(
        df: pd.DataFrame,
        expiration: pd.Timestamp,
        target_delta: float = 0.25
    ) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
        """
        Find 25-delta call and put for risk reversal.
        """
        # Filter by expiration
        exp_df = df[df['expiration_date'] == expiration].copy()
        
        if len(exp_df) == 0:
            return None, None
        
        # Find 25-delta call (closest to +0.25)
        calls = exp_df[exp_df['call_put'] == 'C']
        if len(calls) > 0:
            calls['delta_diff'] = np.abs(calls['delta'] - target_delta)
            call_25d = calls.loc[calls['delta_diff'].idxmin()]
        else:
            call_25d = None
        
        # Find 25-delta put (closest to -0.25)
        puts = exp_df[exp_df['call_put'] == 'P']
        if len(puts) > 0:
            puts['delta_diff'] = np.abs(puts['delta'] + target_delta)
            put_25d = puts.loc[puts['delta_diff'].idxmin()]
        else:
            put_25d = None
        
        return call_25d, put_25d
    
    @staticmethod
    def enter_trade(
        engine: BacktestEngine,
        df: pd.DataFrame,
        entry_date: pd.Timestamp,
        expiration: pd.Timestamp,
        quantity: int = 1
    ) -> Optional[Trade]:
        """
        Enter risk reversal trade.
        """
        call_25d, put_25d = RiskReversalStrategy.find_25_delta_options(df, expiration)
        
        if call_25d is None or put_25d is None:
            return None
        
        # Construct legs
        legs = [
            {
                'option_symbol': call_25d['option_symbol'],
                'call_put': 'C',
                'strike': call_25d['price_strike'],
                'expiration': call_25d['expiration_date'],
                'quantity': quantity,  # Long call
                'bid': call_25d['Bid'],
                'ask': call_25d['Ask'],
                'underlying_price': call_25d['underlying_price'],
                'delta': call_25d['delta'],
                'gamma': call_25d['gamma'],
                'theta': call_25d['theta'],
                'vega': call_25d['vega']
            },
            {
                'option_symbol': put_25d['option_symbol'],
                'call_put': 'P',
                'strike': put_25d['price_strike'],
                'expiration': put_25d['expiration_date'],
                'quantity': -quantity,  # Short put
                'bid': put_25d['Bid'],
                'ask': put_25d['Ask'],
                'underlying_price': put_25d['underlying_price'],
                'delta': put_25d['delta'],
                'gamma': put_25d['gamma'],
                'theta': put_25d['theta'],
                'vega': put_25d['vega']
            }
        ]
        
        return engine.open_trade(entry_date, "Risk_Reversal", legs)


# Example usage
if __name__ == "__main__":
    # Initialize backtest engine
    engine = BacktestEngine(initial_capital=100000)
    
    print(f"Starting capital: ${engine.capital:,.2f}")
    print("\nBacktest engine ready for strategy execution.")
