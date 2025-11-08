"""
Data preprocessing and feature engineering for options analytics.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
import warnings

class OptionsPreprocessor:
    """
    Clean and enrich options chain data with derived features.
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Args:
            risk_free_rate: Annual risk-free rate (default 5%)
        """
        self.risk_free_rate = risk_free_rate
    
    def parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse and normalize date fields.
        Assumes ISO format strings or similar.
        """
        df = df.copy()
        
        # Parse c_date (capture timestamp)
        if df['c_date'].dtype == 'object':
            df['c_date'] = pd.to_datetime(df['c_date'], errors='coerce')
        
        # Parse expiration_date
        if df['expiration_date'].dtype == 'object':
            df['expiration_date'] = pd.to_datetime(df['expiration_date'], errors='coerce')
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['c_date', 'expiration_date'])
        
        return df
    
    def compute_time_to_expiry(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute time to expiry in years (T).
        """
        df = df.copy()
        
        # Time difference in days
        df['days_to_expiry'] = (df['expiration_date'] - df['c_date']).dt.days
        
        # Convert to years (use trading days: 252)
        df['T'] = df['days_to_expiry'] / 365.0
        
        # Handle same-day expiration (minimum 1 hour = 1/252/6.5 years)
        df.loc[df['T'] <= 0, 'T'] = 1.0 / (252.0 * 6.5)
        
        return df
    
    def compute_mid_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute mid price from Bid/Ask.
        Fallback to last trade price if Bid/Ask invalid.
        """
        df = df.copy()
        
        # Check if Bid/Ask are valid
        valid_quotes = (df['Bid'] > 0) & (df['Ask'] > 0) & (df['Ask'] >= df['Bid'])
        
        # Compute mid
        df['mid_price'] = np.nan
        df.loc[valid_quotes, 'mid_price'] = (df.loc[valid_quotes, 'Bid'] + 
                                               df.loc[valid_quotes, 'Ask']) / 2.0
        
        # Fallback to last price
        df.loc[~valid_quotes, 'mid_price'] = df.loc[~valid_quotes, 'price']
        
        # Mark spread width
        df['spread_pct'] = np.where(
            valid_quotes,
            (df['Ask'] - df['Bid']) / df['mid_price'],
            np.nan
        )
        
        return df
    
    def compute_moneyness(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute multiple moneyness metrics.
        """
        df = df.copy()
        
        # Log-moneyness: ln(K/S)
        df['log_moneyness'] = np.log(df['price_strike'] / df['underlying_price'])
        
        # Simple moneyness: K/S
        df['moneyness'] = df['price_strike'] / df['underlying_price']
        
        # Standardized moneyness: (K-S)/S
        df['pct_moneyness'] = (df['price_strike'] - df['underlying_price']) / df['underlying_price']
        
        # ITM/ATM/OTM classification
        df['is_itm'] = (
            ((df['call_put'] == 'C') & (df['underlying_price'] > df['price_strike'])) |
            ((df['call_put'] == 'P') & (df['underlying_price'] < df['price_strike']))
        )
        
        df['is_atm'] = np.abs(df['pct_moneyness']) < 0.02  # Within 2%
        
        return df
    
    def estimate_forward_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate forward price using put-call parity.
        F = S + e^(rT) * (C - P) where C, P are mid prices at same strike.
        """
        df = df.copy()
        
        # Separate calls and puts
        calls = df[df['call_put'] == 'C'].copy()
        puts = df[df['call_put'] == 'P'].copy()
        
        # Merge on same strike and expiry
        merged = calls.merge(
            puts,
            on=['c_date', 'stocks_id', 'expiration_date', 'price_strike'],
            suffixes=('_call', '_put'),
            how='inner'
        )
        
        if len(merged) == 0:
            # No matching pairs, use spot as forward
            df['forward_price'] = df['underlying_price']
            return df
        
        # Compute forward from put-call parity
        discount_factor = np.exp(self.risk_free_rate * merged['T_call'])
        merged['forward_pcp'] = (
            merged['underlying_price_call'] + 
            discount_factor * (merged['mid_price_call'] - merged['mid_price_put'])
        )
        
        # Use ATM forwards (most reliable)
        if 'is_atm_call' in merged.columns:
            atm_merged = merged[merged['is_atm_call']].copy()
        else:
            atm_merged = merged.copy()
        
        if len(atm_merged) > 0:
            # Group by expiry and take median
            forward_by_expiry = atm_merged.groupby('expiration_date')['forward_pcp'].median()
        else:
            # Use all available data
            forward_by_expiry = merged.groupby('expiration_date')['forward_pcp'].median()
        
        # Map forwards back to original dataframe
        df['forward_price'] = df['expiration_date'].map(forward_by_expiry)
        
        # Fill missing with spot
        df['forward_price'].fillna(df['underlying_price'], inplace=True)
        
        return df

    def clean_implied_vol(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate implied volatility data.
        """
        df = df.copy()
        
        # Remove invalid IVs
        df = df[df['iv'].notna()]
        df = df[df['iv'] > 0]
        df = df[df['iv'] < 5.0]  # Remove absurd IVs (>500%)
        
        # Flag suspicious IVs
        df['iv_suspicious'] = (
            (df['iv'] > 2.0) |  # Very high IV
            (df['iv'] < 0.01)   # Very low IV
        )
        
        return df
    
    def filter_stale_quotes(
        self,
        df: pd.DataFrame,
        min_volume: int = 1,
        min_oi: int = 10,
        max_spread_pct: float = 0.5
    ) -> pd.DataFrame:
        """
        Filter out stale or illiquid options.
        """
        df = df.copy()
        
        # Volume/OI filters
        df = df[df['volume'] >= min_volume]
        df = df[df['openinterest'] >= min_oi]
        
        # Spread filter
        if 'spread_pct' in df.columns:
            df = df[(df['spread_pct'].isna()) | (df['spread_pct'] <= max_spread_pct)]
        
        # Remove zero-priced options
        df = df[df['mid_price'] > 0.01]
        
        # Remove zero-strike options
        df = df[df['price_strike'] > 0]
        
        return df
    
    def add_delta_buckets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize options by delta buckets for skew analysis.
        """
        df = df.copy()
        
        # Absolute delta
        df['abs_delta'] = np.abs(df['delta'])
        
        # Delta buckets
        conditions = [
            (df['abs_delta'] >= 0.45) & (df['abs_delta'] <= 0.55),  # ATM
            (df['abs_delta'] >= 0.20) & (df['abs_delta'] < 0.35),   # 25-delta
            (df['abs_delta'] >= 0.08) & (df['abs_delta'] < 0.15),   # 10-delta
            (df['abs_delta'] >= 0.35) & (df['abs_delta'] < 0.45),   # In between
        ]
        choices = ['ATM', '25D', '10D', 'OTHER']
        df['delta_bucket'] = np.select(conditions, choices, default='WING')
        
        return df
    
    def preprocess_chain(
            self,
            df: pd.DataFrame,
            min_volume: int = 1,
            min_oi: int = 10,
            max_spread_pct: float = 0.5
        ) -> pd.DataFrame:
            """
            Full preprocessing pipeline.
            """
            # Parse dates
            df = self.parse_dates(df)
            
            # Compute time to expiry
            df = self.compute_time_to_expiry(df)
            
            # Compute mid price
            df = self.compute_mid_price(df)
            
            # Compute moneyness
            df = self.compute_moneyness(df)
            
            # Estimate forward
            df = self.estimate_forward_price(df)
            
            # Clean IV
            df = self.clean_implied_vol(df)
            
            # Filter stale quotes
            df = self.filter_stale_quotes(df, min_volume, min_oi, max_spread_pct)
            
            # Add delta buckets
            df = self.add_delta_buckets(df)
            
            return df
    
# Example usage
if __name__ == "__main__":
    from data_loader import OptionsDataLoader
    
    # Load data
    loader = OptionsDataLoader("hood_options.db")
    df_raw = loader.load_chain_snapshot(
        stocks_id=1,
        c_date='2024-01-15T16:00:00'
    )
    
    # Preprocess
    preprocessor = OptionsPreprocessor(risk_free_rate=0.05)
    df_clean = preprocessor.preprocess_chain(df_raw)
    
    print(f"Raw records: {len(df_raw)}")
    print(f"Clean records: {len(df_clean)}")
    print(f"\nSample data:\n{df_clean.head()}")
