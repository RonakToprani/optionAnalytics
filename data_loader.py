"""
Data loader for options chain database with caching and efficient queries.
"""
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
import joblib
from functools import lru_cache

class OptionsDataLoader:
    """
    Efficient data loader for HOOD_OC SQLite database with caching.
    """
    
    def __init__(self, db_path: str, cache_dir: str = "./cache"):
        self.db_path = db_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory = joblib.Memory(self.cache_dir, verbose=0)
        
    def get_connection(self):
        """Create optimized SQLite connection."""
        conn = sqlite3.connect(self.db_path)
        # Performance optimizations
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        conn.execute("PRAGMA temp_store=MEMORY")
        return conn
    
    def get_available_underlyings(self) -> pd.DataFrame:
        """Get list of all underlyings with data statistics."""
        query = """
        SELECT 
            stocks_id,
            COUNT(DISTINCT c_date) as n_dates,
            COUNT(DISTINCT expiration_date) as n_expiries,
            MIN(c_date) as first_date,
            MAX(c_date) as last_date,
            COUNT(*) as n_records
        FROM HOOD_OC
        GROUP BY stocks_id
        ORDER BY n_records DESC
        """
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)
    
    def get_date_range(self, stocks_id: Optional[int] = None) -> Tuple[str, str]:
        """Get available date range for an underlying."""
        query = "SELECT MIN(c_date) as min_date, MAX(c_date) as max_date FROM HOOD_OC"
        if stocks_id:
            query += f" WHERE stocks_id = {stocks_id}"
        
        with self.get_connection() as conn:
            result = pd.read_sql_query(query, conn)
            return result['min_date'].iloc[0], result['max_date'].iloc[0]
    
    def get_expiries_for_date(self, stocks_id: int, c_date: str) -> List[str]:
        """Get available expiration dates for a snapshot."""
        query = """
        SELECT DISTINCT expiration_date
        FROM HOOD_OC
        WHERE stocks_id = ? AND c_date = ?
        ORDER BY expiration_date
        """
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(stocks_id, c_date))
            return df['expiration_date'].tolist()
    
    @lru_cache(maxsize=32)
    def load_chain_snapshot(
        self,
        stocks_id: int,
        c_date: str,
        expiration_date: Optional[str] = None,
        min_volume: int = 0,
        min_oi: int = 0
    ) -> pd.DataFrame:
        """
        Load option chain snapshot with filters.
        Cached for performance.
        """
        query = """
        SELECT 
            c_date,
            option_symbol,
            dte,
            stocks_id,
            expiration_date,
            call_put,
            price_strike,
            price_open,
            price_high,
            price_low,
            price,
            volume,
            openinterest,
            iv,
            delta,
            gamma,
            theta,
            vega,
            rho,
            preiv,
            Ask,
            Bid,
            underlying_price,
            calc_OTM,
            option_id
        FROM HOOD_OC
        WHERE stocks_id = ?
          AND c_date = ?
          AND volume >= ?
          AND openinterest >= ?
        """
        params = [stocks_id, c_date, min_volume, min_oi]
        
        if expiration_date:
            query += " AND expiration_date = ?"
            params.append(expiration_date)
        
        query += " ORDER BY expiration_date, price_strike, call_put"
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        return df
    
    def load_time_series(
        self,
        stocks_id: int,
        start_date: str,
        end_date: str,
        expiration_date: Optional[str] = None,
        strike_range: Optional[Tuple[float, float]] = None
    ) -> pd.DataFrame:
        """
        Load time series of option chains for historical analysis.
        """
        query = """
        SELECT *
        FROM HOOD_OC
        WHERE stocks_id = ?
          AND c_date >= ?
          AND c_date <= ?
        """
        params = [stocks_id, start_date, end_date]
        
        if expiration_date:
            query += " AND expiration_date = ?"
            params.append(expiration_date)
        
        if strike_range:
            query += " AND price_strike BETWEEN ? AND ?"
            params.extend(strike_range)
        
        query += " ORDER BY c_date, expiration_date, price_strike"
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        return df
    
    def export_to_parquet(self, stocks_id: int, output_path: str):
        """
        Export underlying data to Parquet for faster repeated access.
        """
        query = "SELECT * FROM HOOD_OC WHERE stocks_id = ?"
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(stocks_id,))
        
        df.to_parquet(output_path, engine='pyarrow', compression='snappy')
        print(f"Exported {len(df):,} rows to {output_path}")
        
        return df


# Example usage queries
SAMPLE_QUERIES = {
    "snapshot": """
        -- Get complete chain for a single date/expiry
        SELECT *
        FROM HOOD_OC
        WHERE stocks_id = 1
          AND c_date = '2024-01-15T16:00:00'
          AND expiration_date = '2024-02-16'
        ORDER BY price_strike, call_put;
    """,
    
    "atm_options": """
        -- Get near-ATM options across all expiries
        SELECT *
        FROM HOOD_OC
        WHERE stocks_id = 1
          AND c_date = '2024-01-15T16:00:00'
          AND ABS(price_strike - underlying_price) / underlying_price < 0.05
        ORDER BY expiration_date, ABS(price_strike - underlying_price);
    """,
    
    "liquid_options": """
        -- Get liquid options with volume/OI thresholds
        SELECT *
        FROM HOOD_OC
        WHERE stocks_id = 1
          AND c_date >= '2024-01-01'
          AND volume > 100
          AND openinterest > 500
        ORDER BY c_date, expiration_date;
    """,
    
    "iv_surface_data": """
        -- Get data for IV surface construction
        SELECT 
            expiration_date,
            price_strike,
            call_put,
            underlying_price,
            (Ask + Bid) / 2.0 as mid_price,
            iv,
            delta,
            volume,
            openinterest
        FROM HOOD_OC
        WHERE stocks_id = ?
          AND c_date = ?
          AND Ask > 0 AND Bid > 0
          AND iv IS NOT NULL AND iv > 0
        ORDER BY expiration_date, price_strike;
    """
}
