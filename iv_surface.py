"""
Implied volatility surface construction using SVI and SABR models.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator, griddata
from scipy.optimize import minimize, differential_evolution
from typing import Tuple, Optional, Dict
import warnings

class SVICalibrator:
    """
    Stochastic Volatility Inspired (SVI) parameterization for IV smile.
    
    SVI formula: σ²(k) = a + b * (ρ * (k - m) + sqrt((k - m)² + σ²))
    where k = log(K/F) is log-moneyness
    """
    
    @staticmethod
    def svi_variance(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
        """
        SVI total variance formula.
        
        Args:
            k: Log-moneyness ln(K/F)
            a, b, rho, m, sigma: SVI parameters
        """
        sqrt_term = np.sqrt((k - m)**2 + sigma**2)
        return a + b * (rho * (k - m) + sqrt_term)
    
    @staticmethod
    def svi_implied_vol(k: np.ndarray, T: float, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
        """Convert SVI variance to implied volatility."""
        variance = SVICalibrator.svi_variance(k, a, b, rho, m, sigma)
        return np.sqrt(np.maximum(variance / T, 1e-6))  # Ensure non-negative
    
    @staticmethod
    def svi_constraints(params: np.ndarray) -> bool:
        """Check SVI no-arbitrage conditions."""
        a, b, rho, m, sigma = params
        
        # b >= 0
        if b < 0:
            return False
        
        # |rho| < 1
        if abs(rho) >= 1:
            return False
        
        # sigma > 0
        if sigma <= 0:
            return False
        
        # a + b * sigma * sqrt(1 - rho²) >= 0 (ensures non-negative variance)
        if a + b * sigma * np.sqrt(1 - rho**2) < 0:
            return False
        
        return True
    
    def calibrate(
        self,
        strikes: np.ndarray,
        ivs: np.ndarray,
        forward: float,
        T: float,
        weights: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calibrate SVI parameters to market implied vols.
        
        Args:
            strikes: Strike prices
            ivs: Market implied volatilities
            forward: Forward price
            T: Time to expiry in years
            weights: Optional weights for each observation (e.g., by vega)
        
        Returns:
            Dictionary of calibrated parameters
        """
        # Log-moneyness
        k = np.log(strikes / forward)
        
        # Target total variance
        target_var = ivs**2 * T
        
        # Weights (default: equal)
        if weights is None:
            weights = np.ones_like(k)
        
        # Objective: weighted RMSE
        def objective(params):
            if not self.svi_constraints(params):
                return 1e10
            
            a, b, rho, m, sigma = params
            model_var = self.svi_variance(k, a, b, rho, m, sigma)
            residuals = (model_var - target_var) * weights
            return np.sum(residuals**2)
        
        # Initial guess
        atm_var = np.interp(0, k, target_var)  # ATM total variance
        initial = np.array([
            atm_var * 0.8,  # a
            0.1,            # b
            -0.2,           # rho (typically negative for equity skew)
            0.0,            # m
            0.1             # sigma
        ])
        
        # Bounds
        bounds = [
            (-1, 2),      # a
            (0, 1),       # b
            (-0.999, 0.999),  # rho
            (-1, 1),      # m
            (0.01, 2)     # sigma
        ]
        
        # Optimize
        result = minimize(
            objective,
            initial,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        if not result.success or not self.svi_constraints(result.x):
            # Fallback: use global optimizer
            result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=100
            )
        
        a, b, rho, m, sigma = result.x
        
        # Compute fit quality
        model_var = self.svi_variance(k, a, b, rho, m, sigma)
        model_iv = np.sqrt(model_var / T)
        rmse = np.sqrt(np.mean((model_iv - ivs)**2))
        
        return {
            'a': a,
            'b': b,
            'rho': rho,
            'm': m,
            'sigma': sigma,
            'rmse': rmse,
            'success': result.success
        }


class SABRCalibrator:
    """
    SABR (Stochastic Alpha Beta Rho) model for IV smile.
    
    Uses Hagan's approximation formula.
    """
    
    @staticmethod
    def sabr_volatility(
        F: float,
        K: float,
        T: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float
    ) -> float:
        """
        SABR implied volatility using Hagan et al. (2002) approximation.
        
        Args:
            F: Forward price
            K: Strike
            T: Time to expiry
            alpha: Initial volatility
            beta: CEV exponent (0=normal, 1=lognormal)
            rho: Correlation
            nu: Vol-of-vol
        """
        # ATM case
        if abs(F - K) < 1e-10:
            FK_mid = F
            z = 0
            x_z = 1
        else:
            FK_mid = (F * K) ** ((1 - beta) / 2)
            z = (nu / alpha) * FK_mid * np.log(F / K)
            x_z = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
        
        # First term
        term1 = alpha / (FK_mid * (1 + ((1-beta)**2 / 24) * (np.log(F/K))**2))
        
        # Second term (z correction)
        if abs(z) < 1e-7:
            term2 = 1
        else:
            term2 = z / x_z
        
        # Third term (time correction)
        A = ((1-beta)**2 / 24) * (alpha**2 / FK_mid**2)
        B = 0.25 * rho * beta * nu * alpha / FK_mid
        C = (2 - 3*rho**2) * nu**2 / 24
        term3 = 1 + (A + B + C) * T
        
        return term1 * term2 * term3
    
    def calibrate(
        self,
        strikes: np.ndarray,
        ivs: np.ndarray,
        forward: float,
        T: float,
        beta: float = 0.5
    ) -> Dict[str, float]:
        """
        Calibrate SABR parameters (fixing beta).
        """
        def objective(params):
            alpha, rho, nu = params
            
            # Bounds checks
            if alpha <= 0 or abs(rho) >= 1 or nu <= 0:
                return 1e10
            
            model_ivs = np.array([
                self.sabr_volatility(forward, K, T, alpha, beta, rho, nu)
                for K in strikes
            ])
            
            return np.sum((model_ivs - ivs)**2)
        
        # Initial guess
        atm_iv = np.interp(forward, strikes, ivs)
        initial = [atm_iv * 0.8, -0.3, 0.3]
        
        # Bounds
        bounds = [(0.01, 2), (-0.999, 0.999), (0.01, 2)]
        
        result = minimize(objective, initial, bounds=bounds, method='L-BFGS-B')
        
        alpha, rho, nu = result.x
        
        # Compute RMSE
        model_ivs = np.array([
            self.sabr_volatility(forward, K, T, alpha, beta, rho, nu)
            for K in strikes
        ])
        rmse = np.sqrt(np.mean((model_ivs - ivs)**2))
        
        return {
            'alpha': alpha,
            'beta': beta,
            'rho': rho,
            'nu': nu,
            'rmse': rmse,
            'success': result.success
        }


class IVSurface:
    """
    Complete implied volatility surface with multiple interpolation methods.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: Preprocessed options dataframe with T, moneyness, iv, forward_price
        """
        self.df = df.copy()
        self.svi_params = {}
        self.sabr_params = {}
        
    def calibrate_per_expiry(self, model: str = 'svi', min_points: int = 5):
        """
        Calibrate model parameters for each expiry separately.
        """
        grouped = self.df.groupby('expiration_date')
        
        for expiry, group in grouped:
            if len(group) < min_points:
                continue
            
            # Extract data
            strikes = group['price_strike'].values
            ivs = group['iv'].values
            forward = group['forward_price'].iloc[0]
            T = group['T'].iloc[0]
            
            if model == 'svi':
                calibrator = SVICalibrator()
                params = calibrator.calibrate(strikes, ivs, forward, T)
                self.svi_params[expiry] = params
            
            elif model == 'sabr':
                calibrator = SABRCalibrator()
                params = calibrator.calibrate(strikes, ivs, forward, T)
                self.sabr_params[expiry] = params
    
    def get_iv_grid(
        self,
        strike_range: Tuple[float, float],
        expiry_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
        n_strikes: int = 50,
        n_expiries: int = 20,
        method: str = 'rbf'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate IV surface grid using interpolation.
        
        Args:
            strike_range: (min_strike, max_strike)
            expiry_range: Optional (min_expiry, max_expiry)
            n_strikes: Grid resolution in strike dimension
            n_expiries: Grid resolution in expiry dimension
            method: 'rbf', 'linear', or 'cubic'
        
        Returns:
            (strike_grid, expiry_grid, iv_grid)
        """
        df = self.df.copy()
        
        # Filter expiries
        if expiry_range:
            df = df[(df['expiration_date'] >= expiry_range[0]) & 
                   (df['expiration_date'] <= expiry_range[1])]
        
        # Filter strikes
        df = df[(df['price_strike'] >= strike_range[0]) & 
               (df['price_strike'] <= strike_range[1])]
        
        # Extract points
        strikes = df['price_strike'].values
        expiries_dt = df['expiration_date'].values
        ivs = df['iv'].values
        
        # Convert expiries to numeric (days from first)
        first_expiry = expiries_dt.min()
        expiries_numeric = (expiries_dt - first_expiry) / np.timedelta64(1, 'D')
        
        # Create grid
        strike_grid = np.linspace(strike_range[0], strike_range[1], n_strikes)
        expiry_grid = np.linspace(expiries_numeric.min(), expiries_numeric.max(), n_expiries)
        
        X_grid, Y_grid = np.meshgrid(strike_grid, expiry_grid)
        
        # Interpolate
        if method == 'rbf':
            points = np.column_stack([strikes, expiries_numeric])
            rbf = RBFInterpolator(points, ivs, kernel='thin_plate_spline', smoothing=0.01)
            iv_grid = rbf(np.column_stack([X_grid.ravel(), Y_grid.ravel()]))
            iv_grid = iv_grid.reshape(X_grid.shape)
        else:
            iv_grid = griddata(
                points=np.column_stack([strikes, expiries_numeric]),
                values=ivs,
                xi=(X_grid, Y_grid),
                method=method
            )
        
        # Convert expiry grid back to dates
        expiry_grid_dates = first_expiry + expiry_grid.astype('timedelta64[D]')
        
        return strike_grid, expiry_grid_dates, iv_grid


# Example usage
if __name__ == "__main__":
    # Sample data creation (normally from database)
    np.random.seed(42)
    n_samples = 100
    
    sample_df = pd.DataFrame({
        'price_strike': np.linspace(90, 110, n_samples),
        'iv': 0.2 + 0.05 * np.random.randn(n_samples),
        'forward_price': 100,
        'T': 0.25,
        'expiration_date': pd.Timestamp('2024-03-15')
    })
    
    # Calibrate SVI
    surface = IVSurface(sample_df)
    surface.calibrate_per_expiry(model='svi')
    
    print("SVI Parameters:", surface.svi_params)
