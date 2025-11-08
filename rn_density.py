"""
Risk-neutral probability density extraction using Breeden-Litzenberger
and parametric approaches.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, CubicSpline
from scipy.integrate import simpson, quad
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from typing import Tuple, Optional, Callable
import warnings

class RiskNeutralDensity:
    """
    Extract risk-neutral probability density from option prices.
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
    
    def black_scholes_call(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """Black-Scholes call price."""
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    
    def breeden_litzenberger(
        self,
        strikes: np.ndarray,
        call_prices: np.ndarray,
        T: float,
        r: Optional[float] = None,
        smoothing: float = 0.001,
        extrapolate: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Callable]:
        """
        Extract risk-neutral density using Breeden-Litzenberger formula:
        
        q(K) = e^{rT} * d²C/dK²
        
        Args:
            strikes: Array of strike prices (must be sorted)
            call_prices: Corresponding call prices
            T: Time to expiry in years
            r: Risk-free rate (default: self.risk_free_rate)
            smoothing: Smoothing parameter for spline (higher = smoother)
            extrapolate: If True, extrapolate beyond observed strikes
        
        Returns:
            (strike_grid, density, pdf_function)
        """
        if r is None:
            r = self.risk_free_rate
        
        # Sort by strike
        idx = np.argsort(strikes)
        strikes = strikes[idx]
        call_prices = call_prices[idx]
        
        # Remove duplicates
        strikes, unique_idx = np.unique(strikes, return_index=True)
        call_prices = call_prices[unique_idx]
        
        # Fit smooth spline to call prices
        # Use cubic smoothing spline with cross-validation
        spline = UnivariateSpline(
            strikes,
            call_prices,
            k=3,
            s=smoothing * len(strikes),
            ext='extrapolate' if extrapolate else 'const'
        )
        
        # Create dense grid for differentiation
        K_min, K_max = strikes.min(), strikes.max()
        
        if extrapolate:
            # Extend grid by 20% on each side
            K_range = K_max - K_min
            K_min_ext = max(K_min - 0.2*K_range, K_min * 0.5)
            K_max_ext = K_max + 0.2*K_range
            strike_grid = np.linspace(K_min_ext, K_max_ext, 200)
        else:
            strike_grid = np.linspace(K_min, K_max, 200)
        
        # Compute second derivative numerically
        # Use second-order finite difference
        h = strike_grid[1] - strike_grid[0]
        call_grid = spline(strike_grid)
        
        # Second derivative: f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
        d2C_dK2 = np.gradient(np.gradient(call_grid, h), h)
        
        # Ensure non-negative (enforce arbitrage bounds)
        d2C_dK2 = np.maximum(d2C_dK2, 0)
        
        # Apply discount factor
        discount = np.exp(r * T)
        density = discount * d2C_dK2
        
        # Normalize to ensure it's a proper PDF
        total_prob = simpson(density, x=strike_grid)
        
        if total_prob > 0:
            density = density / total_prob
        else:
            warnings.warn("Unable to normalize density (total probability near zero)")
        
        # Create interpolation function for PDF
        pdf_spline = UnivariateSpline(strike_grid, density, k=3, s=0, ext='const')
        
        return strike_grid, density, pdf_spline
    
    def parametric_density_from_svi(
        self,
        forward: float,
        T: float,
        svi_params: dict,
        strike_grid: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute risk-neutral density from calibrated SVI parameters.
        
        Uses relationship between IV surface and density via BS formula.
        """
        if strike_grid is None:
            # Create grid around forward
            K_min = forward * 0.5
            K_max = forward * 1.5
            strike_grid = np.linspace(K_min, K_max, 200)
        
        # Extract SVI params
        a = svi_params['a']
        b = svi_params['b']
        rho = svi_params['rho']
        m = svi_params['m']
        sigma = svi_params['sigma']
        
        # Compute IV for each strike
        k = np.log(strike_grid / forward)
        sqrt_term = np.sqrt((k - m)**2 + sigma**2)
        variance = a + b * (rho * (k - m) + sqrt_term)
        iv = np.sqrt(np.maximum(variance / T, 1e-6))
        
        # Compute call prices using Black-Scholes
        call_prices = np.array([
            self.black_scholes_call(forward, K, T, self.risk_free_rate, vol)
            for K, vol in zip(strike_grid, iv)
        ])
        
        # Extract density using Breeden-Litzenberger
        _, density, _ = self.breeden_litzenberger(
            strike_grid,
            call_prices,
            T,
            smoothing=0.0001  # Lower smoothing since SVI is already smooth
        )
        
        return strike_grid, density
    
    def compute_quantiles(
        self,
        strike_grid: np.ndarray,
        density: np.ndarray,
        quantiles: np.ndarray = np.array([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    ) -> dict:
        """
        Compute quantiles and tail probabilities from RN density.
        """
        # Compute CDF via cumulative integration
        cdf = np.cumsum(density) * (strike_grid[1] - strike_grid[0])
        cdf = cdf / cdf[-1]  # Normalize to [0, 1]
        
        # Interpolate to find quantile values
        quantile_values = np.interp(quantiles, cdf, strike_grid)
        
        results = {
            'quantiles': quantiles,
            'values': quantile_values,
            'cdf': cdf,
            'strike_grid': strike_grid
        }
        
        return results
    
    def tail_probabilities(
        self,
        strike_grid: np.ndarray,
        density: np.ndarray,
        reference_price: float
    ) -> dict:
        """
        Compute probabilities of extreme moves.
        """
        # P(S_T < reference_price)
        mask_below = strike_grid < reference_price
        prob_below = simpson(density[mask_below], x=strike_grid[mask_below])
        
        # P(S_T > reference_price)
        prob_above = 1 - prob_below
        
        # Expected value
        expected_price = simpson(strike_grid * density, x=strike_grid)
        
        # Variance
        variance = simpson((strike_grid - expected_price)**2 * density, x=strike_grid)
        
        return {
            'prob_below': prob_below,
            'prob_above': prob_above,
            'expected_price': expected_price,
            'variance': variance,
            'std_dev': np.sqrt(variance)
        }
    
    def compare_with_lognormal(
        self,
        strike_grid: np.ndarray,
        rn_density: np.ndarray,
        forward: float,
        T: float,
        atm_iv: float
    ) -> Tuple[np.ndarray, dict]:
        """
        Compare risk-neutral density with lognormal benchmark.
        """
        # Lognormal density parameters
        mu = np.log(forward) - 0.5 * atm_iv**2 * T
        sigma = atm_iv * np.sqrt(T)
        
        # Lognormal PDF
        lognormal_density = (1 / (strike_grid * sigma * np.sqrt(2*np.pi))) * \
                           np.exp(-0.5 * ((np.log(strike_grid) - mu) / sigma)**2)
        
        # Normalize
        total_prob = simpson(lognormal_density, x=strike_grid)
        lognormal_density = lognormal_density / total_prob
        
        # Compute differences
        kl_divergence = simpson(
            rn_density * np.log(rn_density / (lognormal_density + 1e-10) + 1e-10),
            x=strike_grid
        )
        
        # Skewness and kurtosis
        mean_rn = simpson(strike_grid * rn_density, x=strike_grid)
        std_rn = np.sqrt(simpson((strike_grid - mean_rn)**2 * rn_density, x=strike_grid))
        
        skew_rn = simpson(((strike_grid - mean_rn) / std_rn)**3 * rn_density, x=strike_grid)
        kurt_rn = simpson(((strike_grid - mean_rn) / std_rn)**4 * rn_density, x=strike_grid)
        
        comparison = {
            'kl_divergence': kl_divergence,
            'rn_skewness': skew_rn,
            'rn_kurtosis': kurt_rn,
            'lognormal_skewness': 0,  # By definition
            'lognormal_kurtosis': 3   # By definition
        }
        
        return lognormal_density, comparison


class HistoricalDensity:
    """
    Compute historical (physical) probability density from realized returns.
    """
    
    @staticmethod
    def compute_realized_density(
        returns: np.ndarray,
        forward: float,
        T: float,
        bandwidth: Optional[float] = None,
        n_points: int = 200
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute kernel density estimate of future price from historical returns.
        
        Args:
            returns: Historical log returns
            forward: Current forward price
            T: Forecast horizon in years
            bandwidth: KDE bandwidth (default: Silverman's rule)
            n_points: Number of grid points
        
        Returns:
            (price_grid, density)
        """
        from scipy.stats import gaussian_kde
        
        # Scale returns to forecast horizon
        scaled_returns = returns * np.sqrt(T)
        
        # Compute KDE
        if bandwidth is None:
            kde = gaussian_kde(scaled_returns, bw_method='silverman')
        else:
            kde = gaussian_kde(scaled_returns, bw_method=bandwidth)
        
        # Create price grid
        std_return = np.std(scaled_returns)
        ret_min = np.mean(scaled_returns) - 4*std_return
        ret_max = np.mean(scaled_returns) + 4*std_return
        
        price_grid = forward * np.exp(np.linspace(ret_min, ret_max, n_points))
        
        # Compute density (change of variables: returns -> prices)
        log_returns = np.log(price_grid / forward)
        density_returns = kde(log_returns)
        
        # Jacobian correction: p(S) = p(ln(S)) * (1/S)
        density = density_returns / price_grid
        
        # Normalize
        total_prob = simpson(density, x=price_grid)
        density = density / total_prob
        
        return price_grid, density


# Example usage
if __name__ == "__main__":
    # Generate sample option data
    forward = 100.0
    T = 0.25
    r = 0.05
    sigma = 0.25
    
    rnd = RiskNeutralDensity(risk_free_rate=r)
    
    # Create strikes and call prices
    strikes = np.linspace(80, 120, 30)
    call_prices = np.array([
        rnd.black_scholes_call(forward, K, T, r, sigma)
        for K in strikes
    ])
    
    # Extract density
    K_grid, density, pdf_func = rnd.breeden_litzenberger(strikes, call_prices, T)
    
    # Compute quantiles
    quantiles_result = rnd.compute_quantiles(K_grid, density)
    
    print("Risk-Neutral Quantiles:")
    for q, v in zip(quantiles_result['quantiles'], quantiles_result['values']):
        print(f"  {q*100:.0f}%: {v:.2f}")
    
    # Tail probabilities
    tails = rnd.tail_probabilities(K_grid, density, forward)
    print(f"\nExpected Price: {tails['expected_price']:.2f}")
    print(f"Std Dev: {tails['std_dev']:.2f}")
