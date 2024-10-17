import numpy as np
from scipy.stats import norm

def calculate_d1_d2(S, K, r, T, sigma):
    """Calculate d1 and d2 parameters for Black-Scholes model"""
    if sigma <= 0 or T <= 0:
        raise ValueError(f"Volatility and time to expiration must be positive. Current values: sigma={sigma}, T={T}")
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def blackScholes(S, K, r, T, sigma, option_type="C"):
    """Calculate Black-Scholes option price for a call/put"""
    try:
        d1, d2 = calculate_d1_d2(S, K, r, T, sigma)
        
        if option_type == "C":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == "P":
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError(f"Invalid option type. Use 'C' for call or 'P' for put. Received: {option_type}")
        
        return max(0, price) if price > 1e-10 else 0
    except Exception as e:
        raise ValueError(f"Error in Black-Scholes calculation: {str(e)}. Inputs: S={S}, K={K}, r={r}, T={T}, sigma={sigma}, option_type={option_type}")

def optionDelta(S, K, r, T, sigma, option_type="C"):
    """Calculate option delta"""
    try:
        d1, _ = calculate_d1_d2(S, K, r, T, sigma)
        if option_type == "C":
            return norm.cdf(d1)
        elif option_type == "P":
            return -norm.cdf(-d1)
        else:
            raise ValueError(f"Invalid option type. Use 'C' for call or 'P' for put. Received: {option_type}")
    except Exception as e:
        raise ValueError(f"Error in delta calculation: {str(e)}. Inputs: S={S}, K={K}, r={r}, T={T}, sigma={sigma}, option_type={option_type}")

def optionGamma(S, K, r, T, sigma):
    """Calculate option gamma"""
    try:
        d1, _ = calculate_d1_d2(S, K, r, T, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    except Exception as e:
        raise ValueError(f"Error in gamma calculation: {str(e)}. Inputs: S={S}, K={K}, r={r}, T={T}, sigma={sigma}")

def optionTheta(S, K, r, T, sigma, option_type="C"):
    """Calculate option theta"""
    try:
        d1, d2 = calculate_d1_d2(S, K, r, T, sigma)
        common_term = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if option_type == "C":
            theta = common_term - r * K * np.exp(-r*T) * norm.cdf(d2)
        elif option_type == "P":
            theta = common_term + r * K * np.exp(-r*T) * norm.cdf(-d2)
        else:
            raise ValueError(f"Invalid option type. Use 'C' for call or 'P' for put. Received: {option_type}")
        return theta / 365  # Convert to daily theta
    except Exception as e:
        raise ValueError(f"Error in theta calculation: {str(e)}. Inputs: S={S}, K={K}, r={r}, T={T}, sigma={sigma}, option_type={option_type}")

def optionVega(S, K, r, T, sigma):
    """Calculate option vega"""
    try:
        d1, _ = calculate_d1_d2(S, K, r, T, sigma)
        return S * np.sqrt(T) * norm.pdf(d1) * 0.01  # Multiply by 0.01 to get 1% change
    except Exception as e:
        raise ValueError(f"Error in vega calculation: {str(e)}. Inputs: S={S}, K={K}, r={r}, T={T}, sigma={sigma}")
   
def optionRho(S, K, r, T, sigma, option_type="C"):
    """Calculate option rho"""
    try:
        _, d2 = calculate_d1_d2(S, K, r, T, sigma)
        if option_type == "C":
            rho = 0.01 * K * T * np.exp(-r*T) * norm.cdf(d2)
        elif option_type == "P":
            rho = -0.01 * K * T * np.exp(-r*T) * norm.cdf(-d2)
        else:
            raise ValueError(f"Invalid option type. Use 'C' for call or 'P' for put. Received: {option_type}")
        return rho
    except Exception as e:
        raise ValueError(f"Error in rho calculation: {str(e)}. Inputs: S={S}, K={K}, r={r}, T={T}, sigma={sigma}, option_type={option_type}")