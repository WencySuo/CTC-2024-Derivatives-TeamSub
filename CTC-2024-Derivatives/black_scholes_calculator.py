import numpy as np
from scipy.stats import norm

def blackScholes(S, K, r, T, sigma, type="c"):
    """Calculate Black-Scholes option price for a call/put"""
    if sigma <= 0 or T <= 0:
        raise ValueError("Volatility and time to expiration must be positive")
    
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    try:
        if type == "c":
            if K > S * np.exp(r * T) * 10:  # Deep OTM call
                return max(0, S - K * np.exp(-r * T))
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif type == "p":
            if S > K * np.exp(r * T) * 10:  # Deep OTM put
                return max(0, K * np.exp(-r * T) - S)
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("Invalid option type. Use 'c' for call or 'p' for put.")
        return max(0, price)  # Ensure non-negative price
    except Exception as e:
        raise ValueError(f"Error in Black-Scholes calculation: {str(e)}")

def optionDelta(S, K, r, T, sigma, type="c"):
    """Calculate option delta"""
    if sigma <= 0 or T <= 0:
        raise ValueError("Volatility and time to expiration must be positive")
    
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    
    try:
        if type == "c":
            delta = norm.cdf(d1)
        elif type == "p":
            delta = -norm.cdf(-d1)
        else:
            raise ValueError("Invalid option type. Use 'c' for call or 'p' for put.")
        return delta
    except Exception as e:
        raise ValueError(f"Error in delta calculation: {str(e)}")

def optionGamma(S, K, r, T, sigma):
    """Calculate option gamma"""
    if sigma <= 0 or T <= 0:
        raise ValueError("Volatility and time to expiration must be positive")
    
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    
    try:
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma
    except Exception as e:
        raise ValueError(f"Error in gamma calculation: {str(e)}")

def optionTheta(S, K, r, T, sigma, type="c"):
    """Calculate option theta"""
    if sigma <= 0 or T <= 0:
        raise ValueError("Volatility and time to expiration must be positive")
    
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    try:
        if type == "c":
            theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r*T) * norm.cdf(d2)
        elif type == "p":
            theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r*T) * norm.cdf(-d2)
        else:
            raise ValueError("Invalid option type. Use 'c' for call or 'p' for put.")
        return theta / 365  # Convert to daily theta
    except Exception as e:
        raise ValueError(f"Error in theta calculation: {str(e)}")

def optionVega(S, K, r, T, sigma):
    """Calculate option vega"""
    if sigma <= 0 or T <= 0:
        raise ValueError("Volatility and time to expiration must be positive")
    
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    
    try:
        vega = S * np.sqrt(T) * norm.pdf(d1) * 0.01  # Multiply by 0.01 to get 1% change
        return vega
    except Exception as e:
        raise ValueError(f"Error in vega calculation: {str(e)}")

def optionRho(S, K, r, T, sigma, type="c"):
    """Calculate option rho"""
    if sigma <= 0 or T <= 0:
        raise ValueError("Volatility and time to expiration must be positive")
    
    d2 = (np.log(S/K) + (r - sigma**2/2) * T) / (sigma * np.sqrt(T))
    
    try:
        if type == "c":
            rho = 0.01 * K * T * np.exp(-r*T) * norm.cdf(d2)
        elif type == "p":
            rho = -0.01 * K * T * np.exp(-r*T) * norm.cdf(-d2)
        else:
            raise ValueError("Invalid option type. Use 'c' for call or 'p' for put.")
        return rho
    except Exception as e:
        raise ValueError(f"Error in rho calculation: {str(e)}")
    

#### REMOVED OLD CALC FUNCTIONS IN Starter.py #####
# @staticmethod
#   def d1(asset_price, strike_price, time_expire, risk_free_rate, sigma):
#     # try:
#     #   return (np.log(asset_price / float(strike_price) ) + (risk_free_rate + 0.5 * sigma ** 2) * time_expire) / (sigma * np.sqrt(time_expire))
#     # except Warning as e:
#     #   print(str([asset_price, strike_price, time_expire, risk_free_rate, sigma]))
#     #   return 1
#     # address runtime issues 
#     if sigma <= 0 or time_expire <= 0: 
#       return np.nan
#     try: 
#       numerator = np.log1p((asset_price / strike_price) - 1) + (risk_free_rate + 0.5 * sigma ** 2) * time_expire
#       denominator = sigma * np.sqrt(time_expire)
#       return numerator / denominator
#     except OverflowError: 
#       print(str([asset_price, strike_price, time_expire, risk_free_rate, sigma]))
#       return np.inf if numerator > 0 else -np.inf

    
#   @staticmethod
#   def d2(asset_price, strike_price, time_expire, risk_free_rate, sigma):
#     # return (np.log(asset_price / float(strike_price) ) + (risk_free_rate - 0.5 * sigma ** 2) * time_expire) / (sigma * np.sqrt(time_expire))
#     # calc based off d1
#     d1 = Strategy.d1(asset_price, strike_price, time_expire, risk_free_rate, sigma)
#     if np.isnan(d1):
#       return np.nan
#     return d1 - sigma * np.sqrt(time_expire)

#   @staticmethod
#   #analytical solution of delta from black sholes equation
#   def delta(asset_price, strike_price, time_expire, risk_free_rate, sigma, action):
#     d_1 = Strategy.d1(asset_price, strike_price, time_expire, risk_free_rate, sigma)

#     if action == 'P':
#       return Strategy.cdf(d_1) - 1.0
#     else:
#       return Strategy.cdf(d_1)
    
#   @staticmethod
#   #analytical solution of vega from black sholes equation
#   def vega(asset_price, strike_price, time_expire, risk_free_rate, sigma, action):
#     d_1 = Strategy.d1(asset_price, strike_price, time_expire, risk_free_rate, sigma)
#     return asset_price * scipy.stats.norm._pdf(d_1) * np.sqrt(time_expire) * 0.01
  
#   @staticmethod
#   def gamma(asset_price, strike_price, time_expire, risk_free_rate, sigma, action):
#     d_1 = Strategy.d1(asset_price, strike_price, time_expire, risk_free_rate, sigma)
#     return scipy.stats.norm._pdf(d_1)/( asset_price * sigma * np.sqrt(time_expire))
  
#   @staticmethod
#   def rho( asset_price, strike_price, time_expire, risk_free_rate, sigma, action):
#     #rho is defined as the change in price for each 1 percent change in r, hence we multiply by 0.01.
#     d_2 = Strategy.d2(asset_price, strike_price, time_expire, risk_free_rate, sigma)
#     if action == 'C':
#       return time_expire * strike_price * np.exp(-risk_free_rate*time_expire) * Strategy.cdf(d_2) * .01
#     else:
#       return -time_expire * strike_price * np.exp(-risk_free_rate*time_expire) * Strategy.cdf(-d_2) * .01

#   @staticmethod
#   def theta(asset_price, strike_price, time_expire, risk_free_rate, sigma, action):

#     d_1 = Strategy.d1(asset_price, strike_price, time_expire, risk_free_rate, sigma)
#     d_2 = Strategy.d2(asset_price, strike_price, time_expire, risk_free_rate, sigma)

#     first_term = (-asset_price * scipy.stats.norm._pdf(d_1) * sigma) / (2 * np.sqrt(time_expire))

#     #theta is defined as the change in price for each day change in t, hence we divide by 365.
#     if action == 'C':
#       second_term = risk_free_rate * strike_price * np.exp(-risk_free_rate*time_expire) * scipy.special.ndtr(d_2)
#       return (first_term - second_term)/365.0
    
#     if action == 'P':
#       second_term = risk_free_rate * strike_price * np.exp(-risk_free_rate*time_expire) * scipy.special.ndtr(d_2)
#       return (first_term + second_term)/365.0

  
  
#   @staticmethod
#   def black_scholes(S, K, T, r, sigma, option):

#     d1 = Strategy.d1(S, K, T, r, sigma)
#     d2 = Strategy.d2(S, K, T, r, sigma)

#     if np.isnan(d1) or np.isnan(d2):
#       return np.nan

#     # if option == 'C':
#     #   return S * scipy.special.ndtr(d_1) - K * np.exp(-r * T) * scipy.special.ndtr(d_2)
#     # if option == 'P':
#     #   return K * np.exp(-r * T) * Strategy.cdf(-d_2) - S * Strategy.cdf(-d_1)

#     if option == 'C':
#       return S * Strategy.cdf(d1) - K * np.exp(-r * T) * Strategy.cdf(d2)
#     elif option == 'P':
#       return K * np.exp(-r * T) * Strategy.cdf(-d2) - S * Strategy.cdf(-d1)
