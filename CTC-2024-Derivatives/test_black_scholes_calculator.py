import unittest
import numpy as np
from black_scholes_calculator import (
    blackScholes, optionDelta, optionGamma, optionTheta, optionVega, optionRho
)

class TestBlackScholesCalculator(unittest.TestCase):
    def setUp(self):
        self.S = 100  # Stock price
        self.K = 100  # Strike price
        self.r = 0.05  # Risk-free rate
        self.T = 1  # Time to expiration (in years)
        self.sigma = 0.2  # Volatility

    def test_black_scholes(self):
        # Test call option
        call_price = blackScholes(self.S, self.K, self.r, self.T, self.sigma, "c")
        self.assertAlmostEqual(call_price, 10.4506, places=4)

        # Test put option
        put_price = blackScholes(self.S, self.K, self.r, self.T, self.sigma, "p")
        self.assertAlmostEqual(put_price, 5.5735, places=4)

        # Test put-call parity
        self.assertAlmostEqual(call_price - put_price, self.S - self.K * np.exp(-self.r * self.T), places=4)

        # Test invalid option type
        with self.assertRaises(ValueError):
            blackScholes(self.S, self.K, self.r, self.T, self.sigma, "x")

    def test_option_delta(self):
        # Test call option delta
        call_delta = optionDelta(self.S, self.K, self.r, self.T, self.sigma, "c")
        self.assertAlmostEqual(call_delta, 0.6368, places=4)

        # Test put option delta
        put_delta = optionDelta(self.S, self.K, self.r, self.T, self.sigma, "p")
        self.assertAlmostEqual(put_delta, -0.3632, places=4)

        # Test delta sum
        self.assertAlmostEqual(call_delta - put_delta, 1, places=4)

    def test_option_gamma(self):
        gamma = optionGamma(self.S, self.K, self.r, self.T, self.sigma)
        self.assertAlmostEqual(gamma, 0.0188, places=4)

    def test_option_theta(self):
        # Test call option theta
        call_theta = optionTheta(self.S, self.K, self.r, self.T, self.sigma, "c")
        self.assertAlmostEqual(call_theta, -0.0176, places=4)

        # Test put option theta
        put_theta = optionTheta(self.S, self.K, self.r, self.T, self.sigma, "p")
        self.assertAlmostEqual(put_theta, -0.0045, places=4)

    def test_option_vega(self):
        vega = optionVega(self.S, self.K, self.r, self.T, self.sigma)
        self.assertAlmostEqual(vega, 0.3752, places=4)

    def test_option_rho(self):
        # Test call option rho
        call_rho = optionRho(self.S, self.K, self.r, self.T, self.sigma, "c")
        self.assertAlmostEqual(call_rho, 0.5323, places=4)

        # Test put option rho
        put_rho = optionRho(self.S, self.K, self.r, self.T, self.sigma, "p")
        self.assertAlmostEqual(put_rho, -0.4189, places=4)

    def test_edge_cases(self):
        # Test zero volatility
        with self.assertRaises(ValueError):
            blackScholes(self.S, self.K, self.r, self.T, 0, "c")

        # Test zero time to expiration
        with self.assertRaises(ValueError):
            blackScholes(self.S, self.K, self.r, 0, self.sigma, "c")

        # Test very large volatility
        large_vol_call = blackScholes(self.S, self.K, self.r, self.T, 10, "c")
        self.assertGreater(large_vol_call, self.S * 0.5)

        # Test deep in-the-money call
        deep_itm_call = blackScholes(self.S, self.K * 0.1, self.r, self.T, self.sigma, "c")
        self.assertAlmostEqual(deep_itm_call, self.S - self.K * 0.1 * np.exp(-self.r * self.T), places=2)

        # Test deep out-of-the-money put
        deep_otm_put = blackScholes(self.S, self.K * 10, self.r, self.T, self.sigma, "p")
        self.assertAlmostEqual(deep_otm_put, 0, places=2)

        # Additional edge case tests
        # Test deep out-of-the-money call
        deep_otm_call = blackScholes(self.S, self.K * 10, self.r, self.T, self.sigma, "c")
        self.assertAlmostEqual(deep_otm_call, 0, places=2)

        # Test deep in-the-money put
        deep_itm_put = blackScholes(self.S, self.K * 0.1, self.r, self.T, self.sigma, "p")
        self.assertAlmostEqual(deep_itm_put, self.K * 0.1 * np.exp(-self.r * self.T) - self.S, places=2)

        # Test at-the-money option
        atm_call = blackScholes(self.S, self.S, self.r, self.T, self.sigma, "c")
        atm_put = blackScholes(self.S, self.S, self.r, self.T, self.sigma, "p")
        self.assertAlmostEqual(atm_call - atm_put, 0, places=4)

if __name__ == '__main__':
    unittest.main()