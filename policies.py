"""
Markowitz Modern Portfolio Theory (MPT) Baseline Policy.
"""

import os
import numpy as np
import pandas as pd
import config


class MarkowitzPolicy:
    """
    Mean-Variance Optimization Policy using Monte Carlo Portfolio Generation.
    Selects optimal asset weights over rolling historical windows.
    """

    def __init__(self, n_ports=100, fixed_portfolio=False, coins=None, sample_file="portfolios_sample.csv"):
        self.n_ports = n_ports
        self.fixed_portfolio = fixed_portfolio
        self.coins = coins if coins is not None else list(config.COINS)
        self.sample_file = sample_file
        self.portfolios = self.init_portfolios()

    def init_portfolios(self):
        """Initialize candidate portfolios on a simplex (summing to 100%)."""
        if self.fixed_portfolio and os.path.exists(self.sample_file):
            ps = pd.read_csv(self.sample_file)
        else:
            np.random.seed(42)
            portfolios = []
            for _ in range(self.n_ports):
                # Generate random weights summing to 100
                weights = np.random.dirichlet(np.ones(len(self.coins))) * 100.0
                portfolios.append(weights)
            ps = pd.DataFrame(portfolios, columns=self.coins)
        return ps

    def get_action(self, memory, window=30):
        """
        Calculates optimal allocation based on the past `window` time steps.
        
        Parameters:
            memory: DataFrame containing price histories for the assets.
            window: Number of recent periods to use for mean/variance estimation.

        Returns:
            action: 1D numpy array of length len(coins) + 1 [cash_weight, *asset_weights]
        """
        df_mem = memory.copy().reset_index(drop=True).tail(window)
        
        # Handle long-format data (e.g. from raw dataframe with 'coin' and 'close')
        if "coin" in df_mem.columns and "close" in df_mem.columns:
            index_col = "date" if "date" in df_mem.columns else df_mem.index
            df_mem = df_mem.pivot(index=index_col, columns="coin", values="close")

        # Match price columns (either '<coin>_close' or '<coin>')
        cols_to_use = []
        for c in self.coins:
            if f"{c}_close" in df_mem.columns:
                cols_to_use.append(f"{c}_close")
            elif c in df_mem.columns:
                cols_to_use.append(c)
            elif f"{c}_open" in df_mem.columns:
                cols_to_use.append(f"{c}_open")
            else:
                matching = [col for col in df_mem.columns if c in col]
                if matching:
                    cols_to_use.append(matching[0])
                else:
                    # If columns cannot be determined, fallback to equal allocation
                    eq = 1.0 / (len(self.coins) + 1)
                    return np.ones(len(self.coins) + 1) * eq

        prices = df_mem[cols_to_use].values  # shape: (T, num_coins)
        if len(prices) < 2:
            # Fallback equal allocation
            eq = 1.0 / (len(self.coins) + 1)
            return np.ones(len(self.coins) + 1) * eq

        # Vectorized portfolio valuation: (T, num_coins) @ (num_coins, n_ports) -> (T, n_ports)
        weights_matrix = self.portfolios[self.coins].values.T / 100.0
        portfolio_values = np.dot(prices, weights_matrix)

        # Compute differences (returns): shape: (T-1, n_ports)
        returns = np.diff(portfolio_values, axis=0)

        means = np.mean(returns, axis=0)
        stds = np.std(returns, axis=0)
        stds = np.where(stds == 0, 1e-8, stds)

        # Select portfolio: lowest 25% volatility with maximum mean return
        df_stats = pd.DataFrame({'mean': means, 'std': stds})
        q25_std = df_stats['std'].quantile(0.25)
        low_risk_subset = df_stats[df_stats['std'] <= max(q25_std, 1e-7)]

        if not low_risk_subset.empty:
            best_idx = low_risk_subset['mean'].idxmax()
        else:
            # Fallback to overall highest Sharpe ratio
            sharpe = means / stds
            best_idx = np.argmax(sharpe)

        asset_weights = self.portfolios.loc[best_idx, self.coins].values / 100.0
        # Prefix 0 for cash allocation (fully invested across chosen assets)
        action = np.hstack(([0.0], asset_weights))
        return action
