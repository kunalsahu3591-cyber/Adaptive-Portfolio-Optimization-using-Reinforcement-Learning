"""
Reinforcement Learning Environment for Adaptive Portfolio Management.
Inherits from TF-Agents PyEnvironment.
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

import config


tf.compat.v1.enable_v2_behavior()


class PortfolioEnv(py_environment.PyEnvironment):
    """
    Continuous-action Portfolio Optimization Environment.
    
    Actions:
        Continuous simplex vector of length (len(COINS) + 1):
        [weight_cash, weight_coin_1, weight_coin_2, ...]
        
    Observations:
        Normalized OHLCV and rolling statistical metrics across coins (flattened).
        
    Reward:
        Change in total portfolio valuation from step t to t+1.
    """

    def __init__(self, data_file=None):
        super(PortfolioEnv, self).__init__()
        self.data_file = data_file or config.FILE
        
        num_actions = len(config.COINS) + 1
        num_obs = len(config.OBS_COLS)

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(num_actions,),
            dtype=np.float64,
            minimum=0.0,
            maximum=1.0,
            name='action'
        )

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(num_obs,),
            dtype=np.float64,
            minimum=np.array(config.OBS_COLS_MIN, dtype=np.float64),
            maximum=np.array(config.OBS_COLS_MAX, dtype=np.float64),
            name='observation'
        )

        self.time_delta = pd.Timedelta(config.TIME_DELTA_MINUTES, unit='m')
        self._episode_ended = False
        self.reset()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.memory_return = pd.DataFrame(columns=[t + "_close" for t in config.COINS])
        self._episode_ended = False
        self.index = 0
        self.init_cash = config.INITIAL_CASH
        self.current_cash = self.init_cash
        self.current_value = self.init_cash
        self.previous_value = self.init_cash
        self.step_reward = 0.0

        self.previous_price = {}
        self.old_dict_coin_price_1 = {}
        self.old_dict_coin_price_2 = {}

        # Default allocation: 100% cash
        self.money_split_ratio = np.zeros(len(config.COINS) + 1, dtype=np.float64)
        self.money_split_ratio[0] = 1.0

        # Load dataset
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(
                f"Data file '{self.data_file}' not found. Please run pre_process.py first."
            )

        self.df = pd.read_csv(self.data_file)
        
        # Robust date parsing
        if pd.api.types.is_numeric_dtype(self.df["date"]):
            self.df["date"] = pd.to_datetime(self.df["date"], unit='s')
        else:
            self.df["date"] = pd.to_datetime(self.df["date"])

        self.df = self.df[self.df["coin"].isin(config.COINS)].sort_values("date").reset_index(drop=True)

        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(self.df[config.SCOLS].values)

        self.max_index = self.df.shape[0]
        max_start = max(3, self.max_index - config.EPISODE_LENGTH - 3)
        start_point = (np.random.choice(np.arange(3, max_start)) // 3) * 3
        end_point = start_point + (config.EPISODE_LENGTH // 3) * 3
        self.df = self.df.loc[start_point:end_point + 2].reset_index(drop=True)

        self.init_time = self.df.loc[0, "date"]
        self.current_time = self.init_time
        
        self.dfslice = self.df[
            (self.df["coin"].isin(config.COINS)) &
            (self.df["date"] >= self.current_time) &
            (self.df["date"] < self.current_time + self.time_delta)
        ].copy().drop_duplicates("coin")

        self.current_stock_num_distribution = self.calculate_actual_shares_from_money_split()
        self.current_stock_money_distribution, self.current_value = self.calculate_money_from_num_stocks()
        self.previous_value = self.current_value
        self.money_split_ratio = self.normalize_money_dist()

        obs = self.get_observations()
        self._state = obs[config.OBS_COLS].values.flatten().astype(np.float64)

        # Clip state within bounded specs
        self._state = np.clip(self._state, config.OBS_COLS_MIN, config.OBS_COLS_MAX)

        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        action_sum = np.sum(action)
        if action_sum <= 1e-3:
            self.money_split_ratio = np.ones(len(action), dtype=np.float64) / len(action)
        else:
            self.money_split_ratio = (np.array(action, dtype=np.float64) / action_sum)

        self.current_stock_num_distribution = self.calculate_actual_shares_from_money_split()
        self.step_time()
        self.index += 1

        obs = self.get_observations()
        self._state = obs[config.OBS_COLS].values.flatten().astype(np.float64)
        self._state = np.clip(self._state, config.OBS_COLS_MIN, config.OBS_COLS_MAX)

        reward = float(self.step_reward)
        self._episode_ended = (self.index >= config.EPISODE_LENGTH // 3)

        if self._episode_ended:
            return ts.termination(self._state, reward=0.0)
        else:
            return ts.transition(self._state, reward=reward, discount=1.0)

    def step_time(self):
        self.current_time += self.time_delta
        slice_match = self.df[
            (self.df["coin"].isin(config.COINS)) &
            (self.df["date"] >= self.current_time) &
            (self.df["date"] < self.current_time + self.time_delta)
        ].copy().drop_duplicates("coin")

        if not slice_match.empty:
            self.dfslice = slice_match

        self.previous_value = self.current_value
        self.current_stock_money_distribution, self.current_value = self.calculate_money_from_num_stocks()
        self.money_split_ratio = self.normalize_money_dist()
        self.step_reward = self.current_value - self.previous_value

    def get_observations(self):
        dfs = pd.DataFrame()
        for coin_name, grp in self.dfslice.groupby("coin"):
            scaled_vals = self.scaler.transform(grp[config.SCOLS].values)
            tempdf = pd.DataFrame(scaled_vals, columns=[f"{coin_name}_{c}" for c in config.SCOLS])
            if dfs.empty:
                dfs = tempdf
            else:
                dfs = dfs.merge(tempdf, right_index=True, left_index=True, how='inner')

        # Fallback if any coin feature is missing from current slice
        for col in config.OBS_COLS:
            if col not in dfs.columns:
                dfs[col] = 0.0

        return dfs

    def calculate_actual_shares_from_money_split(self):
        dict_coin_price = self.dfslice[["coin", "open"]].set_index("coin").to_dict()["open"]
        num_shares = []
        for i, c in enumerate(config.COINS):
            price = dict_coin_price.get(c, self.old_dict_coin_price_1.get(c, 1.0))
            if price <= 0:
                price = 1.0
            num_shares.append(self.money_split_ratio[i + 1] * self.current_value // price)

        self.current_cash = self.money_split_ratio[0] * self.current_value
        for c in dict_coin_price:
            self.old_dict_coin_price_1[c] = dict_coin_price[c]

        return num_shares

    def calculate_money_from_num_stocks(self):
        money_dist = [self.current_cash]
        dict_coin_price = self.dfslice[["coin", "open"]].set_index("coin").to_dict()["open"]
        
        for i, c in enumerate(config.COINS):
            price = dict_coin_price.get(c, self.old_dict_coin_price_2.get(c, 1.0))
            money_dist.append(self.current_stock_num_distribution[i] * price)

        for c in dict_coin_price:
            self.old_dict_coin_price_2[c] = dict_coin_price[c]

        total_val = float(sum(money_dist))
        return money_dist, total_val

    def normalize_money_dist(self):
        if self.current_value <= 0:
            return np.ones(len(self.current_stock_money_distribution)) / len(self.current_stock_money_distribution)
        return [c / self.current_value for c in self.current_stock_money_distribution]


# Backward compatibility alias
CardGameEnv = PortfolioEnv
