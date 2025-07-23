import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import sys
import os

# Adjust path to import from parent directories
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.battery.battery import Battery
from src.optimization.rolling_intrinsic import RollingIntrinsicStrategy


class BatteryTradingEnv(gym.Env):
    """
    A custom Gymnasium environment for daily battery trading with future commitments.

    This environment models a full trading day. At each trading timestep, the agent
    can place trades (buy/sell) for any of the future delivery periods within that day.

    - **Episode**: A single trading day.
    - **Action**: A vector of trade sizes (MW) for each delivery period.
    - **Observation**: The battery's state, current market prices for all
      delivery periods, and the agent's current commitments.
    - **Reward**: A mark-to-market profit calculation at each step.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        config,
        daily_market_data: pd.DataFrame,
        ri_strategy: RollingIntrinsicStrategy = None,
    ):
        super(BatteryTradingEnv, self).__init__()

        self.config = config
        self.battery = Battery(config)
        self.ri_strategy = ri_strategy  # Store the RI strategy

        # --- Define Fixed Day Structure ---
        # A day is assumed to have 96 quarter-hourly periods.
        # This ensures the observation and action spaces have a fixed size.
        self.n_delivery_periods = config.get("N_timesteps", 96)
        self.n_trading_steps = config.get("N_trading_steps", 96)
        self.timestep_h = self.config.get("time_step", 0.25)

        # This will be populated by set_daily_data
        self.daily_data = pd.DataFrame()
        self.trading_times = []
        self.delivery_period_map = {}
        self.previous_market_prices = np.zeros(
            self.n_delivery_periods, dtype=np.float32
        )

        # --- Action and Observation Spaces (now with fixed size) ---
        self.action_space = spaces.Box(
            low=-self.battery.capacity_normalized,
            high=self.battery.capacity_normalized,
            shape=(self.n_delivery_periods,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict(
            {
                "soc": spaces.Box(
                    low=0,
                    high=1.0,  # Correctly normalized
                    shape=(1,),
                    dtype=np.float32,
                ),
                "cycles_used": spaces.Box(
                    low=0,
                    high=4
                    * self.battery.N_daily_cycles_max,  # Use fixed max cycles
                    shape=(1,),
                    dtype=np.float32,
                ),
                "commitments": spaces.Box(
                    low=-self.battery.capacity_normalized,  # Use normalized bounds
                    high=self.battery.capacity_normalized,  # Use normalized bounds
                    shape=(self.n_delivery_periods,),
                    dtype=np.float32,
                ),
                "market_prices": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.n_delivery_periods,),
                    dtype=np.float32,
                ),
                "market_availability": spaces.MultiBinary(
                    self.n_delivery_periods
                ),
                "current_time_idx": spaces.Discrete(self.n_trading_steps),
                # "ri_schedule": spaces.Box(
                #     low=-2 * self.battery.capacity_normalized,
                #     high=2 * self.battery.capacity_normalized,
                #     shape=(self.n_delivery_periods,),
                #     dtype=np.float32,
                # ),
            }
        )

        # Initialize with the provided data
        self.set_daily_data(daily_market_data)

    def set_daily_data(
        self, daily_market_data: pd.DataFrame, ri_schedule=None
    ):
        """
        Configures the environment with data for a specific trading day.
        """
        self.daily_data = daily_market_data.copy()

        if (
            self.daily_data.empty
            or "delivery_start" not in self.daily_data.columns
        ):
            self.trading_times = []
            self.delivery_period_map = {}
        else:
            self.trading_times = sorted(
                self.daily_data["delivery_start"].unique()
            )
            if len(self.trading_times) > self.n_trading_steps:
                self.trading_times = self.trading_times[: self.n_trading_steps]

            self.delivery_period_map = {
                dp: i
                for i, dp in enumerate(
                    sorted(self.daily_data["delivery_start"].unique())
                )
            }

        # if ri_schedule is not None:
        #     self.full_ri_schedule = ri_schedule
        # else:
        #     self.full_ri_schedule = np.zeros(
        #         (self.n_trading_steps, self.n_delivery_periods)
        #     )

    def _get_obs(self):
        if not self.trading_times:
            # Return a zero observation if no data is loaded
            return {
                "soc": np.array([self.battery.current_soc], dtype=np.float32),
                "cycles_used": np.array(
                    [self.battery.N_cycles_till_now], dtype=np.float32
                ),
                "commitments": np.zeros(
                    self.n_delivery_periods, dtype=np.float32
                ),
                "market_prices": np.zeros(
                    self.n_delivery_periods, dtype=np.float32
                ),
                "market_availability": np.zeros(
                    self.n_delivery_periods, dtype=np.int8
                ),
                "current_time_idx": self.current_step_idx,
                # "ri_schedule": np.zeros(
                #     self.n_delivery_periods, dtype=np.float32
                # ),
            }

        # Handle case where we are past the last trading step
        current_step_idx = min(
            self.current_step_idx, len(self.trading_times) - 1
        )
        current_time = self.trading_times[current_step_idx]

        # Get current market prices for all delivery periods
        market_prices = np.full(
            self.n_delivery_periods, np.nan, dtype=np.float32
        )
        current_trades = self.daily_data.loc[
            self.daily_data["traded"] == current_time
        ]

        # Use a map for efficient price lookup
        price_map = pd.Series(
            current_trades["VWAP"].values,
            index=current_trades["delivery_start"],
        )
        valid_prices_map = price_map[
            price_map.index.isin(self.delivery_period_map)
        ]
        indices = [
            self.delivery_period_map[dt] for dt in valid_prices_map.index
        ]
        market_prices[indices] = valid_prices_map.values

        # Create a binary mask for market availability (1 if price is not NaN)
        availability_mask = ~np.isnan(market_prices)

        # For the observation, replace any NaNs with 0. The agent sees a value of 0
        # for periods where there is no active market.
        obs_prices = np.nan_to_num(market_prices, nan=0.0)

        # # Get the RI schedule for the current step
        # if hasattr(self, "full_ri_schedule") and self.current_step_idx < len(
        #     self.full_ri_schedule
        # ):
        #     ri_schedule_for_step = self.full_ri_schedule[self.current_step_idx]
        # else:
        #     ri_schedule_for_step = np.zeros(
        #         self.n_delivery_periods, dtype=np.float32
        #     )

        return {
            "soc": np.array([self.battery.current_soc], dtype=np.float32),
            "cycles_used": np.array(
                [self.battery.N_cycles_till_now], dtype=np.float32
            ),
            "commitments": self.commitments.copy(),
            "market_prices": obs_prices,
            "market_availability": availability_mask.astype(np.int8),
            "current_time_idx": self.current_step_idx,
            # "ri_schedule": ri_schedule_for_step,
        }

    def _get_feasible_action_and_payoff(
        self, action, start_soc, market_prices
    ):
        """
        Calculates a physically feasible action schedule based on an intended
        action, and computes its total expected payoff.
        """
        temp_soc = start_soc
        payoff = 0.0
        feasible_action = np.zeros_like(action)

        # Simulate dispatch for all future delivery periods based on the proposed action
        for i in range(self.n_delivery_periods):
            # Only consider actions from the current step forward
            if i < self.current_step_idx:
                continue

            dispatch_mw_normalized = action[i]
            dispatch_price = market_prices[i]

            if abs(dispatch_mw_normalized) > 1e-6 and not np.isnan(
                dispatch_price
            ):
                if dispatch_mw_normalized > 0:  # Planned charge
                    soc_increment = dispatch_mw_normalized * self.timestep_h
                    actual_soc_increment = min(soc_increment, 1.0 - temp_soc)

                    # Convert feasible energy change back to feasible power
                    feasible_power = actual_soc_increment / self.timestep_h
                    feasible_action[i] = feasible_power

                    cost = (
                        (actual_soc_increment / self.battery.charge_efficiency)
                        * dispatch_price
                        * self.battery.size_energy
                    )
                    payoff -= cost
                    temp_soc += actual_soc_increment
                else:  # Planned discharge
                    soc_decrement = (
                        abs(dispatch_mw_normalized) * self.timestep_h
                    )
                    actual_soc_decrement = min(soc_decrement, temp_soc)

                    # Convert feasible energy change back to feasible power (negative for discharge)
                    feasible_power = -(actual_soc_decrement / self.timestep_h)
                    feasible_action[i] = feasible_power

                    revenue = (
                        (
                            actual_soc_decrement
                            * self.battery.discharge_efficiency
                        )
                        * dispatch_price
                        * self.battery.size_energy
                    )
                    payoff += revenue
                    temp_soc -= actual_soc_decrement

        return feasible_action, payoff

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.battery = Battery(self.config)
        self.commitments = np.zeros(self.n_delivery_periods, dtype=np.float32)
        self.current_step_idx = 0
        self.trades_log = []

        # Initialize cumulative tracking for the episode
        self.cumulative_raw_profit = 0.0

        # Reset previous market prices for MTM calculation
        self.previous_market_prices = np.zeros(
            self.n_delivery_periods, dtype=np.float32
        )

        # Recalculate the RI schedule when the environment is reset with new data
        if not self.daily_data.empty and self.ri_strategy:
            self.set_daily_data(self.daily_data)

        obs, info = self._get_obs(), {}

        # Store initial prices for the first MTM calculation in the first step
        if "market_prices" in obs:
            self.previous_market_prices = obs["market_prices"].copy()

        return obs, info

    def step(self, action):
        if not self.trading_times:
            # If there's no data, take no action and terminate
            terminated = True
            reward = 0
            obs = self._get_obs()
            return obs, reward, terminated, False, {}

        # Get the raw market prices for the current timestep to enforce trading rules.
        # This is separate from the agent's observation, which is forward-filled.
        current_time = self.trading_times[self.current_step_idx]
        raw_market_prices = np.full(
            self.n_delivery_periods, np.nan, dtype=np.float32
        )
        current_trades = self.daily_data.loc[
            self.daily_data["traded"] == current_time
        ]
        price_map = pd.Series(
            current_trades["VWAP"].values,
            index=current_trades["delivery_start"],
        )
        valid_prices_map = price_map[
            price_map.index.isin(self.delivery_period_map)
        ]
        indices = [
            self.delivery_period_map[dt] for dt in valid_prices_map.index
        ]
        raw_market_prices[indices] = valid_prices_map.values
        current_market_prices = np.nan_to_num(raw_market_prices, nan=0.0)

        # --- Dynamic Action Re-scaling ---
        # The agent's action is in [-1, 1]. We scale it to the true valid range
        # which is now in terms of normalized power.
        power_limit = self.battery.capacity_normalized

        # For each delivery period, calculate the valid trading range
        # Min trade: From current commitment down to -power_limit
        # Max trade: From current commitment up to +power_limit
        min_trade = -power_limit - self.commitments
        max_trade = power_limit - self.commitments

        # Scale the normalized action to this dynamic range
        # Positive actions (buy) scale between [0, max_trade]
        # Negative actions (sell) scale between [min_trade, 0]
        scaled_action = np.zeros_like(action)
        scaled_action[action > 0] = action[action > 0] * max_trade[action > 0]
        scaled_action[action <= 0] = (
            action[action <= 0] * -min_trade[action <= 0]
        )

        # Use the scaled action for all subsequent logic
        action = scaled_action

        # Disallow trades for periods where there is no price, using the raw market data
        action[np.isnan(raw_market_prices)] = 0

        # From the agent's proposed action, calculate what is actually feasible
        # given SOC limits, and get the payoff of that feasible action.
        feasible_action, reward = self._get_feasible_action_and_payoff(
            action, self.battery.current_soc, current_market_prices
        )

        # Track cumulative raw trading profit (only cash from realized dispatch)
        self.cumulative_raw_profit += reward

        # Update commitments with the physically feasible action
        self.commitments += feasible_action

        # --- Handle Dispatch and Realized Profit Calculation ---
        current_time = self.trading_times[self.current_step_idx]

        if current_time in self.delivery_period_map:
            delivery_idx = self.delivery_period_map[current_time]
            dispatch_mw_normalized = self.commitments[delivery_idx]

            # Get the price for this delivery period
            dispatch_price = raw_market_prices[delivery_idx]

            if abs(dispatch_mw_normalized) > 1e-6 and not np.isnan(
                dispatch_price
            ):
                if dispatch_mw_normalized > 0:  # Bought power -> charge
                    # SOC increase is based on energy bought
                    soc_increment = dispatch_mw_normalized * self.timestep_h
                    actual_soc_increment = min(
                        soc_increment, 1.0 - self.battery.current_soc
                    )

                    self.battery.update_current_soc(
                        self.battery.current_soc + actual_soc_increment
                    )
                    self.battery.N_cycles_till_now += (
                        actual_soc_increment / 2.0
                    )

                else:  # Sold power -> discharge
                    # SOC decrease is based on energy sold
                    soc_decrement = (
                        abs(dispatch_mw_normalized) * self.timestep_h
                    )
                    actual_soc_decrement = min(
                        soc_decrement, self.battery.current_soc
                    )

                    self.battery.update_current_soc(
                        self.battery.current_soc - actual_soc_decrement
                    )
                    self.battery.N_cycles_till_now += (
                        actual_soc_decrement / 2.0
                    )

        # Move to the next trading step
        self.current_step_idx += 1

        # The episode terminates if we have processed all available trading times for the day
        # or if we have reached the maximum number of steps.
        terminated = self.current_step_idx >= len(self.trading_times)

        # Calculate ALL penalties only at the END of the episode
        terminal_penalty = 0.0
        if terminated:
            # Use ACTUAL physical cycles executed, not trading commitments
            # This is the correct constraint - the physical battery limitation
            actual_cycles_executed = self.battery.N_cycles_till_now

            # --- Terminal Penalty for Constraint Violation ---
            if actual_cycles_executed > self.battery.N_daily_cycles_max:
                violation = (
                    actual_cycles_executed - self.battery.N_daily_cycles_max
                )
                violation_penalty = violation * self.config.get(
                    "cycle_violation_penalty", 5000
                )
                terminal_penalty += violation_penalty

            # --- Terminal Penalty for Under-utilization ---
            # Only penalize if significantly underused (less than 50% of limit)
            if actual_cycles_executed < self.battery.N_daily_cycles_max * 0.5:
                underuse = (
                    self.battery.N_daily_cycles_max * 0.5
                ) - actual_cycles_executed
                underuse_penalty = underuse * self.config.get(
                    "cycle_underuse_penalty", 1000
                )
                terminal_penalty += underuse_penalty

            # Apply the terminal penalty
            reward -= terminal_penalty

            # The episode is over. Return final observation.
            obs = {
                "soc": np.array([self.battery.current_soc], dtype=np.float32),
                "cycles_used": np.array(
                    [self.battery.N_cycles_till_now], dtype=np.float32
                ),
                "commitments": self.commitments.copy(),
                "market_prices": np.zeros(
                    self.n_delivery_periods, dtype=np.float32
                ),
                "market_availability": np.zeros(
                    self.n_delivery_periods, dtype=np.int8
                ),
                "current_time_idx": self.current_step_idx - 1,
                # "ri_schedule": np.zeros(
                #     self.n_delivery_periods, dtype=np.float32
                # ),
            }
        else:
            obs = self._get_obs()

        # Create info dictionary with detailed breakdown
        info = {
            "step_trading_profit": reward,
            "step_reward": reward,
            "step_commitment_penalty": 0,  # No per-step penalties anymore
            "step_terminal_penalty": terminal_penalty if terminated else 0,
            "cumulative_raw_profit": self.cumulative_raw_profit,
            "total_penalties": terminal_penalty if terminated else 0,
        }

        return obs, reward, terminated, False, info
