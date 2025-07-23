import numpy as np
from optimization.base_strategy import BaseStrategy
import pyomo.environ as pyo
import os
import logging
from optimization.LSM import find_closest_indices_vectorized

# # Get the logger specifically for gurobipy
# gurobipy_logger = logging.getLogger("gurobipy")

# # Set its level to WARNING or ERROR to mute INFO and DEBUG messages
# gurobipy_logger.setLevel(logging.WARNING)

eps = 1e-2  # Small value to avoid numerical issues in constraints
big_M = 1e6  # Large value to replace inf


class RollingIntrinsicStrategy(BaseStrategy):
    """
    Rolling intrinsic strategy class to be used for optimization of battery trading.
    This model uses an Ornstein-Uhlenbeck process to model price paths and
    generate correlated paths for different delivery start times.
    """

    def __init__(self, battery, config, lsm_strategy=None):
        # TODO: add logging statements
        """
        Initialize the RollingIntrinsic strategy.

        Args:
            battery: An instance of your Battery class.
            config: Dictionary of configuration parameters.
        """
        super().__init__(battery, config)
        self.optimized_paths = None
        self.threads = config.get("threads", None)
        self.alpha = config.get(
            "alpha", 1
        )  # Weight for continuation value guidance
        self.N_timesteps = config.get(
            "N_timesteps", 96
        )  # Standard day length like LSM
        self.lsm_strategy = lsm_strategy

    def run(self, price_path):
        """
        Execute the trading strategy over the simulation period.
        Price path should be a list where price_path[t] contains available prices from time t onwards.
        """
        # Store the price paths structure
        self.price_paths = price_path

        # For compatibility, also store the total number of trading timesteps
        self.N_trading_timesteps = len(price_path)

        # Initialize battery schedules
        self.battery.power_charging_committed = np.zeros(
            self.N_trading_timesteps
        )
        self.battery.power_discharging_committed = np.zeros(
            self.N_trading_timesteps
        )
        self.battery.soc_schedule = self.battery.initial_soc * np.ones(
            self.N_trading_timesteps + 1
        )
        self.results = np.zeros(self.N_trading_timesteps)
        self.profit_results = np.zeros(self.N_trading_timesteps)

        RI_results = self.optimize_on_path()
        RI_results["price_path"] = self.price_paths
        return RI_results

    def optimize_on_path(self):
        """
        Optimize battery operation on price path.
        """

        price_paths = self.price_paths

        if os.environ.get("USER") == "rickvanvoorbergen":
            solver = self.config.get("solver", "highs")
        else:
            solver = self.config.get("solver", "highs")

        solver_factory = pyo.SolverFactory(solver)

        # Initialize battery schedule and SOC
        self.battery.power_committed = np.zeros(self.N_trading_timesteps)
        self.battery.power_actions = np.zeros(
            (self.N_trading_timesteps, self.N_trading_timesteps)
        )
        self.battery.power_schedules = np.zeros(
            (
                self.N_trading_timesteps,
                self.N_trading_timesteps,
            )
        )
        self.battery.soc_schedules = np.zeros(
            [
                self.N_trading_timesteps,
                self.N_trading_timesteps + 1,
            ]
        )
        # Add cycle schedule storage
        self.battery.cycle_schedules = np.zeros(
            [
                self.N_trading_timesteps,
                self.N_trading_timesteps + 1,
            ]
        )
        self.traded_prices = [[] for _ in range(self.N_trading_timesteps)]

        # Track variables for CV integration
        last_optimal_cv_schedule = None

        # Optimize on each timestep
        for t in range(self.N_trading_timesteps):
            remaining_price_path = self.price_paths[t]
            continuation_value_data = None
            baseline_cv_schedule = None

            # Generate continuation value data from LSM if available
            if (
                self.lsm_strategy
                and self.lsm_strategy.fitted_polynomials is not None
            ):
                is_lsm_valid = True
                continuation_value_data = []

                # Build CV functions for the current window t..t+W
                for i in range(len(remaining_price_path) - 1):  # i from 0 to W
                    poly_t_abs = t + i

                    if (
                        poly_t_abs
                        >= self.lsm_strategy.fitted_polynomials.shape[0]
                    ):
                        is_lsm_valid = False
                        break

                    price_for_cv = remaining_price_path[i]

                    # Skip timesteps with NaN prices
                    if np.isnan(price_for_cv):
                        continuation_value_data.append(None)
                        continue

                    # Clip price for polynomial evaluation
                    min_p, max_p = self.lsm_strategy.price_clipping_bounds[
                        poly_t_abs
                    ]
                    price_for_poly_eval = np.clip(price_for_cv, min_p, max_p)

                    soc_levels = self.lsm_strategy.SOC_levels
                    cycle_levels = self.lsm_strategy.cycle_levels
                    polys_grid = self.lsm_strategy.fitted_polynomials[
                        poly_t_abs, :, :
                    ]

                    # Create CV grid by evaluating polynomials
                    cv_grid = np.array(
                        [
                            [poly(price_for_poly_eval) for poly in row]
                            for row in polys_grid
                        ]
                    )

                    continuation_value_data.append(
                        (soc_levels, cycle_levels, cv_grid)
                    )

                if is_lsm_valid:
                    # Build baseline CV schedule
                    baseline_cv_schedule_list = []

                    if t > 0 and last_optimal_cv_schedule is not None:
                        # Use stored CV values from previous iteration as baseline
                        # Shift by 1 timestep since we've moved forward
                        remaining_cv_length = len(remaining_price_path) - 1
                        available_cv_length = len(last_optimal_cv_schedule) - 1

                        for i in range(remaining_cv_length):
                            if i + 1 < available_cv_length:
                                # Use CV from previous optimization (shifted by 1)
                                cv_value = last_optimal_cv_schedule[i + 1]
                                # # Ensure we don't add None values
                                # if cv_value is not None:
                                baseline_cv_schedule_list.append(cv_value)
                                # else:
                                #     baseline_cv_schedule_list.append(0.0)
                            else:
                                (
                                    soc_levels_i,
                                    cycle_levels_i,
                                    cv_grid_i,
                                ) = continuation_value_data[i]
                                baseline_soc_i = self.battery.soc_schedule[
                                    t + i
                                ]
                                current_cycles = getattr(
                                    self.battery, "N_cycles_till_now", 0.0
                                )

                                # Use closest LSM levels (same as first iteration)
                                soc_idx = find_closest_indices_vectorized(
                                    soc_levels_i,
                                    np.array([baseline_soc_i]),
                                )[0]
                                cycle_idx = find_closest_indices_vectorized(
                                    cycle_levels_i,
                                    np.array([current_cycles]),
                                )[0]
                                baseline_cv_i = cv_grid_i[soc_idx, cycle_idx]
                                baseline_cv_schedule_list.append(baseline_cv_i)
                    else:
                        # First iteration: use closest SOC and cycle levels from LSM
                        for i in range(len(remaining_price_path) - 1):
                            if continuation_value_data[i] is None:
                                baseline_cv_schedule_list.append(None)
                                continue

                            (soc_levels_i, cycle_levels_i, cv_grid_i) = (
                                continuation_value_data[i]
                            )
                            baseline_soc_i = self.battery.soc_schedule[t + i]
                            current_cycles = getattr(
                                self.battery, "N_cycles_till_now", 0.0
                            )

                            # Find closest SOC level index
                            soc_idx = find_closest_indices_vectorized(
                                soc_levels_i, np.array([baseline_soc_i])
                            )[0]

                            # Find closest cycle level index
                            cycle_idx = find_closest_indices_vectorized(
                                cycle_levels_i, np.array([current_cycles])
                            )[0]

                            # Use CV value directly from LSM grid (no interpolation)
                            baseline_cv_i = cv_grid_i[soc_idx, cycle_idx]
                            baseline_cv_schedule_list.append(baseline_cv_i)

                    baseline_cv_schedule = np.array(baseline_cv_schedule_list)
                else:
                    continuation_value_data = None
                    baseline_cv_schedule = None

            # Define pyomo optimization model for the current timestep.
            model = self.define_pyomo_model(
                price_path=remaining_price_path,
                optimization_window_timesteps=len(remaining_price_path),
                t=t,
                continuation_value_data=continuation_value_data,
                baseline_cv_schedule=baseline_cv_schedule,
            )

            # Solve optimization problem.
            solver_options = {}
            # if self.threads is not None:
            #     solv.
            results = solver_factory.solve(model)

            # Check if the solve was successful before accessing results
            if (results.solver.status == pyo.SolverStatus.ok) and (
                results.solver.termination_condition
                == pyo.TerminationCondition.optimal
            ):
                logging.info(f"Solver successful at timestep {t}")
                # Load the solution only if optimal
                model.solutions.load_from(results)
                # Update battery SOC and power schedule.
                # Access values directly from the model instance.
                soc_values = np.array(list(model.soc.get_values().values()))
                power_charge_values = np.array(
                    list(model.power_charge.get_values().values())
                )
                power_discharge_values = np.array(
                    list(model.power_discharge.get_values().values())
                )
                power_flow_into_battery_values = np.array(
                    list(model.power_flow_into_battery.get_values().values())
                )
                power_flow_out_of_battery_values = np.array(
                    list(model.power_flow_out_of_battery.get_values().values())
                )
                # Extract total cycles used from the model
                total_cycles_used = model.N_cycles.value

                self.battery.update_soc_schedule(t, soc_values)
                self.battery.update_power_committed(
                    t,
                    power_flow_into_battery_values,
                    power_flow_out_of_battery_values,
                )

                # Calculate cycle schedule based on actual power flows
                window_len = len(power_flow_into_battery_values)
                if window_len > 0:
                    start_cycles = self.battery.N_cycles_till_now
                    cycle_schedule = np.zeros(window_len + 1)
                    cycle_schedule[0] = start_cycles

                    for i in range(window_len):
                        cycle_increment = (
                            (
                                power_flow_into_battery_values[i]
                                + power_flow_out_of_battery_values[i]
                            )
                            * self.battery.time_step
                            / 2.0
                        )
                        cycle_schedule[i + 1] = (
                            cycle_schedule[i] + cycle_increment
                        )

                    self.battery.cycle_schedules[t, : len(cycle_schedule)] = (
                        cycle_schedule
                    )

                window_len = len(power_charge_values)
                for i in range(window_len):
                    price = remaining_price_path[i]
                    p_charge = power_charge_values[i]
                    p_discharge = power_discharge_values[i]

                    if t + i < len(self.traded_prices):
                        if p_charge > eps:
                            self.traded_prices[t + i].append(
                                {"price": price, "power": p_charge}
                            )
                        elif p_discharge > eps:
                            self.traded_prices[t + i].append(
                                {"price": price, "power": -p_discharge}
                            )

                # Update number of cycles in the battery based on committed power
                self.battery.N_cycles_till_now = (
                    (
                        sum(self.battery.power_discharging_committed[: t + 1])
                        + sum(self.battery.power_charging_committed[: t + 1])
                    )
                    * self.battery.time_step
                    / 2.0
                )

                # Update results.
                self.results[t] = pyo.value(model.objective)
                self.profit_results[t] = pyo.value(model.total_profit)

                # Update power actions schedule
                self.battery.power_actions[t, t:] = (
                    power_discharge_values - power_charge_values
                )

                # Store CV schedule if available
                if continuation_value_data:
                    last_optimal_cv_schedule = np.array(
                        list(model.cv.get_values().values())
                    )
                    last_optimal_soc_schedule = np.array(
                        list(model.soc.get_values().values())
                    )

            else:
                print(f"Solver failed at timestep {t}:")
                print(f"  Status: {results.solver.status}")
                print(
                    f"  Termination condition: {results.solver.termination_condition}"
                )

                # Check if infeasible
                if (
                    results.solver.termination_condition
                    == pyo.TerminationCondition.infeasible
                ):
                    print("  Problem is infeasible - checking constraints...")
                    # You could add constraint checking here if needed
                elif (
                    results.solver.termination_condition
                    == pyo.TerminationCondition.unbounded
                ):
                    print("  Problem is unbounded")
                else:
                    print(
                        f"  Unexpected termination: {results.solver.termination_condition}"
                    )

                # Handle the case where the solver fails
                break

            # Update battery power schedule.
            self.battery.power_schedules[t, :] = (
                -self.battery.power_charging_committed
                + self.battery.power_discharging_committed
            )

            # Update battery SOC schedule.
            self.battery.soc_schedules[t, :] = self.battery.soc_schedule

        # After the loop, store the final state and schedules
        self.final_soc_schedule = self.battery.soc_schedules[0, :]
        self.final_power_schedule = self.battery.power_schedules
        self.final_cycles_schedule = self.battery.N_cycles_till_now

        # Return results
        RI_results = {
            "results": self.profit_results,
            "power_schedule": self.battery.power_schedules,
            "power_actions": self.battery.power_actions,
            "cycles_used": total_cycles_used,
            "final_soc_schedule": self.battery.soc_schedules[-1, :],
            "traded_prices": self.traded_prices,
            "soc_schedule": self.battery.soc_schedules,
        }
        return RI_results

    def _prepare_dis_charging_price_path(self, price_path: np.array):
        """
        Prepare the price path for the discharging and charging.
        """
        # replace nan by +inf
        remaining_charging_price_path = np.where(
            np.isnan(price_path), big_M, price_path
        )
        remaining_discharging_price_path = np.where(
            np.isnan(price_path), -big_M, price_path
        )
        return remaining_charging_price_path, remaining_discharging_price_path

    def _get_last_available_price_for_delivery_period(self, delivery_period):
        """
        Get the last available price for a specific delivery period.
        If no price is available for this delivery period, return None.
        """
        return self.historical_prices_discharging.get(delivery_period, None)

    def define_pyomo_model(
        self,
        price_path: np.array,
        optimization_window_timesteps: int,
        t: int,
        continuation_value_data=None,
        baseline_cv_schedule=None,
    ):
        """
        Define the pyomo model.
        Replaces np.maximum in SOC update with linear constraints.
        """
        model = pyo.ConcreteModel()

        # --- SETS ---
        # Define the time steps for the optimization window (e.g., 0, 1, ..., N-1 hours)
        model.T = pyo.RangeSet(0, optimization_window_timesteps - 1)
        model.T_plus_one = pyo.RangeSet(
            0, optimization_window_timesteps
        )  # For SOC update rule

        # --- PARAMETERS ---

        charging_price_path, discharging_price_path = (
            self._prepare_dis_charging_price_path(price_path)
        )

        # # Initialize historical price dictionaries if they don't exist
        # if not hasattr(self, "historical_prices_charging"):
        #     self.historical_prices_charging = {}
        # if not hasattr(self, "historical_prices_discharging"):
        #     self.historical_prices_discharging = {}

        # # Update historical prices with current available prices (skip None values)
        # for local_t in range(len(charging_price_path)):
        #     delivery_period = t + local_t

        #     # Only update if price is not None/NaN (check for big_M values too)
        #     price = price_path[local_t]

        #     if not np.isnan(price) and price is not None:
        #         self.historical_prices_charging[delivery_period] = price
        #         self.historical_prices_discharging[delivery_period] = price

        model.charging_price = pyo.Param(
            model.T, within=pyo.Reals, initialize=charging_price_path
        )  # Price forecast for the window
        model.discharging_price = pyo.Param(
            model.T, within=pyo.Reals, initialize=discharging_price_path
        )  # Price forecast for the window
        model.initial_soc_schedule = pyo.Param(
            model.T_plus_one,
            within=pyo.NonNegativeReals,
            initialize=self.battery.soc_schedule[
                t : t + optimization_window_timesteps + 1
            ],
        )  # Initial SoC for this window
        model.time_step_duration = pyo.Param(
            within=pyo.NonNegativeReals, default=self.battery.time_step
        )  # e.g., 1 for 1 hour
        model.max_power_charge = pyo.Param(
            model.T,
            within=pyo.NonNegativeReals,
            initialize=np.maximum(
                self.battery.capacity_normalized * np.ones(len(model.T))
                - self.battery.power_committed[
                    t : t + optimization_window_timesteps
                ],
                0,
            ),
        )
        model.max_power_discharge = pyo.Param(
            model.T,
            within=pyo.NonNegativeReals,
            initialize=np.maximum(
                self.battery.capacity_normalized * np.ones(len(model.T))
                + self.battery.power_committed[
                    t : t + optimization_window_timesteps
                ],
                0,
            ),
        )

        model.max_soc = pyo.Param(within=pyo.NonNegativeReals, default=1.0)
        model.min_soc = pyo.Param(within=pyo.NonNegativeReals, default=0.0)
        model.charge_efficiency = pyo.Param(
            within=pyo.NonNegativeReals, default=self.battery.charge_efficiency
        )
        model.discharge_efficiency = pyo.Param(
            within=pyo.NonNegativeReals,
            default=self.battery.discharge_efficiency,
        )

        # Schedule for this window (ensure this slice is correct)
        model.power_charging_committed = pyo.Param(
            model.T,
            within=pyo.Reals,
            initialize=self.battery.power_charging_committed[
                t : t + optimization_window_timesteps
            ],
        )
        model.power_discharging_committed = pyo.Param(
            model.T,
            within=pyo.Reals,
            initialize=self.battery.power_discharging_committed[
                t : t + optimization_window_timesteps
            ],
        )
        # --- VARIABLES ---
        model.soc = pyo.Var(
            model.T_plus_one,
            domain=pyo.NonNegativeReals,
            bounds=(model.min_soc, model.max_soc),
        )
        model.power_charge = pyo.Var(
            model.T,
            domain=pyo.NonNegativeReals,
        )  # Raw charging power from external source
        model.power_discharge = pyo.Var(
            model.T, domain=pyo.NonNegativeReals
        )  # Raw discharging power to external sink
        model.is_charging = pyo.Var(model.T, domain=pyo.Binary)

        # Auxiliary variables for the effective power flow into/out of the battery terminals
        # These replace the np.maximum terms in the SOC update
        model.power_flow_into_battery = pyo.Var(
            model.T, domain=pyo.NonNegativeReals
        )
        model.power_flow_out_of_battery = pyo.Var(
            model.T, domain=pyo.NonNegativeReals
        )

        # Set bounds for raw power_charge and power_discharge variables
        for t_idx in model.T:
            model.power_charge[t_idx].bounds = (
                0,
                model.max_power_charge[t_idx],
            )
            model.power_discharge[t_idx].bounds = (
                0,
                model.max_power_discharge[t_idx],
            )
            model.power_flow_into_battery[t_idx].bounds = (
                0,
                self.battery.capacity_normalized,
            )
            model.power_flow_out_of_battery[t_idx].bounds = (
                0,
                self.battery.capacity_normalized,
            )

        # --- CONSTRAINTS ---
        # Initial SOC constraint
        def initial_soc_rule(model):
            return model.soc[0] == model.initial_soc_schedule[0]

        model.initial_soc_constraint = pyo.Constraint(rule=initial_soc_rule)

        # Constraints to link raw power, schedule, binary, and effective power flows
        # These linearize the logic of the original np.maximum terms

        # M is a number, greater than any possible power flow value.
        # The power flow at each stage is limited by 2 * self.battery.capacity_normalized.
        M = 2 * self.battery.capacity_normalized

        # If is_charging is 1, power_flow_into_battery should be power_charge + schedule,
        # and power_flow_out_of_battery should be 0.
        # If is_charging is 0, power_flow_into_battery should be 0,
        # and power_flow_out_of_battery should be power_discharge - schedule.

        # Link power_flow_into_battery and power_flow_out_of_battery to the net power (charge - discharge + schedule)
        # This ensures that the net effect on SOC is accounted for by the effective flows.
        def net_power_flow_link_rule(model, t):
            # The net power at the interaction point is power_charge - power_discharge + schedule
            # This net power must equal the net effective flow into/out of the battery (into - out)
            return (
                model.power_discharge[t]
                + model.power_discharging_committed[t]
                - model.power_charge[t]
                - model.power_charging_committed[t]
            ) == model.power_flow_out_of_battery[
                t
            ] - model.power_flow_into_battery[
                t
            ]

        model.net_power_flow_link_constraint = pyo.Constraint(
            model.T, rule=net_power_flow_link_rule
        )

        # Use the binary variable to enforce that only one of power_flow_into_battery
        # or power_flow_out_of_battery can be non-zero at any time step.
        # This is a standard big-M formulation for selecting between two non-negative variables.

        # If is_charging is 1, power_flow_out_of_battery must be 0.
        def effective_flow_binary_link_out(model, t):
            return model.power_flow_out_of_battery[t] <= M * (
                1 - model.is_charging[t]
            )

        model.effective_flow_binary_link_out_constraint = pyo.Constraint(
            model.T, rule=effective_flow_binary_link_out
        )

        # If is_charging is 0, power_flow_into_battery must be 0.
        def effective_flow_binary_link_into(model, t):
            return model.power_flow_into_battery[t] <= M * model.is_charging[t]

        model.effective_flow_binary_link_into_constraint = pyo.Constraint(
            model.T, rule=effective_flow_binary_link_into
        )

        # Directly link power_charge/discharge to binary for stricter mutual exclusion
        def link_charge_binary_rule(model, t):
            return (
                model.power_charge[t]
                <= model.is_charging[t] * model.max_power_charge[t]
            )

        model.link_charge_binary_constraint = pyo.Constraint(
            model.T, rule=link_charge_binary_rule
        )

        def link_discharge_binary_rule(model, t):
            return (
                model.power_discharge[t]
                <= (1 - model.is_charging[t]) * model.max_power_discharge[t]
            )

        model.link_discharge_binary_constraint = pyo.Constraint(
            model.T, rule=link_discharge_binary_rule
        )

        # SOC update constraint using the new effective power flow variables
        # This replaces the original rule with np.maximum
        def soc_update_rule(model, t):
            return (
                model.soc[t + 1]
                == model.soc[t]
                + (
                    model.power_flow_into_battery[t]
                    - model.power_flow_out_of_battery[t]
                )
                * model.time_step_duration
            )

        model.soc_update_constraint = pyo.Constraint(
            model.T, rule=soc_update_rule
        )

        # SIMPLIFIED CYCLE TRACKING (much faster than incremental tracking)
        # Instead of tracking cycles at every timestep, just limit total cycles used in window
        model.N_cycles = pyo.Var(
            domain=pyo.NonNegativeReals,
            bounds=(0, self.battery.N_daily_cycles_max),
        )

        # Total cycles constraint: sum of all throughput in the optimization window
        def total_cycles_constraint_rule(model):
            window_cycle_throughput = pyo.quicksum(
                (
                    model.power_flow_into_battery[t]
                    + model.power_flow_out_of_battery[t]
                )
                * model.time_step_duration
                / 2.0
                for t in model.T
            )
            return (
                model.N_cycles
                == self.battery.N_cycles_till_now + window_cycle_throughput
            )

        model.total_cycles_constraint = pyo.Constraint(
            rule=total_cycles_constraint_rule
        )

        # --- CV INTERPOLATION SETUP ---
        if continuation_value_data and baseline_cv_schedule is not None:
            cv_data_length = len(continuation_value_data)

            # CV variables over T for interpolation
            model.cv = pyo.Var(model.T)

            # Initialize baseline_cv for all indices, handling None values
            baseline_cv_full = {}
            for i in model.T:
                if (
                    i < len(baseline_cv_schedule)
                    and baseline_cv_schedule[i] is not None
                ):
                    baseline_cv_full[i] = baseline_cv_schedule[i]
                else:
                    baseline_cv_full[i] = 0.0  # Use 0 for None values

            model.baseline_cv = pyo.Param(
                model.T,
                initialize=baseline_cv_full,
                default=0.0,  # Default value for any missing indices
            )

            # Add interpolation constraints for CV
            for i in range(cv_data_length):
                if (
                    i in model.T_plus_one
                    and continuation_value_data[i] is not None
                ):
                    soc_levels, cycle_levels, cv_grid = (
                        continuation_value_data[i]
                    )

                    self._add_direct_interpolation_constraints(
                        model,
                        i,
                        soc_levels,
                        cycle_levels,
                        cv_grid,
                        current_trading_timestep=t,
                    )

        # --- OBJECTIVE ---
        # Add a separate profit variable for easy extraction
        model.total_profit = pyo.Var(domain=pyo.Reals)

        profit = (
            (
                pyo.sum_product(
                    model.power_discharge,
                    model.discharging_price,
                    index=model.T,
                )
                * model.discharge_efficiency
                - pyo.sum_product(
                    model.power_charge, model.charging_price, index=model.T
                )
                / model.charge_efficiency
            )
            * model.time_step_duration
            * self.battery.size_energy
        )

        model.profit_constraint = pyo.Constraint(
            expr=model.total_profit == profit
        )

        if continuation_value_data and baseline_cv_schedule is not None:
            # Objective with continuation values

            # --- Unwind Cost Calculation ---
            # Per user request, calculate unwind cost for each timestep `t_local`
            # using the price at that timestep.
            # Price priority:
            # 1. Current price if available (not NaN).
            # 2. Historical price for that delivery period.
            # 3. Fallback to 0.0 if no price has ever been seen.
            unwind_prices = {}
            for t_local in model.T:
                absolute_t = t + t_local
                current_price = price_path[t_local]

                if not np.isnan(current_price):
                    unwind_prices[t_local] = current_price
                else:
                    # Fallback to historical price, with a final default to 0
                    unwind_prices[t_local] = (
                        self.historical_prices_charging.get(absolute_t, 0.0)
                    )

            # Calculate total unwind cost using a single expression with per-timestep prices
            total_unwind_cost = (
                pyo.quicksum(
                    (
                        model.power_flow_into_battery[t_local]
                        - model.power_flow_out_of_battery[t_local]
                    )
                    * unwind_prices[t_local]
                    for t_local in model.T
                )
                * model.time_step_duration
                * self.battery.size_energy
            )

            # Scale alpha by window length to avoid over-counting CV changes
            if optimization_window_timesteps > 0:
                scaled_alpha = (
                    self.alpha
                    # number of cycles left to complete the day
                    * (
                        self.battery.N_daily_cycles_max
                        - self.battery.N_cycles_till_now
                    )
                    # number of timesteps required for one cycle
                    * (1 / model.time_step_duration)
                    / optimization_window_timesteps
                )
            else:
                scaled_alpha = self.alpha

            # Calculate total profit using quicksum
            total_profit_expr = pyo.quicksum(
                (
                    model.power_discharge[t_local]
                    * model.discharging_price[t_local]
                    * model.discharge_efficiency
                    - model.power_charge[t_local]
                    * model.charging_price[t_local]
                    / model.charge_efficiency
                )
                * model.time_step_duration
                * self.battery.size_energy
                for t_local in model.T
            )

            # Calculate total CV change using quicksum
            cv_data_length = len(continuation_value_data)
            total_cv_change = pyo.quicksum(
                model.cv[t_local] - model.baseline_cv[t_local]
                for t_local in model.T
                if (
                    t_local < cv_data_length
                    and t_local < len(continuation_value_data)
                    and continuation_value_data[t_local] is not None
                    and t_local < len(baseline_cv_schedule)
                    and baseline_cv_schedule[t_local] is not None
                )
            )

            # Combine all parts of the objective function
            objective_expr = total_profit_expr + scaled_alpha * (
                total_cv_change - total_unwind_cost
            )

            model.objective = pyo.Objective(
                expr=objective_expr, sense=pyo.maximize
            )
        else:
            # Simple profit maximization without CV
            model.objective = pyo.Objective(expr=profit, sense=pyo.maximize)

        return model

    def _add_direct_interpolation_constraints(
        self,
        model,
        t_idx,
        soc_levels,
        cycle_levels,
        cv_grid,
        current_trading_timestep=0,
    ):
        """
        Direct interpolation for continuation values.
        Pre-computes a simple linear function CV = a * SOC + b and uses it directly.
        """
        n_soc = len(soc_levels)
        n_cycle = len(cycle_levels)

        # Step 1: Get the cycle level from the cycle schedule
        absolute_timestep = current_trading_timestep + t_idx

        if (
            current_trading_timestep > 0
            and hasattr(self.battery, "cycle_schedules")
            and self.battery.cycle_schedules is not None
            and absolute_timestep < self.battery.cycle_schedules.shape[1]
        ):
            # Use cycle level from previous iteration
            previous_iteration = current_trading_timestep - 1
            current_cycle_val = self.battery.cycle_schedules[
                previous_iteration, absolute_timestep
            ]
        else:
            # Fallback: estimate based on current state
            current_cycle_val = getattr(self.battery, "N_cycles_till_now", 0.0)
            if t_idx > 0:
                max_daily_cycles = getattr(
                    self.battery, "N_daily_cycles_max", 1.0
                )
                timesteps_per_day = getattr(self, "N_timesteps", 96)
                cycles_per_timestep = max_daily_cycles / timesteps_per_day
                additional_cycles = t_idx * cycles_per_timestep * 0.6
                current_cycle_val += additional_cycles

        # Step 2: Find closest cycle level
        cycle_idx = 0
        if current_cycle_val <= cycle_levels[0]:
            cycle_idx = 0
        elif current_cycle_val >= cycle_levels[-1]:
            cycle_idx = n_cycle - 1
        else:
            for j in range(n_cycle - 1):
                if cycle_levels[j] <= current_cycle_val <= cycle_levels[j + 1]:
                    if abs(current_cycle_val - cycle_levels[j]) <= abs(
                        current_cycle_val - cycle_levels[j + 1]
                    ):
                        cycle_idx = j
                    else:
                        cycle_idx = j + 1
                    break

        # Step 3: Extract CV values for the selected cycle level
        cv_values_at_cycle = cv_grid[:, cycle_idx]

        # Step 4: Fit linear function CV = a * SOC + b
        if len(soc_levels) >= 2:
            A = np.vstack([soc_levels, np.ones(len(soc_levels))]).T
            a, b = np.linalg.lstsq(A, cv_values_at_cycle, rcond=None)[0]

            # Create constraint: CV = a * SOC + b
            def cv_direct_rule(model):
                return model.cv[t_idx] == a * model.soc[t_idx + 1] + b

            setattr(
                model,
                f"cv_direct_{t_idx}",
                pyo.Constraint(rule=cv_direct_rule),
            )
        else:
            # Constant case
            setattr(
                model,
                f"cv_constant_{t_idx}",
                pyo.Constraint(expr=model.cv[t_idx] == cv_values_at_cycle[0]),
            )

    def _add_2d_direct_interpolation_constraints(
        self,
        model,
        t_idx,
        soc_levels,
        cycle_levels,
        cv_grid,
        current_trading_timestep=0,
    ):
        """
        2D direct interpolation for continuation values.
        Pre-computes a bilinear function CV = a * SOC + b * cycles + c using all data points.
        """
        n_soc = len(soc_levels)
        n_cycle = len(cycle_levels)

        # Step 1: Get the current cycle level from the cycle schedule
        absolute_timestep = current_trading_timestep + t_idx

        if (
            current_trading_timestep > 0
            and hasattr(self.battery, "cycle_schedules")
            and self.battery.cycle_schedules is not None
            and absolute_timestep < self.battery.cycle_schedules.shape[1]
        ):
            # Use cycle level from previous iteration
            previous_iteration = current_trading_timestep - 1
            current_cycle_val = self.battery.cycle_schedules[
                previous_iteration, absolute_timestep
            ]
        else:
            # Fallback: estimate based on current state
            current_cycle_val = getattr(self.battery, "N_cycles_till_now", 0.0)
            if t_idx > 0:
                max_daily_cycles = getattr(
                    self.battery, "N_daily_cycles_max", 1.0
                )
                timesteps_per_day = getattr(self, "N_timesteps", 96)
                cycles_per_timestep = max_daily_cycles / timesteps_per_day
                additional_cycles = t_idx * cycles_per_timestep * 0.6
                current_cycle_val += additional_cycles

        # Step 2: Prepare data for 2D regression
        # Create all combinations of SOC and cycle levels
        soc_points = []
        cycle_points = []
        cv_values = []

        for i, soc in enumerate(soc_levels):
            for j, cycle in enumerate(cycle_levels):
                soc_points.append(soc)
                cycle_points.append(cycle)
                cv_values.append(cv_grid[i, j])

        # Step 3: Fit bilinear function CV = a * SOC + b * cycles + c
        if len(soc_points) >= 3:
            # Create design matrix [SOC, cycles, 1]
            A = np.column_stack(
                [
                    np.array(soc_points),
                    np.array(cycle_points),
                    np.ones(len(soc_points)),
                ]
            )

            # Solve least squares regression
            try:
                coeffs = np.linalg.lstsq(A, np.array(cv_values), rcond=None)[0]
                a, b, c = coeffs

                # Create constraint: CV = a * SOC + b * cycles + c
                def cv_2d_direct_rule(model):
                    return model.cv[t_idx] == (
                        a * model.soc[t_idx + 1] + b * current_cycle_val + c
                    )

                setattr(
                    model,
                    f"cv_2d_direct_{t_idx}",
                    pyo.Constraint(rule=cv_2d_direct_rule),
                )

            except np.linalg.LinAlgError:
                # Fallback to 1D interpolation if 2D fails
                self._add_direct_interpolation_constraints(
                    model,
                    t_idx,
                    soc_levels,
                    cycle_levels,
                    cv_grid,
                    current_trading_timestep,
                )
        else:
            # Fallback to 1D interpolation if insufficient data
            self._add_direct_interpolation_constraints(
                model,
                t_idx,
                soc_levels,
                cycle_levels,
                cv_grid,
                current_trading_timestep,
            )
