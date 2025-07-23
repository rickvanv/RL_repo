import numpy as np


class Battery:
    def __init__(self, config):
        """
        Initialize the Battery class.

        Args:
            config: Dictionary of configuration parameters.
        """
        self.config = config
        self.capacity = config["capacity"]
        self.size_hours = config["size_hours"]
        self.size_energy = self.size_hours * self.capacity
        self.capacity_normalized = 1 / self.size_hours
        self.charge_efficiency = config["charge_efficiency"]
        self.discharge_efficiency = config["discharge_efficiency"]
        self.energy_value = config["energy_value"]
        self.time_step = config["time_step"]
        self.history = []
        self.N_cycles_till_now = 0
        self.N_daily_cycles_max = config["N_daily_cycles_max"]
        # Initialize the SOC and power schedule
        self.initial_soc = config["initial_soc"]
        self.current_soc = self.initial_soc
        self.soc_schedule = config["initial_soc"] * np.ones(
            config["N_timesteps"] + 1
        )

    def update_current_soc(self, new_soc):
        """
        Update the current SOC of the battery.
        """
        self.current_soc = new_soc

    def update_soc_schedule(self, t, new_soc_schedule):
        """
        Update the state of charge (SOC) of the battery.

        Args:
            new_soc_schedule: New SOC schedule.
            price_path: Price path.
        """
        self.soc_schedule[t : t + len(new_soc_schedule)] = np.maximum(
            new_soc_schedule, 0
        )

    def update_power_committed(
        self,
        t,
        power_flow_into_battery,
        power_flow_out_of_battery,
    ):
        """
        Update the battery power schedule.

        Args:
            power_flow_into_battery: Power charging schedule.
            power_flow_out_of_battery: Power discharging schedule.
            price_path: Price path.
        """
        self.power_charging_committed[t : t + len(power_flow_into_battery)] = (
            power_flow_into_battery
        )
        self.power_discharging_committed[
            t : t + len(power_flow_out_of_battery)
        ] = power_flow_out_of_battery
