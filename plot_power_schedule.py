import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
import pickle
import os
import sys

# Adjust path to include the root directory of the project
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from trading_env import BatteryTradingEnv
from stable_baselines3 import PPO

# Set style for better looking plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def simulate_agent_day(
    model, config, daily_data, day_name="Sample Day", ri_strategy=None
):
    """
    Simulate a full day with the RL agent and track all decisions.
    """
    env = BatteryTradingEnv(config, daily_data, ri_strategy=ri_strategy)
    obs, _ = env.reset()

    # Track data throughout the day
    time_log = []
    action_log = []
    soc_log = []
    cycles_log = []
    price_log = []
    commitment_log = []
    reward_log = []
    dispatch_log = []

    step = 0
    done = False

    while not done:
        # Get agent's action
        action, _ = model.predict(obs, deterministic=True)

        # Store current state before action
        current_time = env.trading_times[min(step, len(env.trading_times) - 1)]
        time_log.append(current_time)
        action_log.append(action.copy())
        soc_log.append(env.battery.current_soc)
        cycles_log.append(env.battery.N_cycles_till_now)
        commitment_log.append(env.commitments.copy())

        # Get current market prices for context
        current_trades = daily_data.loc[daily_data["traded"] == current_time]
        prices = np.full(env.n_delivery_periods, np.nan)
        if not current_trades.empty:
            price_map = pd.Series(
                current_trades["VWAP"].values,
                index=current_trades["delivery_start"],
            )
            for dt, price in price_map.items():
                if dt in env.delivery_period_map:
                    idx = env.delivery_period_map[dt]
                    prices[idx] = price
        price_log.append(prices.copy())

        # Take the step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        reward_log.append(reward)

        # Check if this timestep corresponds to a dispatch
        dispatch_power = 0
        if current_time in env.delivery_period_map:
            delivery_idx = env.delivery_period_map[current_time]
            dispatch_power = env.commitments[delivery_idx]
        dispatch_log.append(dispatch_power)

        step += 1

    # Create comprehensive results dataframe
    results = pd.DataFrame(
        {
            "time": time_log,
            "soc": soc_log,
            "cycles": cycles_log,
            "reward": reward_log,
            "dispatch_power": dispatch_log,
        }
    )

    # Add action and commitment details
    for i in range(env.n_delivery_periods):
        results[f"action_{i}"] = [
            act[i] if i < len(act) else 0 for act in action_log
        ]
        results[f"commitment_{i}"] = [
            comm[i] if i < len(comm) else 0 for comm in commitment_log
        ]
        results[f"price_{i}"] = [
            price[i] if i < len(price) else np.nan for price in price_log
        ]

    return results, env


def plot_power_schedule(results, env, day_name="Sample Day", save_path=None):
    """
    Create comprehensive plots of the agent's power schedule.
    """
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    fig.suptitle(
        f"RL Agent Power Schedule Analysis - {day_name}",
        fontsize=16,
        fontweight="bold",
    )

    # Prepare time axis
    time_hours = [(t.hour + t.minute / 60) for t in results["time"]]

    # Plot 1: SOC and Cycle Evolution
    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    line1 = ax1.plot(
        time_hours, results["soc"], "b-", linewidth=2, label="SOC"
    )
    ax1.set_ylabel("State of Charge", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    line2 = ax1_twin.plot(
        time_hours, results["cycles"], "r-", linewidth=2, label="Cycles Used"
    )
    ax1_twin.axhline(
        y=2.0, color="red", linestyle="--", alpha=0.7, label="Cycle Limit"
    )
    ax1_twin.set_ylabel("Cycles Used", color="r")
    ax1_twin.tick_params(axis="y", labelcolor="r")

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines] + ["Cycle Limit"]
    ax1.legend(lines + [ax1_twin.lines[-1]], labels, loc="upper left")
    ax1.set_title("Battery State Evolution")

    # Plot 2: Power Dispatch Schedule
    ax2 = axes[1]
    dispatch_mw = (
        results["dispatch_power"] * env.battery.size_energy
    )  # Convert to MW
    colors = [
        "green" if p > 0 else "red" if p < 0 else "gray" for p in dispatch_mw
    ]

    bars = ax2.bar(time_hours, dispatch_mw, color=colors, alpha=0.7, width=0.2)
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    ax2.set_ylabel("Power (MW)")
    ax2.set_title(
        "Actual Power Dispatch Schedule (Green=Charge, Red=Discharge)"
    )
    ax2.grid(True, alpha=0.3)

    # Add capacity limits
    max_power = env.battery.size_energy  # MW
    ax2.axhline(
        y=max_power,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=f"Max Power ({max_power} MW)",
    )
    ax2.axhline(y=-max_power, color="orange", linestyle="--", alpha=0.7)
    ax2.legend()

    # Plot 3: Market Prices and Trading Opportunities
    ax3 = axes[2]

    # Get the market prices for the trading periods
    trading_prices = []
    trading_times_hours = []

    for i, t in enumerate(results["time"]):
        # Look for the price at the current trading time for the current delivery period
        if t in env.delivery_period_map:
            delivery_idx = env.delivery_period_map[t]
            price = results[f"price_{delivery_idx}"].iloc[i]
            if not np.isnan(price):
                trading_prices.append(price)
                trading_times_hours.append(time_hours[i])

    if trading_prices:
        ax3.plot(
            trading_times_hours,
            trading_prices,
            "ko-",
            alpha=0.7,
            linewidth=1,
            markersize=4,
        )
        ax3.set_ylabel("Price (€/MWh)")
        ax3.set_title("Market Prices During Trading")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(
            0.5,
            0.5,
            "No price data available",
            transform=ax3.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
        ax3.set_title("Market Prices During Trading")

    # Plot 4: Reward Evolution
    ax4 = axes[3]
    cumulative_reward = np.cumsum(results["reward"])

    ax4.plot(
        time_hours,
        results["reward"],
        "g-",
        alpha=0.6,
        linewidth=1,
        label="Step Reward",
    )
    ax4_twin = ax4.twinx()
    ax4_twin.plot(
        time_hours,
        cumulative_reward,
        "b-",
        linewidth=2,
        label="Cumulative Reward",
    )

    ax4.set_ylabel("Step Reward", color="g")
    ax4.tick_params(axis="y", labelcolor="g")
    ax4_twin.set_ylabel("Cumulative Reward", color="b")
    ax4_twin.tick_params(axis="y", labelcolor="b")
    ax4.set_xlabel("Hour of Day")
    ax4.set_title("Reward Evolution")
    ax4.grid(True, alpha=0.3)

    # Final formatting
    for ax in axes:
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 4))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Power schedule plot saved to: {save_path}")

    plt.show()
    return fig


def plot_commitment_vs_dispatch(
    results, env, day_name="Sample Day", save_path=None
):
    """
    Plot the difference between trading commitments and actual dispatch.
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    fig.suptitle(
        f"Trading Commitments vs Dispatch - {day_name}",
        fontsize=16,
        fontweight="bold",
    )

    time_hours = [(t.hour + t.minute / 60) for t in results["time"]]

    # Plot 1: All commitments over time
    ax1 = axes[0]

    for i in range(
        min(24, env.n_delivery_periods)
    ):  # Limit to 24 periods for readability
        commitments = results[f"commitment_{i}"] * env.battery.size_energy
        if (
            commitments.abs().max() > 0.01
        ):  # Only plot if there are significant commitments
            ax1.plot(
                time_hours,
                commitments,
                alpha=0.6,
                linewidth=1,
                label=f"Period {i}",
            )

    ax1.set_ylabel("Commitment (MW)")
    ax1.set_title("Trading Commitments by Delivery Period")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="black", linestyle="-", alpha=0.5)

    # Plot 2: Net position evolution
    ax2 = axes[1]

    # Calculate net position (sum of all commitments at each time)
    net_position = np.zeros(len(results))
    for i in range(env.n_delivery_periods):
        net_position += results[f"commitment_{i}"]

    net_position_mw = net_position * env.battery.size_energy
    dispatch_mw = results["dispatch_power"] * env.battery.size_energy

    ax2.plot(
        time_hours,
        net_position_mw,
        "b-",
        linewidth=2,
        label="Net Commitment",
        alpha=0.8,
    )
    ax2.bar(
        time_hours,
        dispatch_mw,
        alpha=0.5,
        width=0.2,
        color=[
            "green" if p > 0 else "red" if p < 0 else "gray"
            for p in dispatch_mw
        ],
        label="Actual Dispatch",
    )

    ax2.set_ylabel("Power (MW)")
    ax2.set_xlabel("Hour of Day")
    ax2.set_title("Net Trading Position vs Actual Dispatch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.5)

    for ax in axes:
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 4))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Commitment vs dispatch plot saved to: {save_path}")

    plt.show()
    return fig


def analyze_agent_schedule(
    model_path="results/rl_agent_results/rl_trading_model_daily.zip",
    config_path="config.yml",
    data_path="data/df_test_data_spring.pkl",
    output_dir="results/rl_agent_results",
    num_days=3,
):
    """
    Analyze the agent's power schedule for multiple days.
    """
    print("📊 ANALYZING RL AGENT POWER SCHEDULE")
    print("=" * 50)

    # Load model and config
    model = PPO.load(model_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load data
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    # Analyze multiple days
    unique_days = data["delivery_start"].dt.date.unique()

    for i, day in enumerate(unique_days[:num_days]):
        print(f"\nAnalyzing day {i+1}: {day}")

        # Get daily data
        daily_data = data.loc[
            (data["delivery_start"].dt.date == day)
            & (data["traded"].dt.date == day)
        ]

        if daily_data.empty:
            print(f"No data available for {day}")
            continue

        # Simulate the day
        results, env = simulate_agent_day(
            model, config, daily_data, f"Day {day}"
        )

        # Generate plots
        schedule_path = os.path.join(
            output_dir, f"power_schedule_day_{i+1}_{day}.png"
        )
        commitment_path = os.path.join(
            output_dir, f"commitments_day_{i+1}_{day}.png"
        )

        plot_power_schedule(results, env, f"Day {day}", schedule_path)
        plot_commitment_vs_dispatch(
            results, env, f"Day {day}", commitment_path
        )

        # Print summary
        final_soc = results["soc"].iloc[-1]
        final_cycles = results["cycles"].iloc[-1]
        total_reward = results["reward"].sum()
        max_dispatch = (
            results["dispatch_power"].abs().max() * env.battery.size_energy
        )

        print(f"  Final SOC: {final_soc:.3f}")
        print(
            f"  Cycles used: {final_cycles:.3f} / {config.get('N_daily_cycles_max', 2.0)}"
        )
        print(f"  Total reward: {total_reward:.0f}")
        print(f"  Max dispatch: {max_dispatch:.2f} MW")

        if final_cycles > config.get("N_daily_cycles_max", 2.0):
            excess = final_cycles - config.get("N_daily_cycles_max", 2.0)
            print(f"  ⚠️ Cycle violation: {excess:.3f} excess cycles")
        else:
            print(f"  ✅ Within cycle limits")


if __name__ == "__main__":
    analyze_agent_schedule()
