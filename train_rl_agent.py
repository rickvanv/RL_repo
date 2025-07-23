import os
import sys
import pandas as pd
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Adjust path to include the root directory of the project
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from trading_env import BatteryTradingEnv
from battery import Battery
from rolling_intrinsic import RollingIntrinsicStrategy

# Set style for better looking plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def train_agent_with_tracking(
    config, full_train_data, ri_strategy, eval_frequency=50
):
    """
    Train a PPO agent with periodic evaluation and progress tracking.
    """
    print("--- Starting Agent Training with Progress Tracking ---")

    # Group data by day to create daily episodes
    unique_days = full_train_data["delivery_start"].dt.date.unique()

    # Create a dummy environment to initialize the agent.
    # We initialize with a sample day of data.
    sample_day_data = full_train_data.loc[
        (full_train_data["delivery_start"].dt.date == unique_days[1])
        & (full_train_data["traded"].dt.date == unique_days[1])
    ]
    env = DummyVecEnv(
        [
            lambda: BatteryTradingEnv(
                config, sample_day_data, ri_strategy=ri_strategy
            )
        ]
    )

    # Decouple the rollout buffer size from the episode length to avoid edge cases.
    # n_steps determines how much data is collected before a model update.
    # Making it smaller than the episode length ensures updates happen mid-episode.
    n_steps_for_update = config.get("n_steps_for_update", 96)
    # The batch_size is for the gradient update step and should be smaller than n_steps.
    batch_size = config.get("batch_size", 96)

    # Get the initial learning rate from the config to be used in the schedule.
    initial_lr = config.get("learning_rate", 0.0001)

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=0,
        tensorboard_log="./rl_tensorboard_logs/",
        learning_rate=initial_lr,
        n_steps=n_steps_for_update,
        batch_size=batch_size,
        gamma=config.get("gamma", 1.0),
    )

    # Progress tracking
    progress_data = []
    n_epochs = config.get("n_training_epochs", 100)

    # # Pre-calculate all RI schedules before the training loop
    ri_schedules_by_day = {}
    if ri_strategy:
        print("--- Pre-calculating all RI schedules for training data ---")
        for day in unique_days:
            daily_data = full_train_data.loc[
                (full_train_data["delivery_start"].dt.date == day)
                & (full_train_data["traded"].dt.date == day)
            ]
            if not daily_data.empty:
                price_paths = []
                test_datetime_range = sorted(daily_data["traded"].unique())
                for t, datetime in enumerate(test_datetime_range):
                    df_traded = daily_data.loc[
                        (daily_data["traded"] == datetime)
                    ].copy()
                    prices = df_traded["VWAP"].values
                    price_paths.append(prices)

                ri_results = ri_strategy.run(price_paths)
                ri_schedules_by_day[day] = ri_results["power_actions"]
        print("--- RI schedules pre-calculation complete ---")

    # Training loop with periodic evaluation
    for epoch in range(n_epochs):
        print(f"--- Epoch {epoch + 1}/{n_epochs} ---")

        # Train for this epoch
        for day in unique_days[1:]:
            daily_data = full_train_data.loc[
                (full_train_data["delivery_start"].dt.date == day)
                & (full_train_data["traded"].dt.date == day)
            ]
            if daily_data.empty:
                continue

            # Load the pre-calculated RI schedule for the day
            ri_schedule_for_day = ri_schedules_by_day.get(day)

            # Set the environment with the new daily data and RI schedule
            env.envs[0].set_daily_data(
                daily_data, ri_schedule=ri_schedule_for_day
            )
            env.envs[0].reset()

            # Dynamically set the number of timesteps for this day's learning
            n_steps_this_day = len(env.envs[0].trading_times)
            if n_steps_this_day == 0:
                continue

            # The total timesteps for learn() is per-day here
            model.learn(
                total_timesteps=n_steps_this_day,
                reset_num_timesteps=True,  # Reset to ensure clean episode boundaries
                log_interval=1000,  # Log every 1000 steps
            )

        # Evaluate progress at specified intervals
        if (epoch + 1) % eval_frequency == 0 or epoch == n_epochs - 1:
            print(f"--- Evaluating after epoch {epoch + 1} ---")

            # Quick evaluation on a subset of test data for progress tracking
            eval_results = evaluate_agent(
                model, config, full_train_data, ri_strategy
            )

            # Calculate metrics
            avg_raw_profit = eval_results["raw_profit"].mean()
            avg_total_reward = eval_results["total_reward"].mean()
            avg_cycles = eval_results["cycles_used"].mean()
            profit_std = eval_results["raw_profit"].std()

            progress_data.append(
                {
                    "epoch": epoch + 1,
                    "avg_raw_profit": avg_raw_profit,
                    "avg_total_reward": avg_total_reward,
                    "avg_cycles_used": avg_cycles,
                    "profit_std": profit_std,
                }
            )

            print(
                f"Epoch {epoch + 1}: Avg Raw Profit = {avg_raw_profit:.2f}, "
                f"Avg Total Reward = {avg_total_reward:.2f}, "
                f"Profit Std = {profit_std:.2f}"
            )

        print(f"Epoch {epoch + 1} complete.")

    print("--- Agent Training Complete ---")
    return model, progress_data


def train_agent(config, full_train_data, ri_strategy):
    """
    Train a PPO agent by iterating through daily episodes.`
    This is the original training function without tracking.
    """
    print("--- Starting Agent Training ---")

    # Group data by day to create daily episodes
    unique_days = full_train_data["delivery_start"].dt.date.unique()

    # Create a dummy environment to initialize the agent.
    # We initialize with a sample day of data.
    sample_day_data = full_train_data.loc[
        (full_train_data["delivery_start"].dt.date == unique_days[1])
        & (full_train_data["traded"].dt.date == unique_days[1])
    ]
    env = DummyVecEnv(
        [
            lambda: BatteryTradingEnv(
                config, sample_day_data, ri_strategy=ri_strategy
            )
        ]
    )

    # Decouple the rollout buffer size from the episode length to avoid edge cases.
    # n_steps determines how much data is collected before a model update.
    # Making it smaller than the episode length ensures updates happen mid-episode.
    n_steps_for_update = config.get("n_steps_for_update", 96)
    # The batch_size is for the gradient update step and should be smaller than n_steps.
    batch_size = config.get("batch_size", 96)

    # Get the initial learning rate from the config to be used in the schedule.
    initial_lr = config.get("learning_rate", 0.0001)

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=0,
        tensorboard_log="./rl_tensorboard_logs/",
        learning_rate=initial_lr,
        n_steps=n_steps_for_update,
        batch_size=batch_size,
        gamma=config.get("gamma", 1.0),
    )

    # Training loop over multiple episodes (days)
    n_epochs = config.get("n_training_epochs", 100)
    for epoch in range(n_epochs):
        print(f"--- Epoch {epoch + 1}/{n_epochs} ---")
        for day in unique_days[1:]:
            daily_data = full_train_data.loc[
                (full_train_data["delivery_start"].dt.date == day)
                & (full_train_data["traded"].dt.date == day)
            ]
            if daily_data.empty:
                continue

            # Set the environment with the new daily data
            env.envs[0].set_daily_data(daily_data)
            env.envs[0].reset()

            # Dynamically set the number of timesteps for this day's learning
            n_steps_this_day = len(env.envs[0].trading_times)
            if n_steps_this_day == 0:
                continue

            # The total timesteps for learn() is per-day here
            model.learn(
                total_timesteps=n_steps_this_day,
                reset_num_timesteps=True,  # Reset to ensure clean episode boundaries
                log_interval=1000,  # Log every 1000 steps
            )
            # print(f"Day {day} complete.")
        print(f"Epoch {epoch + 1} complete.")

    print("--- Agent Training Complete ---")
    return model


def evaluate_agent(model, config, full_test_data, ri_strategy):
    """
    Evaluate the agent's performance over multiple days.
    """
    print("--- Starting Agent Evaluation ---")
    unique_days = full_test_data["delivery_start"].dt.date.unique()
    all_daily_results = []

    for day in unique_days[1:]:
        daily_data = full_test_data.loc[
            (full_test_data["delivery_start"].dt.date == day)
            & (full_test_data["traded"].dt.date == day)
        ]
        if daily_data.empty:
            continue

        # Create a fresh environment for each evaluation day
        eval_env = BatteryTradingEnv(
            config, daily_data, ri_strategy=ri_strategy
        )
        obs, _ = eval_env.reset()

        done = False
        daily_profit = 0
        daily_raw_profit = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            daily_profit += reward
            if done:  # Get final cumulative raw profit
                daily_raw_profit = info["cumulative_raw_profit"]

        final_soc = eval_env.battery.current_soc
        final_cycles = eval_env.battery.N_cycles_till_now

        all_daily_results.append(
            {
                "day": day,
                "total_reward": daily_profit,
                "raw_profit": daily_raw_profit,
                "final_soc": final_soc,
                "cycles_used": final_cycles,
            }
        )
        print(
            f"Day {day}: Total Reward={daily_profit:.2f}, Raw Trading Profit={daily_raw_profit:.2f}, Cycles={final_cycles:.4f}"
        )

    print("--- Agent Evaluation Complete ---")
    return pd.DataFrame(all_daily_results)


def plot_training_progress(progress_data, output_dir):
    """
    Plot training progress across epochs.
    """
    if not progress_data:
        print("No progress data available for plotting.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(progress_data)

    # Create the epoch progress plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Training Progress Across Epochs", fontsize=16, fontweight="bold"
    )

    # Plot 1: Raw Profit vs Epochs
    ax1 = axes[0, 0]
    ax1.plot(
        df["epoch"],
        df["avg_raw_profit"],
        "o-",
        color="green",
        linewidth=2,
        markersize=6,
    )
    ax1.set_xlabel("Training Epochs")
    ax1.set_ylabel("Average Raw Profit")
    ax1.set_title("Learning Curve: Raw Trading Profit")
    ax1.grid(True, alpha=0.3)

    # Add trend line if we have enough data points
    if len(df) > 2:
        z = np.polyfit(df["epoch"], df["avg_raw_profit"], 1)
        p = np.poly1d(z)
        ax1.plot(
            df["epoch"],
            p(df["epoch"]),
            "--",
            alpha=0.8,
            color="darkgreen",
            label=f"Trend (slope: {z[0]:.2f})",
        )
        ax1.legend()

    # Plot 2: Total Reward vs Epochs
    ax2 = axes[0, 1]
    ax2.plot(
        df["epoch"],
        df["avg_total_reward"],
        "o-",
        color="red",
        linewidth=2,
        markersize=6,
    )
    ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Training Epochs")
    ax2.set_ylabel("Average Total Reward")
    ax2.set_title("Learning Curve: Total Reward (with penalties)")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Consistency (Standard Deviation)
    ax3 = axes[1, 0]
    ax3.plot(
        df["epoch"],
        df["profit_std"],
        "o-",
        color="blue",
        linewidth=2,
        markersize=6,
    )
    ax3.set_xlabel("Training Epochs")
    ax3.set_ylabel("Profit Standard Deviation")
    ax3.set_title("Consistency: Lower is Better")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Cycle Usage
    ax4 = axes[1, 1]
    ax4.plot(
        df["epoch"],
        df["avg_cycles_used"],
        "o-",
        color="orange",
        linewidth=2,
        markersize=6,
    )
    ax4.axhline(
        y=2.0, color="red", linestyle="--", alpha=0.7, label="Max Cycles"
    )
    ax4.set_xlabel("Training Epochs")
    ax4.set_ylabel("Average Cycles Used")
    ax4.set_title("Cycle Usage Over Training")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = os.path.join(output_dir, "training_progress_epochs.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Training progress plot saved to: {output_path}")

    # Also save progress data
    progress_df_path = os.path.join(output_dir, "epoch_progress.csv")
    df.to_csv(progress_df_path, index=False)
    print(f"Progress data saved to: {progress_df_path}")

    plt.show()


if __name__ == "__main__":
    print("Loading configuration...")
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)
    print("Configuration loaded.")

    print("Loading test data...")
    with open("data/df_test_data_winter.pkl", "rb") as f:
        test_data = pickle.load(f)
    print("Test data loaded.")

    # Use the same data for training and testing for this example
    train_data = test_data.copy()

    # Check if user wants progress tracking
    enable_tracking = config.get("enable_progress_tracking", True)
    eval_frequency = config.get(
        "evaluation_frequency", 50
    )  # Evaluate every 50 epochs

    # Initialize the RI strategy
    use_ri_guidance = config.get("use_ri_guidance", False)
    ri_strategy = None
    if use_ri_guidance:
        print("Initializing Rolling Intrinsic strategy for guidance...")
        battery = Battery(config)  # RI needs a battery instance
        ri_strategy = RollingIntrinsicStrategy(battery, config)
        print("RI strategy initialized.")

    # --- Agent Training ---
    if enable_tracking:
        print(
            f"Training with progress tracking (evaluating every {eval_frequency} epochs)..."
        )
        rl_agent, progress_data = train_agent_with_tracking(
            config=config,
            full_train_data=train_data,
            ri_strategy=ri_strategy,
            eval_frequency=eval_frequency,
        )
    else:
        print("Training without progress tracking...")
        rl_agent = train_agent(
            config=config, full_train_data=train_data, ri_strategy=ri_strategy
        )
        progress_data = []

    # --- Agent Evaluation ---
    evaluation_results = evaluate_agent(
        model=rl_agent,
        config=config,
        full_test_data=test_data,
        ri_strategy=ri_strategy,
    )

    # --- Save Results ---
    output_dir = "results/rl_agent_results"
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "rl_trading_model_daily.zip")
    rl_agent.save(model_path)
    print(f"Trained model saved to: {model_path}")

    results_path = os.path.join(output_dir, "evaluation_summary_daily.csv")
    evaluation_results.to_csv(results_path, index=False)
    print(f"Evaluation results saved to: {results_path}")

    config_path = os.path.join(output_dir, "config.yml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    print(f"Configuration saved to: {config_path}")

    # --- Generate Progress Plots ---
    if enable_tracking and progress_data:
        print("Generating training progress plots...")
        plot_training_progress(progress_data, output_dir)

        # Print summary of learning progress
        df = pd.DataFrame(progress_data)
        if len(df) > 1:
            initial_profit = df.iloc[0]["avg_raw_profit"]
            final_profit = df.iloc[-1]["avg_raw_profit"]
            improvement = final_profit - initial_profit
            print(f"\n=== LEARNING SUMMARY ===")
            print(f"Initial Raw Profit: {initial_profit:.2f}")
            print(f"Final Raw Profit: {final_profit:.2f}")
            print(f"Total Improvement: {improvement:.2f}")
            if improvement > 0:
                print("✅ Agent showed improvement during training!")
            else:
                print(
                    "⚠️ No improvement detected - consider adjusting hyperparameters"
                )

    # --- Generate Power Schedule Analysis ---
    print("\n=== Generating Power Schedule Analysis ===")
    try:
        from plot_power_schedule import (
            simulate_agent_day,
            plot_power_schedule,
            plot_commitment_vs_dispatch,
        )

        # Analyze the first few days to understand agent behavior
        unique_days = test_data["delivery_start"].dt.date.unique()
        num_analysis_days = min(3, len(unique_days))

        print(f"Analyzing power schedules for {num_analysis_days} days...")

        for i, day in enumerate(unique_days[:num_analysis_days]):
            daily_data = test_data.loc[
                (test_data["delivery_start"].dt.date == day)
                & (test_data["traded"].dt.date == day)
            ]

            if daily_data.empty:
                print(f"No data available for {day}")
                continue

            print(f"Analyzing day {i+1}: {day}")

            # Simulate the day with the trained agent and RI strategy
            results, env = simulate_agent_day(
                rl_agent, config, daily_data, f"Day {day}", ri_strategy
            )

            # Generate power schedule plots
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

            # Print detailed analysis
            final_soc = results["soc"].iloc[-1]
            final_cycles = results["cycles"].iloc[-1]
            total_reward = results["reward"].sum()
            max_dispatch = (
                results["dispatch_power"].abs().max() * env.battery.size_energy
            )
            total_trading_volume = (
                results["dispatch_power"].abs().sum()
                * env.battery.size_energy
                * 0.25
            )  # MWh

            print(f"  Final SOC: {final_soc:.3f}")
            print(
                f"  Cycles used: {final_cycles:.3f} / {config.get('N_daily_cycles_max', 2.0)}"
            )
            print(f"  Total reward: {total_reward:.0f}")
            print(f"  Max dispatch: {max_dispatch:.2f} MW")
            print(f"  Total trading volume: {total_trading_volume:.2f} MWh")

            if final_cycles > config.get("N_daily_cycles_max", 2.0):
                excess = final_cycles - config.get("N_daily_cycles_max", 2.0)
                print(f"  ⚠️ Cycle violation: {excess:.3f} excess cycles")
            else:
                print(f"  ✅ Within cycle limits")

        print("Power schedule analysis complete!")

    except Exception as e:
        print(f"Error generating power schedule analysis: {e}")
        print("Continuing with standard analysis...")

    print("\n=== Training and Evaluation Complete ===")
