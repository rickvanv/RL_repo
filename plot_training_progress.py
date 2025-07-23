import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from pathlib import Path

# Set style for better looking plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def plot_training_progress(results_dir="results/rl_agent_results"):
    """
    Plot training progress from saved results.
    """
    results_path = os.path.join(results_dir, "evaluation_summary_daily.csv")

    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    # Load results
    df = pd.read_csv(results_path)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("RL Agent Training Progress", fontsize=16, fontweight="bold")

    # Plot 1: Total Reward vs Raw Profit by Day
    ax1 = axes[0, 0]
    days = range(len(df))
    ax1.plot(
        days,
        df["total_reward"],
        "o-",
        label="Total Reward (with penalties)",
        color="red",
        alpha=0.7,
    )
    ax1.plot(
        days,
        df["raw_profit"],
        "s-",
        label="Raw Trading Profit",
        color="green",
        alpha=0.8,
    )
    ax1.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Day Index")
    ax1.set_ylabel("Profit")
    ax1.set_title("Daily Performance: Rewards vs Raw Profits")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cycle Usage
    ax2 = axes[0, 1]
    ax2.plot(days, df["cycles_used"], "o-", color="blue", alpha=0.8)
    max_cycles = 2.0  # Assuming this is your daily max
    ax2.axhline(
        y=max_cycles,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Max Cycles ({max_cycles})",
    )
    ax2.set_xlabel("Day Index")
    ax2.set_ylabel("Cycles Used")
    ax2.set_title("Daily Cycle Usage")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Final SOC
    ax3 = axes[1, 0]
    ax3.plot(days, df["final_soc"], "o-", color="orange", alpha=0.8)
    ax3.axhline(
        y=0.5, color="black", linestyle="--", alpha=0.5, label="50% SOC"
    )
    ax3.set_xlabel("Day Index")
    ax3.set_ylabel("Final SOC")
    ax3.set_title("Final State of Charge")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary Statistics
    ax4 = axes[1, 1]
    metrics = ["Raw Profit", "Total Reward", "Cycles Used", "Final SOC"]
    values = [
        df["raw_profit"].mean(),
        df["total_reward"].mean(),
        df["cycles_used"].mean(),
        df["final_soc"].mean(),
    ]
    colors = ["green", "red", "blue", "orange"]

    bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
    ax4.set_title("Average Performance Metrics")
    ax4.set_ylabel("Average Value")

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + height * 0.01,
            f"{value:.1f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(results_dir, "training_progress.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Training progress plot saved to: {output_path}")

    plt.show()

    # Print summary statistics
    print("\n=== TRAINING SUMMARY ===")
    print(f"Days evaluated: {len(df)}")
    print(f"Average raw trading profit: {df['raw_profit'].mean():.2f}")
    print(f"Average total reward: {df['total_reward'].mean():.2f}")
    print(f"Average cycles used: {df['cycles_used'].mean():.2f}")
    print(f"Raw profit consistency (std): {df['raw_profit'].std():.2f}")

    # Check if agent is learning
    if len(df) > 3:
        first_half = df.iloc[: len(df) // 2]["raw_profit"].mean()
        second_half = df.iloc[len(df) // 2 :]["raw_profit"].mean()
        improvement = second_half - first_half
        print(
            f"Profit improvement (second half vs first half): {improvement:.2f}"
        )
        if improvement > 0:
            print("✅ Agent appears to be learning (profits improving)")
        else:
            print("⚠️ No clear learning trend detected")


def create_epoch_tracking_script():
    """
    Create a modified training script that tracks progress across epochs.
    """
    training_script = '''
import os
import pandas as pd
import numpy as np
from src.rl_trading.train_rl_agent import train_agent, evaluate_agent
import yaml
import pickle

def train_with_progress_tracking(config, train_data, test_data, eval_frequency=50):
    """
    Train the agent and evaluate progress at regular intervals.
    """
    print("Starting training with progress tracking...")
    
    # Create results directory
    results_dir = "results/training_progress"
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize tracking
    progress_data = []
    
    # Get total epochs
    total_epochs = config.get("n_training_epochs", 500)
    
    # Split training into chunks for periodic evaluation
    chunk_size = eval_frequency
    
    for start_epoch in range(0, total_epochs, chunk_size):
        end_epoch = min(start_epoch + chunk_size, total_epochs)
        
        # Update config for this chunk
        chunk_config = config.copy()
        chunk_config["n_training_epochs"] = end_epoch - start_epoch
        
        print(f"Training epochs {start_epoch+1} to {end_epoch}...")
        
        # Train for this chunk
        if start_epoch == 0:
            # First training
            model = train_agent(chunk_config, train_data)
        else:
            # Continue training from previous model
            model = train_agent(chunk_config, train_data)
        
        # Evaluate current performance
        print(f"Evaluating after epoch {end_epoch}...")
        eval_results = evaluate_agent(model, config, test_data)
        
        # Add epoch info to results
        eval_results['epoch'] = end_epoch
        eval_results['training_step'] = start_epoch // chunk_size
        
        # Calculate summary metrics
        avg_raw_profit = eval_results['raw_profit'].mean()
        avg_total_reward = eval_results['total_reward'].mean()
        avg_cycles = eval_results['cycles_used'].mean()
        
        progress_data.append({
            'epoch': end_epoch,
            'avg_raw_profit': avg_raw_profit,
            'avg_total_reward': avg_total_reward,
            'avg_cycles_used': avg_cycles,
            'profit_std': eval_results['raw_profit'].std(),
        })
        
        print(f"Epoch {end_epoch}: Avg Raw Profit = {avg_raw_profit:.2f}, Avg Total Reward = {avg_total_reward:.2f}")
        
        # Save incremental results
        progress_df = pd.DataFrame(progress_data)
        progress_df.to_csv(os.path.join(results_dir, "epoch_progress.csv"), index=False)
        
        # Save detailed results for this evaluation
        eval_results.to_csv(os.path.join(results_dir, f"eval_epoch_{end_epoch}.csv"), index=False)
    
    print("Training with progress tracking complete!")
    return model, progress_data

if __name__ == "__main__":
    # Load config and data
    with open("src/optimization/config.yml", "r") as f:
        config = yaml.safe_load(f)
    
    with open("src/optimization/test_data/df_test_data_winter.pkl", "rb") as f:
        data = pickle.load(f)
    
    # Run training with tracking
    final_model, progress = train_with_progress_tracking(
        config, data, data, eval_frequency=100  # Evaluate every 100 epochs
    )
'''

    # Save the tracking script
    with open("src/rl_trading/train_with_tracking.py", "w") as f:
        f.write(training_script)

    print(
        "Created enhanced training script: src/rl_trading/train_with_tracking.py"
    )


def plot_epoch_progress(results_dir="results/training_progress"):
    """
    Plot progress across training epochs from epoch tracking data.
    """
    progress_path = os.path.join(results_dir, "epoch_progress.csv")

    if not os.path.exists(progress_path):
        print(f"Epoch progress file not found: {progress_path}")
        print(
            "Run train_with_tracking.py first to generate epoch-by-epoch data."
        )
        return

    # Load epoch progress data
    df = pd.read_csv(progress_path)

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

    # Add trend line
    z = np.polyfit(df["epoch"], df["avg_raw_profit"], 1)
    p = np.poly1d(z)
    ax1.plot(df["epoch"], p(df["epoch"]), "--", alpha=0.8, color="darkgreen")

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
    output_path = os.path.join(results_dir, "epoch_progress_plot.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Epoch progress plot saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    print("=== RL Training Progress Visualization ===")
    print("1. Plotting current results...")
    plot_training_progress()

    print("\n2. Creating enhanced training script for epoch tracking...")
    create_epoch_tracking_script()

    print("\n3. Checking for epoch progress data...")
    plot_epoch_progress()
