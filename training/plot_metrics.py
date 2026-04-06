import json
import matplotlib.pyplot as plt

def plot_learning_curve(json_path="scores.json", output_path="learning_curve.png"):
    try:
        with open(json_path, "r") as f:
            scores = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_path} not found. Please run train.py first.")
        return

    # Calculate moving average (window=10)
    window = 10
    moving_averages = []
    for i in range(len(scores)):
        if i < window:
            moving_averages.append(sum(scores[:i+1]) / (i+1))
        else:
            moving_averages.append(sum(scores[i-window+1:i+1]) / window)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(scores, label="Episode Reward", alpha=0.3, color="gray")
    plt.plot(moving_averages, label=f"Moving Average (k={window})", color="blue", linewidth=2)
    
    plt.title("Agent Performance over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved successfully to {output_path}")

if __name__ == "__main__":
    plot_learning_curve()
