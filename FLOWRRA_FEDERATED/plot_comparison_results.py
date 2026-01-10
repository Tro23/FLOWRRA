import pandas as pd
import matplotlib.pyplot as plt
import glob

def plot_flowrra_results():
    # Find the specific reward mean file in your experiment folder
    reward_files = glob.glob("benchmarl_results/*/scalars/collection_reward_reward_mean.csv")

    plt.figure(figsize=(10, 6))
    for file in reward_files:
        # Extract the experiment name from the path
        name = file.split('/')[-6]
        df = pd.read_csv(file)
        plt.plot(df.iloc[:, 0], df.iloc[:, 1], label=f"{name} Reward")

    plt.title("32-Agent Performance: FLOWRRA vs SOTA")
    plt.xlabel("Step")
    plt.ylabel("Mean Reward")
    plt.legend()
    plt.show()
