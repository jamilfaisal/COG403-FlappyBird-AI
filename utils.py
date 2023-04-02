import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def convert_frames_to_score(row):
    frame = int(row["episode_length"]) - 58
    if frame < 0:
        return 0
    elif frame < 43:
        return 1
    else:
        return (frame // 42) + 1
def plot_training_graph(csv_file, save_name, frames_to_score=False):
    df = pd.read_csv(csv_file)
    if frames_to_score:
        df["score"] = df.apply(lambda row: convert_frames_to_score(row), axis=1)
    plt.scatter(df["iteration"], df["score"])
    plt.plot(np.unique(df["iteration"]), np.poly1d(np.polyfit(df["iteration"], df["score"], 1))(np.unique(df["iteration"])))
    # sns.regplot(x=df["iteration"], y=df["score"])
    # plt.show()
    plt.savefig(save_name)


if __name__ == "__main__":
    plot_training_graph(os.path.join("PPO", "pm_ppo_final_version_with_8_instances", "output.csv"),
                        "pm_ppo_final_version_with_8_instances_training.png",
                        True)