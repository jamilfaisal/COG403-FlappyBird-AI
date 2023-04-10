import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr


def convert_frames_to_score(row):
    frame = int(row["episode_length"]) - 58
    if frame < 0:
        return 0
    elif frame < 43:
        return 1
    else:
        return (frame // 42) + 1


def convert_reward_to_score(df):
    previous = 0
    living_factor = 3.8999999999999986 / 48
    scores_list = []
    for i in range(len(df["iteration"])):
        iteration = df["iteration"][i]
        totalr = df["total_reward"][i]
        total_living_iter = iteration - previous
        score = totalr - living_factor * total_living_iter

        scores_list.append(int(score))
        previous = iteration

        # Debugging
        # if score != 0:
        #     print(score, iteration)
    return scores_list


def plot_training_graph_and_pearson_corr(dqn_file, dsn_file, ppo_file, random_file, save_name,
                                         frames_to_score=False,
                                         rewards_to_score=False):
    dqn_df = pd.read_csv(dqn_file)
    dsn_df = pd.read_csv(dsn_file)
    ppo_df = pd.read_csv(ppo_file)
    random_df = pd.read_csv(random_file)
    if frames_to_score:
        ppo_df["score"] = ppo_df.apply(lambda row: convert_frames_to_score(row), axis=1)
    if rewards_to_score:
        dqn_df["score"] = convert_reward_to_score(dqn_df)
        dsn_df["score"] = convert_reward_to_score(dsn_df)

    # PPO graph
    plt.scatter(ppo_df["iteration"], ppo_df["score"], label="PPO", c="green")
    # plt.plot(np.unique(ppo_df["iteration"]),
    #          np.poly1d(np.polyfit(ppo_df["iteration"], ppo_df["score"], 1))(np.unique(ppo_df["iteration"])),
    #          c="xkcd:light green"
    #          )
    print("PPO Pearson Correlation: ", pearsonr(ppo_df["iteration"], ppo_df["score"]))

    # DQN graph
    plt.scatter(dqn_df["iteration"], dqn_df["score"], label="DQN", c="orangered")
    print("DQN Pearson Correlation: ", pearsonr(dqn_df["iteration"], dqn_df["score"]))

    # DSN graph
    plt.scatter(dsn_df["iteration"], dsn_df["score"], label="DSN", c="blue")
    print("DSN Pearson Correlation: ", pearsonr(dsn_df["iteration"], dsn_df["score"]))

    # Random graph
    plt.scatter(random_df["iteration"], random_df["score"], label="Random Agent", c="grey", s=12)
    print("Random Pearson Correlation ", pearsonr(random_df["iteration"], random_df["score"]))

    plt.xlabel("Training Iteration")
    plt.ylabel("Score")
    plt.legend(loc="upper left")
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 7)
    plt.savefig(save_name)


if __name__ == "__main__":
    dqn_path = os.path.join("Files for Project Report", "output_dqn_train.csv")
    deep_sarsa_path = os.path.join("Files for Project Report", "output_dsn_train.csv")
    ppo_path = os.path.join("Files for Project Report", "output_ppo_train.csv")
    random_path = os.path.join("Files for Project Report", "output_random_train.csv")
    save_path = os.path.join("Files for Project Report", "training_vs_score_graph.png")

    plot_training_graph_and_pearson_corr(dqn_path, deep_sarsa_path, ppo_path, random_path, save_path, True, True)