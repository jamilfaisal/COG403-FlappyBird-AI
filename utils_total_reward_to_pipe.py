import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

living_factor = 3.8999999999999986/48
counter = 0

def convert_frames_to_score(counter, prev_0, row):
    a = row["iteration"]
    total_living_iter = row["iteration"] - prev_0
    score = row["total_reward"] - living_factor*total_living_iter
    score = int(score)
    prev = row["iteration"]

    if counter != 0:
        a = 1
    counter += 1


    return score
def plot_training_graph(csv_file, save_name, frames_to_score=False):
    df = pd.read_csv(csv_file)
    a = df["iteration"]
    b = df["total_reward"]
    prev = 0
    score_lst = []
    for i in range(len(a)):
        iteration = a[i]
        totalr = b[i]
        total_living_iter = iteration - prev
        score = totalr - living_factor * total_living_iter
        score = int(score)
        score_lst.append(score)
        prev = iteration
        if score != 0:
            print(score, iteration)

    if frames_to_score:

        df["score"] = df.apply(lambda row: convert_frames_to_score(counter, prev, row), axis=1)
    plt.scatter(df["iteration"], score_lst)
    distance = [0,1,2,3,4]
    plt.yticks(range(len(distance)), distance)
    plt.plot(np.unique(df["iteration"]), np.poly1d(np.polyfit(df["iteration"], score_lst, 1))(np.unique(df["iteration"])))
    # sns.regplot(x=df["iteration"], y=df["score"])
    plt.show()
    plt.savefig(save_name)


if __name__ == "__main__":
    plot_training_graph(os.path.join("100gap", "pandas", "output_dsn.csv"),
                        "pm_ppo_final_version_with_8_instances_training.png",
                        True)