import csv
import os
import random
import time

from game.flappy_bird import GameState

NUMBER_OF_ITERATIONS = 2000000
SAVE_MODULO = 100000
SAVE_FOLDER = "random_output"
def train(start):

    # instantiate game
    game_state = GameState(caption="random_train")

    # Initialize iteration, episode_length list
    it_score_list = []

    iteration = 0

    # main infinite loop
    while iteration < NUMBER_OF_ITERATIONS:

        random_action = random.randint(0, 1)
        print(random_action)
        action = [0, 0]
        action[random_action] = 1
        image_data, reward, terminal, score = game_state.frame_step(action)

        if terminal:
            it_score_list.append([iteration, score])

        if iteration % SAVE_MODULO == 0:
            if not os.path.exists(SAVE_FOLDER):
                os.mkdir(SAVE_FOLDER)
            with open(os.path.join(SAVE_FOLDER, "output_random_train.csv"), "w", newline='') as f:
                csv_output = csv.writer(f)
                csv_output.writerow(["iteration", "score"])
                csv_output.writerows(it_score_list)
            print("Iteration: ", iteration)
            print("Elapsed Time: ", time.time() - start)

        iteration += 1


def test():
    # instantiate game
    game_state = GameState(caption="random_test")

    # Initialize run #, score list
    run_score_list = []

    run = 1
    prev_score = 0

    while run < 11:
        random_action = random.randint(0, 1)
        action = [0, 0]
        action[random_action] = 1
        image_data, reward, terminal, score = game_state.frame_step(action)

        if terminal:
            print("Run {}, Score {}".format(run, prev_score))
            run_score_list.append([run, prev_score])
            run += 1
        prev_score = score

    with open(os.path.join(SAVE_FOLDER, "output_random_test.csv"), "w", newline='') as f:
        csv_output = csv.writer(f)
        csv_output.writerow(["run", "score"])
        csv_output.writerows(run_score_list)


def main(mode):

    if mode == 'test':
        test()
    elif mode == 'train':
        start = time.time()
        train(start)


if __name__ == "__main__":
    main('test')
