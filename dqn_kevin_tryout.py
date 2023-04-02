import csv
import os
import random
import sys
import time

import cv2
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.optim as optim

from game.flappy_bird import GameState


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.075
        self.number_of_iterations = 2000000
        self.replay_memory_size = 20000
        self.minibatch_size = 32
        self.save_modulo = 100000
        self.save_folder = "100gap"

        self.conv1 = nn.Conv2d(in_channels=5, out_channels=32, kernel_size=8, stride=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)

        return out


def init_weights(net):
    if type(net) == nn.Conv2d:
        torch.nn.init.uniform(net.weight, -0.01, 0.01)
        net.bias.data.fill_(0.01)
    if type(net) == nn.Linear:
        torch.nn.init.uniform(net.weight, -0.01, 0.01)
        net.bias.data.fill_(0.01)



def image_to_tensor(image):
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available():  # put on GPU if CUDA is available
        image_tensor = image_tensor.cuda()
    return image_tensor


# def resize_and_bgr2gray(image):
#     image = image[0:288, 0:404]
#     image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
#     image_data[image_data > 0] = 255
#     image_data = np.reshape(image_data, (84, 84, 1))
#     return image_data

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))

def resize_and_bgr2gray(image):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image[0:288, 0:340]
    #image_downsampled = ndimage.zoom(image,.3)
    #image_data = cv2.resize(image_downsampled, (84, 84))
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    #image_data = normalize(image_data)
    image_data = np.reshape(image_data, (84, 84, 1))
    return image_data


def train(model, start):
    # define Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    # initialize mean squared error loss
    criterion = nn.MSELoss()

    # instantiate game
    game_state = GameState(caption="dqn_5c_255")

    # initialize replay memory
    replay_memory = []
    max_reward = 0
    it_rw_lst = []

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal, score = game_state.frame_step(action)
    max_reward += reward
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)

    state = torch.cat((image_data, image_data, image_data, image_data, image_data)).unsqueeze(0)

    # initialize epsilon value
    epsilon = model.initial_epsilon
    iteration = 0

    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)

    max_score = [0,0]
    # main infinite loop
    while iteration < model.number_of_iterations:
        # get output from the neural network
        output = model(state)[0]

        # initialize action
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # epsilon greedy exploration
        random_action = random.random() <= epsilon
        # if random_action:
        #     print("Performed random action!")
        action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()

        action[action_index] = 1

        # get next state and reward
        image_data_1, reward, terminal, score = game_state.frame_step(action)
        max_reward += reward
        if iteration == 300 or iteration == 2456 or iteration == 1044:
            a = 0
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)

        state_1 = torch.cat((state.squeeze(0)[1:, :, :,], image_data_1)).unsqueeze(0)

        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        # save transition to replay memory
        replay_memory.append((state, action, reward, state_1, terminal))

        # if replay memory is full, remove the oldest transition
        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        # epsilon annealing
        epsilon = epsilon_decrements[iteration]

        # sample minibatch and train w/ back prop
        if len(replay_memory) >= 3000:
        # sample random minibatch
            minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

            # unpack minibatch
            state_batch = torch.cat(tuple(d[0] for d in minibatch))
            action_batch = torch.cat(tuple(d[1] for d in minibatch))
            reward_batch = torch.cat(tuple(d[2] for d in minibatch))
            state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

            if torch.cuda.is_available():  # put on GPU if CUDA is available
                state_batch = state_batch.cuda()
                action_batch = action_batch.cuda()
                reward_batch = reward_batch.cuda()
                state_1_batch = state_1_batch.cuda()

            # get output for the next state
            output_1_batch = model(state_1_batch)

            # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
            y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                      else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
                                      for i in range(len(minibatch))))

            # extract Q-value
            q_value = torch.sum(model(state_batch) * action_batch, dim=1)

            # PyTorch accumulates gradients by default, so they need to be reset in each pass
            optimizer.zero_grad()

            # returns a new Tensor, detached from the current graph, the result will never require gradient
            y_batch = y_batch.detach()

            # calculate loss
            loss = criterion(q_value, y_batch)

            # do backward pass
            loss.backward()
            optimizer.step()

        if terminal is True:
            it_rw_lst.append([iteration, max_reward])
            max_reward = 0

        # set state to be state_1
        state = state_1
        iteration += 1


        if score >= max_score[0]:
            max_score[0] = score
            max_score[1] = iteration

        if iteration % model.save_modulo == 0:
            torch.save(model, "100gap/dqn/current_model_" + str(iteration) + ".pth")
            with open(os.path.join(model.save_folder, "output_dqn.csv"), "w", newline='') as f:
                csv_output = csv.writer(f)
                csv_output.writerow(["iteration", "total_reward"])
                csv_output.writerows(it_rw_lst)
            a = 1
        if iteration % 3000 == 0:
            print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
                  action_index.cpu().detach().numpy(), "max reward:", max_reward , "Q max:",
                  np.max(output.cpu().detach().numpy()), ""
                                                         "score:", max_score, "location:", "pm7",
                  "mem_q_val:", sum(q_value.cpu().detach().numpy()), "y_batch:", sum(y_batch.cpu().detach().numpy()))
        if iteration == 3000:
            print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
                  action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
                  np.max(output.cpu().detach().numpy()), ""
                                                         "score:", max_score, "location:", "pm7",
                  "mem_q_val:", sum(q_value.cpu().detach().numpy()), "y_batch:", sum(y_batch.cpu().detach().numpy()))
        if iteration == 0 or iteration == 300 or iteration == 30 or iteration == 10000:
            print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
                  action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
                  np.max(output.cpu().detach().numpy()), ""
                                                         "score:", max_score, "location:", "pm7", )


def test(model):
    game_state = GameState()

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal, score = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)




    state = torch.cat((image_data, image_data, image_data, image_data, image_data)).unsqueeze(0)

    while True:
        # get output from the neural network
        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # get action
        action_index = torch.argmax(output)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()
        action[action_index] = 1

        # get next state
        image_data_1, reward, terminal, score = game_state.frame_step(action)
        print(score)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)

        state_1 = torch.cat((state.squeeze(0)[1:, :, :,], image_data_1)).unsqueeze(0)

        # set state to be state_1
        state = state_1


def main(mode):
    cuda_is_available = torch.cuda.is_available()

    if mode == 'test':
        model = torch.load(
            '100gap/dqn/current_model_1000000.pth',
            map_location='cpu' if not cuda_is_available else None
        ).eval()

        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()

        test(model)

    elif mode == 'train':
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')

        model = NeuralNetwork()

        # model = torch.load(
        #     'pretrained_model/current_model_500000.pth',
        #     map_location='cpu' if not cuda_is_available else None
        # )
        # model.initial_epsilon = 0.75


        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()

        model.apply(init_weights)
        start = time.time()

        train(model, start)


if __name__ == "__main__":
    main('test')
