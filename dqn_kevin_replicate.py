# https://cs229.stanford.edu/proj2015/362_report.pdf
import os
import random

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import time
import flappy_bird_gym
from scipy import ndimage
from skimage.measure import block_reduce
import numpy as np

from game.flappy_bird import GameState

EPSIL_FLOOR = 600000
MAX_ITER = 1800000
class KevinCNN(nn.Module):

    def __init__(self):
        super(KevinCNN, self).__init__()

        self.number_of_actions = 2
        self.gamma = 0.95
        self.final_epsilon = 0.01
        self.initial_epsilon = 0.1
        self.number_of_iterations = MAX_ITER
        self.replay_memory_size = 20000
        self.minibatch_size = 32

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
def image_to_tensor(image):
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available():  # put on GPU if CUDA is available
        image_tensor = image_tensor.cuda()
    return image_tensor

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))
def resize_and_bgr2gray(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image[0:288, 0:340]
    image_downsampled = ndimage.zoom(image,.3)
    image_data = cv2.resize(image_downsampled, (84, 84))
    #image_data[image_data > 0] = 255
    image_data = normalize(image_data)
    image_data = np.reshape(image_data, (84, 84, 1))
    return image_data

def init_weights(net):
    if type(net) == nn.Conv2d:
        net.weight.data.normal_(mean=0.0, std=np.sqrt(0.1))
        net.bias.data.zero_()
    if type(net) == nn.Linear:
        torch.nn.init.uniform(net.weight, -0.01, 0.01)
        net.bias.data.fill_(0.0)

def train(model, start):

    max_score = [0,0]
    total_reward = 0


    target_network = KevinCNN()
    target_network.load_state_dict(model.state_dict())
    if torch.cuda.is_available():  # put on GPU if CUDA is available
        target_network = target_network.cuda()
    # RMSP
    optimizer = optim.RMSprop(model.parameters(), lr=1e-6, weight_decay=0.9, momentum=0.95)

    # mean squared error formula for loss loss
    mse = nn.MSELoss()

    # instantiate game
    game_state = GameState()


    history_length = []
    # initialize replay memory
    replay_memory = []
    image_data_init = torch.zeros([1, 84, 84], dtype=torch.float32)

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    action_prev = action.unsqueeze(0)

    reward_init = 0
    reward_prev = torch.from_numpy(np.array([reward_init], dtype=np.float32)).unsqueeze(0)

    terminal_init = False
    terminal_prev = terminal_init

    image_data, reward, terminal, score = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)

    state_init = torch.cat(
        (image_data_init, image_data_init, image_data_init, image_data_init, image_data_init)).unsqueeze(0)
    state_prev = state_init

    # initialize epsilon value
    epsilon = model.initial_epsilon
    iteration = 0
    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, EPSIL_FLOOR)
    a = 0
    # main infinite loop
    while iteration < model.number_of_iterations:

        # obtain s_t
        if len(history_length) < 5:
            history_length.append(image_data)
        else:
            history_length.pop(0)
            history_length.append(image_data)
        # not sure
        if len(history_length) < 5:
            state = torch.cat((image_data, image_data, image_data, image_data, image_data)).unsqueeze(0)
        else:
            hist_tuple = (history_length[0], history_length[1], history_length[2], history_length[3], history_length[4])
            state = torch.cat(hist_tuple).unsqueeze(0)

        if torch.cuda.is_available():
            state = state.cuda()
            state_prev = state_prev.cuda()
            action_prev = action_prev.cuda()
            reward_prev = reward_prev.cuda()
        # save transition to replay memory
        replay_memory.append((state_prev, action_prev, reward_prev, state, terminal_prev))
        # if replay memory is full, remove the oldest transition
        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        state_prev = state
        reward_prev = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)
        terminal_prev = terminal
        # get output from the neural network for taking best action
        output = model(state)[0]

        # initialize action
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # epsilon greedy exploration
        if random.uniform(0, 1) <= epsilon:
            random_action = True
        else:
            random_action = False
        # if not random_action:
        #     print("Performed not random action!")

        # index the best action
        if random_action:
            k = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)]
            action_index = k[0]
        else:
            k = [torch.argmax(output)]
            action_index = k[0]

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()

        action[action_index] = 1
        action_prev = action.unsqueeze(0)

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
            output_1_batch = target_network(state_1_batch)

            # set y_j to r_j for crash(terminal state), otherwise to r_j + gamma*max(Q)
            y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                      else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
                                      for i in range(len(minibatch))))
            # for i in range(len(minibatch)):
            #     if minibatch[i][4] is True:
            #         torch.cat(tuple(reward_batch[i]))
            #     else:
            #         y_batch = reward_batch[i] + model.gamma * torch.max(output_1_batch[i])

            # extract Q-value
            #print(model(state_batch))
            # not sure
            q_value = torch.sum(model(state_batch) * action_batch, dim=1)
            # PyTorch accumulates gradients by default, so they need to be reset in each pass
            optimizer.zero_grad()

            # returns a new Tensor, detached from the current graph, the result will never require gradient
            y_batch = y_batch.detach()

            # calculate loss
            loss = mse(q_value, y_batch)

            # do backward pass
            loss.backward()
            optimizer.step()
            # get next state and reward

        # epsilon decrease
        if iteration < EPSIL_FLOOR:
            epsilon = epsilon_decrements[iteration]
        else:
            epsilon = epsilon_decrements[EPSIL_FLOOR-1]

        # target network update for stability
        if iteration % 1000 == 0:
            target_network.load_state_dict(model.state_dict())

        # take the best action
        image_data, reward, terminal, score = game_state.frame_step(action)
        image_data = resize_and_bgr2gray(image_data)
        image_data = image_to_tensor(image_data)
        # not sure
        state = torch.cat((state.squeeze(0)[1:, :, :], image_data)).unsqueeze(0)
        if torch.cuda.is_available():
            state = state.cuda()
        # print(action)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)
        total_reward += reward

        # replay_memory.append((state, action, reward, state_next, terminal))
        # set state to be state_1
        iteration += 1
        # q = output.cpu().detach().numpy()
        # qm = np.max(output.cpu().detach().numpy())
        if score > max_score[0]:
            max_score[0] = score
            max_score[1] = iteration

        if iteration % 100000 == 0:
            torch.save(model, "pm7/current_model_" + str(iteration) + ".pth")
        if iteration % 30000 == 0:
            print( "iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
                  action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
                  np.max(output.cpu().detach().numpy()), ""
                  "score:", max_score, "location:", "pm7",
                  "mem_q_val:", sum(q_value.cpu().detach().numpy()), "y_batch:", sum(y_batch.cpu().detach().numpy()))
        if iteration ==3000:
            print( "iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
                  action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
                  np.max(output.cpu().detach().numpy()), ""
                  "score:", max_score, "location:", "pm7",
                  "mem_q_val:", sum(q_value.cpu().detach().numpy()), "y_batch:", sum(y_batch.cpu().detach().numpy()))
        if iteration == 0 or iteration == 300 or iteration == 30:
            print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
                  action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
                  np.max(output.cpu().detach().numpy()), ""
                                                         "score:", max_score, "location:", "pm7",)


def test(model):
    game_state = GameState()

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal, score = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data,image_data)).unsqueeze(0)
    iteration = 0

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
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        # set state to be state_1
        state = state_1
        iteration += 1

        if iteration % 150 == 0:
            # print( "iteration:", iteration,   "action:",
            #       action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
            #       np.max(output.cpu().detach().numpy()), "score:", score)\
            a = 1

def run(mode):
    cuda_is_available = torch.cuda.is_available()

    if mode == 'test':
        model = torch.load(
            'pm/current_model_600000.pth',
            map_location='cpu' if not cuda_is_available else None
        ).eval()

        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()

        test(model)
    if mode == 'train':
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')

        model = KevinCNN()

        # model = torch.load(
        #     'pm1_epsilon_1_to_0.1/current_model_320000.pth',
        #     map_location='cpu' if not cuda_is_available else None
        # )


        if torch.cuda.is_available():  # put on GPU if CUDA is available
            model = model.cuda()

        model.apply(init_weights)
        start = time.time()

        train(model, start)

if __name__ == "__main__":

    print(f"Is CUDA supported by this system? ,"+str(torch.cuda.is_available()))
    print(f"CUDA version: "+torch.version.cuda)

    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device:"+ str(torch.cuda.current_device()))

    print(f"Name of current CUDA device:"+  str(  torch.cuda.get_device_name(cuda_id)))

    run('train')

# env = flappy_bird_gym.make("FlappyBird-v0")
# env_rgb = flappy_bird_gym.make("FlappyBird-rgb-v0")
#
# obs = env.reset()
# obs_rgb = env_rgb.reset()
# while True:
#     # Next action:
#     # (feed the observation to your agent here)
#     action = env.action_space.sample()  # env.action_space.sample() for a random action
#
#     # Processing:
#     obs, reward, done, info = env.step(action)
#     info  = env_rgb.step(action)
#     # Rendering the game:
#     # (remove this two lines during training)
#     env.render()
#     time.sleep(1 / 30)  # FPS
#
#     # Checking if the player is still alive
#     if done:
#         break
#
# env.close()