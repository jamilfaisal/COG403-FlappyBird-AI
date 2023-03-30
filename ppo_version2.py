"""
This version uses two different neural networks and uses two policies.
"""

import os
import time

import cv2
import numpy as np
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from game.flappy_bird import GameState


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        # Environment parameters
        self.number_of_actions = 2

        # Actor Neural network
        self.actor = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.number_of_actions),
            nn.Softmax(dim=-1)
        )

        # Critic Neural Network
        self.critic = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )

    def actor_output(self, state):
        # Forward pass
        action_softmax = self.actor(state)
        action_distribution = Categorical(action_softmax)

        action = action_distribution.sample()
        action_log_prob = action_distribution.log_prob(action)

        return action, action_log_prob

    def forward(self):
        raise NotImplementedError

    def critic_output(self, state, action):
        # Forward pass
        action_softmax = self.actor(state)
        action_distribution = Categorical(action_softmax)
        action_log_prob = action_distribution.log_prob(action)

        entropy = action_distribution.entropy()
        critic_state_value = self.critic(state)
        return critic_state_value, action_log_prob, entropy

class PPO(nn.Module):

    def __init__(self):
        super(PPO, self).__init__()

        # Model saving parameters
        self.save_modulo = 25000
        self.save_folder = "pm_ppo"

        # Hyperparameters
        self.number_of_iterations = 3000000
        self.optimization_modulo = 40
        self.gamma = 0.99
        self.gradient_clip = 0.1
        self.max_gradient = 40 # To prevent the gradient from exploding
        self.value_loss_regularizer = 0.5
        self.entropy_loss_regularizer = 0.01

        # Initialize actor and critic policies
        self.policy_main = Policy().to(device)
        self.policy_previous = Policy().to(device)
        self.policy_previous.load_state_dict(self.policy_main.state_dict())

        # Optimizers
        self.optimizer = optim.Adam([
            {"params": self.policy_main.actor.parameters(), "lr": 1e-5},
            {"params": self.policy_main.critic.parameters(), "lr": 1e-5},
        ])

        # MSE loss
        self.mse = nn.MSELoss()

        # Replay memory
        self.states = []
        self.actions = []
        self.action_logSoftmax = []
        self.rewards = []
        self.terminals = []

    def optimize_custom(self):
        # state, action, action_logSoftmax, critic_state_value, reward, terminal = process_replay_memory(replay_memory)

        states = torch.stack(self.states, dim=0).to(device)
        actions = torch.stack(self.actions, dim=0).to(device)
        action_logSoftmax_prev = torch.stack(self.action_logSoftmax, dim=0).to(device)

        state_shape = states.size()[2:]
        action_shape = actions.size()[-1]

        # Calculate returns
        returns = []
        disc_reward = 0.
        for i in reversed(range(len(self.rewards))):
            if self.terminals[i]:
                disc_reward = 0.
            disc_reward = self.rewards[i] + self.gamma * disc_reward
            returns.append(disc_reward)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        # returns = (returns - returns.mean()) / (returns.std((len(self.rewards) + 1, 1, 1)) + 1e-5)

        # Process batch
        critic_state_value, action_logSoftmax, entropy = self.policy_main.critic_output(states.view(-1, *state_shape),
                                                                                        actions.view(-1, action_shape))

        advantages = returns - critic_state_value.detach()

        # Calculate Action loss
        ratio = torch.exp(action_logSoftmax - action_logSoftmax_prev)
        clamped_ratio = torch.clamp(ratio, 1 - self.gradient_clip, 1 + self.gradient_clip)
        action_loss = -torch.min(ratio, clamped_ratio) * advantages

        # Value loss
        # value_loss = (returns - critic_state_value).pow(2).mean()
        # value_loss = self.value_loss_regularizer * value_loss
        value_loss = self.value_loss_regularizer * self.mse(critic_state_value, returns)

        # Entropy loss
        entropy_loss = -self.entropy_loss_regularizer * entropy

        # Total loss
        total_loss = action_loss + value_loss + entropy_loss

        # Optimizer step
        self.optimizer.zero_grad()
        total_loss.mean().backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), self.max_gradient)
        self.optimizer.step()

        self.policy_previous.load_state_dict(self.policy_main.state_dict())

        return total_loss.mean(), value_loss, action_loss.mean(), entropy_loss.mean()


# def process_replay_memory(replay_memory):
#     state = []
#     action = []
#     action_log_prob = []
#     value = []
#     reward = []
#     mask = []
#     for memory in replay_memory:
#         state.append(memory[0])
#         action.append(memory[1])
#         action_log_prob.append(memory[2])
#         value.append(memory[3])
#         reward.append([[memory[4]]])
#         mask.append(memory[5])
#
#     state = torch.stack(state).detach()
#     action = torch.stack(action).detach()
#     action_logSoftmax = torch.stack(action_log_prob).detach()
#     critic_state_value = torch.stack(value).detach()
#     reward = torch.tensor(reward).detach()
#     terminal = torch.stack(mask).detach()
#
#     return state, action, action_logSoftmax, critic_state_value, reward, terminal


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)

def image_to_tensor(image):
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available():  # put on GPU if CUDA is available
        image_tensor = image_tensor.cuda()
    return image_tensor


def resize_and_bgr2gray(image):
    image = image[0:288, 0:404]
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))
    return image_data

def train(model: PPO, start):

    # instantiate game
    game_state = GameState()

    # Initialize episode length
    episode_length = 0

    # Initial action is to do nothing
    action = torch.zeros([model.policy_main.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, _, _, _ = game_state.frame_step(action)

    # Transform image to get initial state
    image_data = image_to_tensor(resize_and_bgr2gray(image_data))
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    for iteration in range(1, model.number_of_iterations):

        # Get next action given state
        with torch.no_grad():
            action, action_logSoftmax = model.policy_previous.actor_output(state)
        if action.detach().item() == 0:
            action_for_game = [1, 0]
        else:
            action_for_game = [0, 1]

        # Execute action and get next state and reward
        image_data, reward, terminal, score = game_state.frame_step(action_for_game)

        image_data = image_to_tensor(resize_and_bgr2gray(image_data))
        next_state = torch.cat((state.squeeze(0)[1:, :, :], image_data)).unsqueeze(0)

        # Save transition to replay memory
        model.states.append(state.detach())
        model.actions.append(action.detach())
        model.action_logSoftmax.append(action_logSoftmax.detach())
        model.rewards.append(reward)
        model.terminals.append(terminal)

        # replay_memory.append((state, action, action_logSoftmax, reward, torch.tensor([[float(terminal)]])))

        # Optimization
        if iteration % model.optimization_modulo == 0:
            total_loss, value_loss, action_loss, entropy_loss = model.optimize_custom()
            print("Total loss: ", total_loss)
            print("Value Loss: ", value_loss)
            print("Action Loss: ", action_loss)
            print("Entropy Loss: ", entropy_loss)
            print("")
            # Reset memory
            model.states = []
            model.actions = []
            model.action_logSoftmax = []
            model.rewards = []
            model.terminals = []

        # Update/Print & Reset episode length
        if terminal is False:
            episode_length += 1
        else:
            # TODO Save this value somewhere to generate a graph
            if episode_length != 49:
                print(episode_length)
            episode_length = 0

        # Save model
        if iteration % model.save_modulo == 0:
            if not os.path.exists(model.save_folder):
                os.mkdir(model.save_folder)
            torch.save(model.state_dict(), os.path.join(model.save_folder, str(iteration) + ".pth"))
            print("Iteration: ", iteration)
            print("Elapsed Time: ", time.time() - start)


        # Set current state as the next state
        state = next_state


def main(mode):
    cuda_is_available = torch.cuda.is_available()

    if mode == "train":
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')

        model = PPO()
        if cuda_is_available:
            model = model.cuda()
        model.apply(init_weights)

        start = time.time()
        train(model, start)



if __name__ == "__main__":
    main("train")
