import csv
import os
import time

import cv2
import flappy_bird_gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pygame.surfarray import make_surface, array2d
from torch.distributions import Categorical


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()

        self.number_of_actions = 2

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

    def forward(self):
        raise NotImplementedError

    def actor_output(self, state):
        action_softmax = self.actor(state)
        action_distribution = Categorical(action_softmax)

        action = action_distribution.sample()
        action_log_prob = action_distribution.log_prob(action)
        critic_state_value = self.critic(state)

        return action.detach(), action_log_prob.detach(), critic_state_value.detach()

    def critic_output(self, state, action):
        action_softmax = self.actor(state)
        action_distribution = Categorical(action_softmax)
        action_log_prob = action_distribution.log_prob(action)

        entropy = action_distribution.entropy()
        critic_state_values = self.critic(state)

        return action_log_prob, critic_state_values, entropy

class PPO:

    def __init__(self):
        # Model saving parameters
        self.save_modulo = 100000
        self.save_folder = "pm_ppo_version4"

        # Hyperparameters
        self.number_of_iterations = 3000000
        self.gamma = 0.99 # discount factor
        self.gradient_clip = 0.2 # gradient clip parameter
        self.epochs = 80 # How many epochs in one optimization update
        self.optimization_modulo = 1200 # How many timesteps before optimizing
        self.actor_lr = 0.0003 # Actor learning rate
        self.critic_lr = 0.001 # Critic learning rate

        # Memory Replay
        self.actions = []
        self.states = []
        self.action_logSoftmaxes = []
        self.rewards = []
        self.critic_state_values = []
        self.terminals = []

        # Initialize actor and critic policies
        self.policy_main = Policy().to(device)
        self.optimizer = optim.Adam([
            {"params": self.policy_main.actor.parameters(), 'lr': self.actor_lr},
            {"params": self.policy_main.critic.parameters(), 'lr': self.critic_lr}
        ])

        self.policy_old = Policy().to(device)
        self.mse = nn.MSELoss()

    def pick_action(self, state):
        with torch.no_grad():
            action, action_logSoftmax, critic_state_value = self.policy_old.actor_output(state)

        self.states.append(state)
        self.actions.append(action)
        self.action_logSoftmaxes.append(action_logSoftmax)
        self.critic_state_values.append(critic_state_value)

        return action.item()

    def optimize_custom(self):

        # Convert lists to tensors
        prev_states = torch.stack(self.states, dim=0).to(device)
        prev_actions = torch.stack(self.actions, dim=0).to(device)
        prev_action_logSoftmaxes = torch.stack(self.action_logSoftmaxes, dim=0).to(device)
        prev_critic_state_values = torch.stack(self.critic_state_values, dim=0).to(device)

        prev_states_shape = prev_states.size()[2:]
        prev_actions_shape = prev_actions.size()[-1]


        # Monte carlo estimate of rewards
        rewards = []
        disc_reward = 0
        for reward, terminal in zip(reversed(self.rewards), reversed(self.terminals)):
            if terminal:
                disc_reward = 0
            disc_reward = reward + self.gamma * disc_reward
            rewards.insert(0, disc_reward)

        # Normalize rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Calculate advantages
        advantages = rewards.detach() - prev_critic_state_values.detach()


        # Critic output of old actions and old critic state values
        action_logSoftmax, critic_state_values, entropy = self.policy_main.critic_output(prev_states.view(-1, *prev_states_shape),
                                                                                         prev_actions.view(-1, prev_actions_shape))

        # Match dimensions between critic state values tensor with rewards tensor
        critic_state_values = torch.squeeze(critic_state_values)

        # Calculate Action Loss
        ratio = torch.exp(action_logSoftmax - prev_action_logSoftmaxes.detach())
        surrogate_loss1 = ratio * advantages
        surrogate_loss2 = torch.clamp(ratio, 1-self.gradient_clip, 1+self.gradient_clip) * advantages
        action_loss = -torch.min(surrogate_loss1, surrogate_loss2)

        # Calculate Value loss
        value_loss = 0.5 * self.mse(critic_state_values, rewards)

        # Calculate Entropy loss
        entropy_loss = -0.01 * entropy

        # Total loss
        loss = action_loss + value_loss + entropy_loss

        # Gradient step
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        # Copy the new weights into the old policy
        self.policy_old.load_state_dict(self.policy_main.state_dict())

        # Clear memory replay
        self.actions = []
        self.states = []
        self.action_logSoftmaxes = []
        self.rewards = []
        self.critic_state_values = []
        self.terminals = []

        return loss.mean(), value_loss, action_loss.mean(), entropy_loss.mean()


def custom_image_processor(image):
    state = np.array(array2d(make_surface(image)))

    state = state[:, :400]
    state = cv2.resize(state, (84, 84))
    state = state/255.

    if torch.cuda.is_available():
        return torch.tensor([state]).float().cuda()
    return torch.tensor([state]).float()


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

def train(model, start):

    # Initialize iteration, episode_length list
    it_ep_length_list = []

    # Initialize episode length and reward
    episode_length = 0

    image_data = env.reset()
    image_data = custom_image_processor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)


    for iteration in range(1, model.number_of_iterations):

        action = model.pick_action(state)
        image_data, reward, terminal, info = env.step(action)
        image_data = custom_image_processor(image_data)
        next_state = torch.cat((state.squeeze(0)[1:, :, :], image_data)).unsqueeze(0)

        # Remove during training
        # env.render()
        # time.sleep(1 / 60)  # FPS

        # Save to memory replay
        model.rewards.append(reward)
        model.terminals.append(terminal)

        if iteration % model.optimization_modulo == 0:
            total_loss, value_loss, action_loss, entropy_loss = model.optimize_custom()
            print("Total loss: ", total_loss)
            print("Value Loss: ", value_loss)
            print("Action Loss: ", action_loss)
            print("Entropy Loss: ", entropy_loss)
            print("")

        if iteration % model.save_modulo == 0:
            if not os.path.exists(model.save_folder):
                os.mkdir(model.save_folder)
            torch.save(model.policy_old.state_dict(), os.path.join(model.save_folder, str(iteration) + ".pth"))
            with open(os.path.join(model.save_folder, "output.csv"), "w", newline='') as f:
                csv_output = csv.writer(f)
                csv_output.writerow(["iteration", "episode_length"])
                csv_output.writerows(it_ep_length_list)
            print("Iteration: ", iteration)
            print("Elapsed Time: ", time.time() - start)

        if terminal is False:
            episode_length += 1
        else:
            it_ep_length_list.append([iteration, episode_length])
            print(iteration, episode_length)
            episode_length = 0

            image_data = env.reset()
            image_data = custom_image_processor(image_data)
            next_state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

        state = next_state
    env.close()

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)

if __name__ == "__main__":
    device = torch.device('cpu')
    cuda_is_available = torch.cuda.is_available()
    if cuda_is_available:
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()

    mode = "train"
    if mode == "train":
        env = flappy_bird_gym.make("FlappyBird-rgb-v0")
        model = PPO()
        model.policy_main.apply(init_weights)
        model.policy_old.apply(init_weights)
        start = time.time()
        train(model, start)