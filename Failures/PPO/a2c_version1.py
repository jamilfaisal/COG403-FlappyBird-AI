import csv
import os
import time

import cv2
import numpy as np
import torch
from pygame.surfarray import array2d, make_surface
from torch import nn, optim

from game.flappy_bird import GameState


class A2CPolicy(nn.Module):

    def __init__(self):
        super(A2CPolicy, self).__init__()

        self.conv1 = nn.Conv2d(4, 16, 8, 4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(2592, 256)
        self.relu3 = nn.ReLU()

        self.actor = nn.Linear(256, 2)
        self.critic = nn.Linear(256, 1)

        self.softmax = nn.Softmax()
        self.logSoftmax = nn.LogSoftmax()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.relu2(out)

        out = out.view(out.size()[0], -1)

        out = self.fc3(out)
        out = self.relu3(out)

        action = self.actor(out)
        critic_state_value = self.critic(out)

        return critic_state_value, action

    def actor_output(self, state):
        critic_state_value, action = self.forward(state)

        action_softmax = self.softmax(action)
        action_logSoftmax = self.logSoftmax(action)

        # Pick action stochastically
        action = action_softmax.multinomial(1)

        action_logSoftmax = action_logSoftmax.gather(1, action)
        return critic_state_value, action, action_logSoftmax

    def critic_output(self, state, actions):
        # Forward pass
        critic_state_value, action = self.forward(state)

        action_softmax = self.softmax(action)
        action_logSoftmax = self.logSoftmax(action)

        # Evaluate actions
        action_logSoftmax = action_logSoftmax.gather(1, actions)
        dist_entropy = -(action_logSoftmax * action_softmax).sum(-1).mean()
        return critic_state_value, action_logSoftmax, dist_entropy


class A2C:

    def __init__(self):

        self.number_of_actions = 2

        self.max_gradient = 40
        self.value_loss_regularizer = 0.5
        self.entropy_loss_regularizer = 0.01
        self.gamma = 0.99
        self.optimization_modulo = 20

        self.number_of_iterations = 3000000
        self.save_modulo = 100000
        self.save_folder = "pm_a2c_version1"


        self.network = A2CPolicy()
        self.network.apply(init_weights)

        if torch.cuda.is_available():
            self.network = self.network.cuda()

        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-4)

    def optimize_custom(self, replay_memory):
        state = []
        action = []
        action_log_prob = []
        value = []
        reward = []
        mask = []
        for memory in replay_memory:
            state.append(memory[0])
            action.append(memory[1])
            action_log_prob.append(memory[2])
            value.append(memory[3])
            reward.append([[memory[4]]])
            mask.append(memory[5])

        batch = {
            'state': torch.stack(state).detach(),
            'action': torch.stack(action).detach(),
            'reward': torch.tensor(reward).detach(),
            'mask': torch.stack(mask).detach(),
        }
        state_shape = batch['state'].size()[2:]
        action_shape = batch['action'].size()[-1]

        next_critic_state_value, next_action = self.network(batch["state"][-1])

        # Compute returns
        returns = torch.zeros(self.optimization_modulo + 1, 1, 1)
        returns[-1] = next_critic_state_value
        for i in reversed(range(self.optimization_modulo)):
            returns[i] = returns[i + 1] * self.gamma * batch['mask'][i] + batch['reward'][i]
        returns = returns[:-1]

        # Process batch
        values, action_log_probs, dist_entropy = self.network.critic_output(batch['state'].view(-1, *state_shape),
                                                                        batch['action'].view(-1, action_shape))
        values = values.view(self.optimization_modulo, 1, 1)
        action_log_probs = action_log_probs.view(self.optimization_modulo, 1, 1)

        # Compute advantages
        advantages = returns.cuda() - values.detach().cuda()
        value_loss = advantages.pow(2).mean() * self.value_loss_regularizer
        action_loss = (-advantages * action_log_probs).mean()
        entropy_loss = - dist_entropy * self.entropy_loss_regularizer

        loss = value_loss + action_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.network.parameters(), self.max_gradient)
        self.optimizer.step()

        return loss, value_loss, action_loss, entropy_loss


def custom_image_processor(image):
    state = np.array(array2d(make_surface(image)), dtype='uint8')

    state = state[:, :400]
    state = cv2.resize(state, (84, 84))
    state = state/255.
    if torch.cuda.is_available():
        return torch.tensor([state]).float().cuda()
    return torch.tensor([state]).float()


def train(model, start):
    # instantiate game
    game_state = GameState()

    # Initialize episode length
    episode_length = 0

    # Initialize iteration, episode_length list
    it_ep_length_list = []

    # Initialize replay memory
    replay_memory = []

    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, _, _, _ = game_state.frame_step(action)

    # Transform image to get initial state
    image_data = custom_image_processor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    for iteration in range(1, model.number_of_iterations):
        # Get next action given state
        critic_state_value, action, action_logSoftmax = model.network.actor_output(state)
        if action[0] == 0:
            action_for_game = [1, 0]
        else:
            action_for_game = [0, 1]

        # Execute action and get next state and reward
        image_data, reward, terminal, score = game_state.frame_step(action_for_game)

        image_data = custom_image_processor(image_data)
        next_state = torch.cat((state.squeeze(0)[1:, :, :], image_data)).unsqueeze(0)

        # Save transition to replay memory
        replay_memory.append((state, action, action_logSoftmax, critic_state_value, reward, torch.tensor([[float(terminal)]])))

        # Optimization
        if iteration % model.optimization_modulo == 0:
            total_loss, value_loss, action_loss, entropy_loss = model.optimize_custom(replay_memory)
            print("Total loss: ", total_loss)
            print("Value Loss: ", value_loss)
            print("Action Loss: ", action_loss)
            print("Entropy Loss: ", entropy_loss)
            print("")
            # Reset memory
            replay_memory = []

        # Update/Print & Reset episode length
        if terminal is False:
            episode_length += 1
        else:
            it_ep_length_list.append([iteration, episode_length])
            episode_length = 0

        # Save model
        if iteration % model.save_modulo == 0:
            if not os.path.exists(model.save_folder):
                os.mkdir(model.save_folder)
            torch.save(model.network.state_dict(), os.path.join(model.save_folder, str(iteration) + ".pth"))
            with open(os.path.join(model.save_folder, "output.csv"), "w", newline='') as f:
                csv_output = csv.writer(f)
                csv_output.writerow(["iteration", "episode_length"])
                csv_output.writerows(it_ep_length_list)
            print("Iteration: ", iteration)
            print("Elapsed Time: ", time.time() - start)

        # Set current state as the next state
        state = next_state


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)

if __name__ == "__main__":
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()

    a2c_model = A2C()

    start_time = time.time()
    train(a2c_model, start_time)

