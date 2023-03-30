"""
This version uses one neural network for both the actor and the critic
"""
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game.flappy_bird import GameState


class PPO(nn.Module):

    def __init__(self):
        super(PPO, self).__init__()

        self.number_of_actions = 2
        self.number_of_iterations = 3000000
        self.optimization_modulo = 20
        self.save_modulo = 1600
        self.save_folder = "pm_ppo"

        # Optimization hyperparameters
        self.gamma = 0.99
        self.gradient_clip = 0.1
        self.value_loss_regularizer = 0.5
        self.entropy_loss_regularizer = 0.01
        self.max_gradient = 40

        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU(inplace=True)

        self.actor = nn.Linear(512, self.number_of_actions)
        self.critic = nn.Linear(512, 1)

        self.softmax = nn.Softmax()
        self.logSoftmax = nn.LogSoftmax()

        self.optimizer = optim.Adam(self.parameters(), lr=1e-5)

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
            'action_log_prob': torch.stack(action_log_prob).detach(),
            'value': torch.stack(value).detach(),
            'reward': torch.tensor(reward).detach(),
            'mask': torch.stack(mask).detach(),
        }
        state_shape = batch['state'].size()[2:]
        action_shape = batch['action'].size()[-1]

        # Compute returns
        returns = torch.zeros(self.optimization_modulo + 1, 1, 1)
        for i in reversed(range(self.optimization_modulo)):
            returns[i] = returns[i + 1] * self.gamma * batch['mask'][i] + batch['reward'][i]
        returns = returns[:-1]
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Process batch
        values, action_log_probs, dist_entropy = self.critic_output(batch['state'].view(-1, *state_shape),
                                                                        batch['action'].view(-1, action_shape))
        values = values.view(self.optimization_modulo, 1, 1)
        action_log_probs = action_log_probs.view(self.optimization_modulo, 1, 1)

        # Compute advantages
        advantages = returns.cuda() - values.detach().cuda()

        # Action loss
        ratio = torch.exp(action_log_probs - batch['action_log_prob'].detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.gradient_clip, 1 + self.gradient_clip) * advantages
        action_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = (returns.cuda() - values.cuda()).pow(2).mean()
        value_loss = self.value_loss_regularizer * value_loss

        # Total loss
        loss = value_loss + action_loss - dist_entropy * self.entropy_loss_regularizer

        # Optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), self.max_gradient)
        self.optimizer.step()

        return loss, value_loss * self.value_loss_regularizer, action_loss, - dist_entropy * self.entropy_loss_regularizer


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

    # Initialize replay memory
    replay_memory = []

    # Initial action is to do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, _, _, _ = game_state.frame_step(action)

    # Transform image to get initial state
    image_data = image_to_tensor(resize_and_bgr2gray(image_data))
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    for iteration in range(1, model.number_of_iterations):

        # Get next action given state
        critic_state_value, action, action_logSoftmax = model.actor_output(state)
        if action[0] == 0:
            action_for_game = [1, 0]
        else:
            action_for_game = [0, 1]

        # Execute action and get next state and reward
        image_data, reward, terminal, score = game_state.frame_step(action_for_game)

        image_data = image_to_tensor(resize_and_bgr2gray(image_data))
        next_state = torch.cat((state.squeeze(0)[1:, :, :], image_data)).unsqueeze(0)

        # Save transition to replay memory
        replay_memory.append((state, action, action_logSoftmax, critic_state_value, reward, torch.tensor([[float(terminal)]])))

        # Optimization
        if iteration % model.optimization_modulo == 0:
            total_loss, value_loss, action_loss, entropy_loss = model.optimize_custom(replay_memory)
            # model.optimize_custom(replay_memory)
            # Reset memory
            replay_memory = []

        # Update/Print & Reset episode length
        if terminal is False:
            episode_length += 1
        else:
            # TODO Save this value somewhere to generate a graph
            print(episode_length)
            episode_length = 0

        # Save model
        if iteration % model.save_modulo == 0:
            if not os.path.exists(model.save_folder):
                os.mkdir(model.save_folder)
            torch.save(model.state_dict(), os.path.join(model.save_folder, str(iteration) + ".pth"))
            print("Iteration: ", iteration)
            print("Elapsed Time: ", time.time() - start)
            # print("Total loss: ", total_loss)
            # print("Value Loss: ", value_loss)
            # print("Action Loss: ", action_loss)
            # print("Entropy Loss: ", entropy_loss)

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
