import csv
import os
import time

import torch
from torch import nn, optim


from game.wrapper import Game

class PPO(nn.Module):

    def __init__(self):

        super(PPO, self).__init__()

        self.number_of_actions = 2
        self.number_of_iterations = 3000000
        self.optimization_modulo = 20
        self.save_modulo = 100000
        self.save_folder = "pm_ppo_final_version"

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
        self.memory_replay = MemoryReplay()

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
        critic_state_value, action = self.forward(state)

        action_softmax = self.softmax(action)
        action_logSoftmax = self.logSoftmax(action)

        action_logSoftmax = action_logSoftmax.gather(1, actions)
        dist_entropy = -(action_logSoftmax * action_softmax).sum(-1).mean()
        return critic_state_value, action_logSoftmax, dist_entropy


    def optimize_custom(self):

        prev_states = torch.stack(self.memory_replay.states).detach()
        prev_actions = torch.stack(self.memory_replay.actions).detach()
        prev_action_logSoftmaxes = torch.stack(self.memory_replay.action_logSoftmaxes).detach()
        prev_crtitic_state_values = torch.stack(self.memory_replay.crtitic_state_values).detach()
        prev_rewards = torch.tensor(self.memory_replay.rewards).detach()
        prev_terminals = torch.stack(self.memory_replay.terminals).detach()

        states_shape = prev_states.size()[2:]
        actions_shape = prev_actions.size()[-1]

        returns = torch.zeros(self.optimization_modulo + 1, 1, 1)
        for i in reversed(range(self.optimization_modulo)):
            returns[i] = returns[i + 1] * self.gamma * prev_terminals[i] + prev_rewards[i]
        returns = returns[:-1]
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        critic_state_value, action_logSoftmax, entropy = self.critic_output(prev_states.view(-1, *states_shape),
                                                                    prev_actions.view(-1, actions_shape))

        critic_state_value = critic_state_value.view(self.optimization_modulo, 1, 1)
        action_logSoftmax = action_logSoftmax.view(self.optimization_modulo, 1, 1)

        # Compute advantages
        advantages = returns.cuda() - critic_state_value.detach().cuda()

        # Action loss
        ratio = torch.exp(action_logSoftmax - prev_action_logSoftmaxes.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.gradient_clip, 1 + self.gradient_clip) * advantages
        action_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = (returns.cuda() - critic_state_value.cuda()).pow(2).mean()
        value_loss = self.value_loss_regularizer * value_loss

        # Entropy loss
        entropy_loss = - entropy * self.entropy_loss_regularizer

        # Total loss
        loss = value_loss + action_loss + entropy_loss

        # Optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), self.max_gradient)
        self.optimizer.step()

        return loss, value_loss, action_loss, entropy_loss


class MemoryReplay:

    def __init__(self):

        self.states = []
        self.actions = []
        self.action_logSoftmaxes = []
        self.crtitic_state_values = []
        self.rewards = []
        self.terminals = []

    def reset(self):
        self.states = []
        self.actions = []
        self.action_logSoftmaxes = []
        self.crtitic_state_values = []
        self.rewards = []
        self.terminals = []



def init_weights(model):
    if type(model) == nn.Conv2d or type(model) == nn.Linear:
        torch.nn.init.uniform(model.weight, -0.01, 0.01)
        model.bias.data.fill_(0.01)


def train(model, start):
    # instantiate game
    game_state = Game(84)

    # Initialize episode length
    episode_length = 0

    # Initialize iteration, episode_length list
    it_ep_length_list = []

    # Initial action is to do nothing
    image_data, reward, terminal = game_state.step(False)
    if torch.cuda.is_available():
        image_data = image_data.cuda()
    state = torch.stack([torch.cat([image_data, image_data, image_data, image_data])])
    for iteration in range(1, model.number_of_iterations):
        critic_state_value, action, action_logSoftmax = model.actor_output(state)

        # Execute action and get next state and reward
        image_data, reward, terminal = game_state.step(action)
        if torch.cuda.is_available():
            image_data = image_data.cuda()
        next_state = torch.stack([torch.cat([state[0][1:], image_data])])

        # Save transition to replay memory
        model.memory_replay.states.append(state.data)
        model.memory_replay.actions.append(action.data)
        model.memory_replay.action_logSoftmaxes.append(action_logSoftmax.data)
        model.memory_replay.crtitic_state_values.append(critic_state_value.data)
        model.memory_replay.rewards.append([reward])
        model.memory_replay.terminals.append(torch.tensor([[float(terminal)]]))

        if iteration % model.optimization_modulo == 0:
            total_loss, value_loss, action_loss, entropy_loss = model.optimize_custom()
            print("Total loss: ", total_loss)
            print("Value Loss: ", value_loss)
            print("Action Loss: ", action_loss)
            print("Entropy Loss: ", entropy_loss)
            print("")
            # Reset memory
            model.memory_replay.reset()

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
            torch.save(model.state_dict(), os.path.join(model.save_folder, str(iteration) + ".pth"))
            with open(os.path.join(model.save_folder, "output.csv"), "w", newline='') as f:
                csv_output = csv.writer(f)
                csv_output.writerow(["iteration", "episode_length"])
                csv_output.writerows(it_ep_length_list)
            print("Iteration: ", iteration)
            print("Elapsed Time: ", time.time() - start)

        state = next_state


if __name__ == "__main__":
    ppo_model = PPO()
    if torch.cuda.is_available():
        ppo_model = ppo_model.cuda()
    ppo_model.apply(init_weights)

    time_start = time.time()
    train(ppo_model, time_start)