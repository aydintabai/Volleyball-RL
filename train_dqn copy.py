import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
import slimevolleygym

from dataclasses import dataclass

@dataclass
class HyperParams:
    BATCH_SIZE: int = 512
    GAMMA: float = 0.99
    EPS_START: float = 0.9
    EPS_END: float = 0.05
    EPS_DECAY: int = 5000  # Increased decay for more exploration
    TAU: float = 0.005
    LR: float = 1e-4
    MEMORY_SIZE: int = 50000  # Increased memory size for more diverse experiences


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# set up interactive matplotlib
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

class SlimeVolleyWrapper:
    """
    Wraps SlimeVolley so it can be used with the DQN implementation.
    The original SlimeVolley has a continuous action space, but we'll discretize it.
    """

    def __init__(self, env_name="SlimeVolley-v0"):
        """
        Initialize the SlimeVolley environment.
        
        Args:
            env_name (str): The name of the SlimeVolley environment to use.
        """
        self.env = gym.make(env_name)
        
        # Define a discrete action space for DQN
        # SlimeVolley has 3 binary actions: LEFT, RIGHT, JUMP
        # This results in 2^3 = 8 possible action combinations
        self.action_space_n = 8
        
        # Map from discrete action index to continuous action space
        self.action_map = {
            0: [0, 0, 0],  # NOOP
            1: [1, 0, 0],  # LEFT
            2: [0, 1, 0],  # RIGHT
            3: [0, 0, 1],  # JUMP
            4: [1, 0, 1],  # LEFT + JUMP
            5: [0, 1, 1],  # RIGHT + JUMP
            6: [1, 1, 0],  # LEFT + RIGHT (generally not useful)
            7: [1, 1, 1],  # LEFT + RIGHT + JUMP (generally not useful)
        }
        
        # Get observation space size
        self.obs_size = self.env.observation_space.shape[0]

    def reset(self):
        """
        Reset the environment and return the initial observation.
        
        Returns:
            np.ndarray: The initial observation.
            dict: Empty dictionary for compatibility with Gym API.
        """
        observation = self.env.reset()
        return observation, {}

    def step(self, action_index):
        """
        Take a step in the environment with the given action.
        
        Args:
            action_index (int): The index of the action to take.
            
        Returns:
            tuple: (observation, reward, done, truncated, info)
        """
        action = self.action_map[action_index]
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, False, info


class ReplayMemory:
    """
    Replay memory to store transitions.
    """

    def __init__(self, capacity: int):
        """Initialize the replay memory.

        Args:
            capacity (int): The maximum number of transitions to store.
        """
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """
        Sample a batch of transitions.

        Args:
            batch_size: The number of transitions to sample.

        Returns:
            list: A list of sampled transitions.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        """
        Initializes the DQN model.

        Args:
            n_observations (int): The size of the input observation space.
            n_actions (int): The number of possible actions.
        """
        super(DQN, self).__init__()
        # Slimevolley has more complex observations, so we use a larger network
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, n_actions)

    def forward(self, x):
        """
        Forward pass of the DQN model.

        Args:
            x (torch.Tensor): Input tensor representing the state.

        Returns:
            torch.Tensor: Output tensor representing Q-values for each action.
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


class DQNTrainer:
    def __init__(
        self,
        env: SlimeVolleyWrapper,
        memory: ReplayMemory,
        device: torch.device,
        params: HyperParams,
        max_steps_per_episode: int = 1000,  # Increased for longer episodes
        num_episodes: int = 1000,  # Increased for more training
    ) -> None:
        """
        Initializes the DQNTrainer with the required components to train a DQN agent.
        """
        self.env = env
        self.policy_net = DQN(env.obs_size, env.action_space_n).to(device)
        self.target_net = DQN(env.obs_size, env.action_space_n).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=params.LR, amsgrad=True)
        self.memory = memory
        self.device = device
        self.params = params
        self.max_steps_per_episode = max_steps_per_episode
        self.num_episodes = num_episodes

        # Track rewards per episode
        self.episode_rewards = []
        self.avg_rewards = []  # For tracking average rewards
        self.steps_done = 0
        
        # For evaluation
        self.evaluation_rewards = []
        self.eval_episodes = 100
        self.eval_interval = 50  # Evaluate every 50 episodes

    def select_action(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Selects an action using an epsilon-greedy policy based on current Q-network.
        """
        # Compute epsilon threshold
        sample = random.random()
        eps_threshold = self.params.EPS_END + (self.params.EPS_START - self.params.EPS_END) * \
        math.exp(-1.0 * self.steps_done / self.params.EPS_DECAY)

        # Update steps
        self.steps_done += 1

        # Exploit or explore
        if sample > eps_threshold:
            with torch.no_grad():
                # Choose best action from Q-network
                return self.policy_net(state_tensor).max(1).indices.view(1, 1)
        else:
            # Choose random action
            return torch.tensor(
                [[random.randrange(self.env.action_space_n)]],
                device=self.device,
                dtype=torch.long,
            )

    def optimize_model(self) -> None:
        """
        Performs one gradient descent update on the policy network using a random minibatch sampled from replay memory.
        """
        if len(self.memory) < self.params.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.params.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device, dtype=torch.bool
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.params.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.params.GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def soft_update(self) -> None:
        """
        Performs a soft update of the target network parameters.
        """
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.params.TAU + target_net_state_dict[key] * (1 - self.params.TAU)
        
        self.target_net.load_state_dict(target_net_state_dict)

    def evaluate(self) -> float:
        """
        Evaluates the current policy by running several episodes.
        
        Returns:
            float: Average reward across evaluation episodes.
        """
        total_reward = 0.0
        for _ in range(self.eval_episodes):
            obs, _ = self.env.reset()
            state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            episode_reward = 0.0
            done = False
            
            while not done:
                with torch.no_grad():
                    action = self.policy_net(state).max(1).indices.view(1, 1)
                next_obs, reward, done, _, _ = self.env.step(action.item())
                episode_reward += reward
                
                if done:
                    break
                    
                state = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                
            total_reward += episode_reward
            
        avg_reward = total_reward / self.eval_episodes
        self.evaluation_rewards.append(avg_reward)
        return avg_reward

    def plot_rewards(self, show_result: bool = False) -> None:
        """
        Plots accumulated rewards for each episode.
        """
        plt.figure(1)
        rewards_t = torch.tensor(self.episode_rewards, dtype=torch.float)
        
        # Calculate moving average
        if len(rewards_t) >= 100:
            means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            self.avg_rewards = means.numpy()
        
        # Decide whether to clear figure or show final result
        if show_result:
            plt.title("Result")
        else:
            plt.clf()
            plt.title("Training (Reward)")

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.plot(rewards_t.numpy(), alpha=0.6, label="Episode Reward")
        
        if len(self.avg_rewards) > 0:
            plt.plot(self.avg_rewards, label="100-episode Average")
            
        if len(self.evaluation_rewards) > 0:
            eval_x = np.arange(0, len(self.episode_rewards), self.eval_interval)[:len(self.evaluation_rewards)]
            plt.plot(eval_x, self.evaluation_rewards, 'r-', label="Evaluation Reward")
        
        plt.legend()
        plt.pause(0.001)
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def train(self) -> None:
        """
        Runs the main training loop across the specified number of episodes.
        """
        best_eval_reward = float('-inf')
        best_model_path = "best_slimevolley_dqn.pt"
        
        for i_episode in range(self.num_episodes):
            # Reset the environment and initialize state and episode_reward
            obs, _ = self.env.reset()
            state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            episode_reward = 0.0

            for t in range(self.max_steps_per_episode):
                # Select an action
                action = self.select_action(state)
                # Execute the action in the environment
                next_obs, reward, done, _, _ = self.env.step(action.item())
                # Convert observations to tensor
                next_state = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0) if not done else None
                # Save the transition in replay memory
                self.memory.push(state, action, next_state, torch.tensor([reward], device=self.device))
                # Advance state to next_state
                state = next_state if next_state is not None else torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                # Run optimization step
                self.optimize_model()
                # Perform soft update
                self.soft_update()
                # Accumulate the reward for the episode
                episode_reward += reward
                # Break the loop when a terminal state is reached
                if done:
                    break

            # Tracking episode reward and plotting rewards
            self.episode_rewards.append(episode_reward)
            
            # Print episode info
            print(f"Episode {i_episode}: Reward = {episode_reward:.2f}, Epsilon = {self.params.EPS_END + (self.params.EPS_START - self.params.EPS_END) * math.exp(-1.0 * self.steps_done / self.params.EPS_DECAY):.4f}")
            
            # Evaluate periodically
            if i_episode % self.eval_interval == 0:
                avg_eval_reward = self.evaluate()
                print(f"Evaluation after episode {i_episode}: Average Reward = {avg_eval_reward:.2f}")
                
                # Save the best model
                if avg_eval_reward > best_eval_reward:
                    best_eval_reward = avg_eval_reward
                    torch.save(self.policy_net.state_dict(), best_model_path)
                    print(f"New best model saved with reward {best_eval_reward:.2f}")
            
            # Update the rewards plot
            self.plot_rewards()

        print("Training complete")
        print(f"Best evaluation reward: {best_eval_reward:.2f}")
        self.plot_rewards(show_result=True)
        plt.ioff()
        plt.show()
        plt.savefig("rewards_plot_slimevolley_dqn.png")
        
        # Save final model
        torch.save(self.policy_net.state_dict(), "final_slimevolley_dqn.pt")
        print("Final model saved")


def main():
    # Set up the environment and parameters
    env = SlimeVolleyWrapper(env_name="SlimeVolley-v0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create hyperparameters
    params = HyperParams(
        BATCH_SIZE=512,
        GAMMA=0.99,
        EPS_START=0.9,
        EPS_END=0.05,
        EPS_DECAY=5000,
        TAU=0.005,
        LR=1e-4,
        MEMORY_SIZE=50000
    )
    
    # Create replay memory
    memory = ReplayMemory(params.MEMORY_SIZE)
    
    # Create trainer
    trainer = DQNTrainer(
        env=env,
        memory=memory,
        device=device,
        params=params,
        max_steps_per_episode=1000,
        num_episodes=1000
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()