"""
Cross-Entropy Method (CEM) Implementation for a Volleyball Agent
"""

import numpy as np
import torch
from typing import List, Dict


## Used a notebook on colab to call these functions to test model  
def get_device():
    #to make sure i'm using apple silicon on my laptop
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class CEMAgent:

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        population_size: int = 500,
        elite_ratio: float = 0.2,
        noise_std: float = 0.3,
        learning_rate: float = 0.02,
        device: str = None
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.noise_std = noise_std
        self.learning_rate = learning_rate

        self.device = torch.device(device) if device else get_device()
        print(f"Using device: {self.device}")

        self.param_dim = state_dim * action_dim
        self.elite_size = int(population_size * elite_ratio)

        self.mean = torch.zeros(self.param_dim, device=self.device)
        self.std = torch.ones(self.param_dim, device=self.device)

        self.population = torch.randn(population_size, self.param_dim, device=self.device)

        self.best_mean = None
        self.best_reward = float('-inf')

    def select_action(self, state) -> np.ndarray:
        
        #Select an action by sampling around the current mean parameters.
        
        if isinstance(state, (int, float)):
            state = np.array([state], dtype=np.float32)
        elif isinstance(state, list):
            state = np.array(state, dtype=np.float32)
        elif isinstance(state, np.ndarray) and state.dtype != np.float32:
            state = state.astype(np.float32)
        elif not isinstance(state, (torch.Tensor, np.ndarray)):
            raise TypeError(f"Unsupported state type: {type(state)}")

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        else:
            state = state.to(self.device)

        if state.dim() == 0:
            state = state.unsqueeze(0)

        params = self.mean.reshape(self.state_dim, self.action_dim)

        action = torch.matmul(state, params).squeeze()

        noise = torch.randn_like(action) * self.noise_std
        action = action + noise

        action = torch.clamp(action, -1, 1)
        return action.cpu().numpy()

    def update(
        self,
        states: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        elite_frac: float = None
    ) -> dict:
        """
        Update the parameter distribution using CEM.
        """
        elite_ratio = elite_frac if elite_frac is not None else self.elite_ratio
        elite_size = int(self.population_size * elite_ratio)

        if isinstance(states, list):
            processed_states = []
            for s in states:
                if isinstance(s, (int, float)):
                    s = np.array([s], dtype=np.float32)
                elif isinstance(s, np.ndarray) and s.dtype != np.float32:
                    s = s.astype(np.float32)
                processed_states.append(s)
            states = np.array(processed_states)
        elif isinstance(states, np.ndarray) and states.dtype != np.float32:
            states = states.astype(np.float32)
        else:
            raise TypeError(f"Unsupported states type: {type(states)}")

        if isinstance(actions, list):
            actions = np.array(actions, dtype=np.float32)
        if isinstance(rewards, list):
            rewards = np.array(rewards, dtype=np.float32)

        states = torch.from_numpy(states).float().to(self.device)
        actions = (torch.from_numpy(actions).float().to(self.device)
                   if isinstance(actions, np.ndarray)
                   else actions.to(self.device))
        rewards = (torch.from_numpy(rewards).float().to(self.device)
                   if isinstance(rewards, np.ndarray)
                   else rewards.to(self.device))

        if states.dim() == 1:
            states = states.unsqueeze(0)

        original_rewards_mean = torch.mean(rewards).item()
        original_rewards_std = torch.std(rewards).item()

        if rewards.std() > 1e-8:
            normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            normalized_rewards = rewards - rewards.mean()

        population_rewards = []
        for i in range(self.population_size):
            params = self.mean + self.std * self.population[i]
            params = params.reshape(self.state_dim, self.action_dim)

            predicted_actions = torch.matmul(states, params)
            action_loss = torch.mean((predicted_actions - actions)**2)

            weighted_loss = action_loss * normalized_rewards.mean()
            population_rewards.append(-weighted_loss.item())

        all_rewards = torch.tensor(population_rewards, device=self.device)
        elite_indices = all_rewards.argsort()[-elite_size:]
        elite_population = self.population[elite_indices]

        self.mean += self.learning_rate * (elite_population.mean(dim=0) - self.mean)
        self.std += self.learning_rate * (elite_population.std(dim=0) - self.std)
        self.std = torch.clamp(self.std, min=0.01)

        if original_rewards_mean > self.best_reward:
            self.best_reward = original_rewards_mean
            self.best_mean = self.mean.clone()

        self.population = torch.randn(self.population_size, self.param_dim, device=self.device)

        return {
            "mean_reward": original_rewards_mean,
            "mean_reward_std": original_rewards_std,
            "elite_reward": np.mean([population_rewards[i] for i in elite_indices.cpu().numpy()])
        }

    def save(self, path: str):

        torch.save({
            'mean': self.mean,
            'std': self.std,
            'best_mean': self.best_mean if self.best_mean is not None else self.mean,
            'best_reward': self.best_reward,
            'device': self.device,
            'hyperparameters': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'population_size': self.population_size,
                'elite_ratio': self.elite_ratio,
                'noise_std': self.noise_std,
                'learning_rate': self.learning_rate
            }
        }, path)

    def load(self, path: str):

        checkpoint = torch.load(path, map_location=self.device)
        self.mean = checkpoint['mean']
        self.std = checkpoint['std']
        self.best_mean = checkpoint.get('best_mean', None)
        self.best_reward = checkpoint.get('best_reward', float('-inf'))

        if 'hyperparameters' in checkpoint:
            hyperparams = checkpoint['hyperparameters']
            self.state_dim = hyperparams.get('state_dim', self.state_dim)
            self.action_dim = hyperparams.get('action_dim', self.action_dim)
            self.population_size = hyperparams.get('population_size', self.population_size)
            self.elite_ratio = hyperparams.get('elite_ratio', self.elite_ratio)
            self.noise_std = hyperparams.get('noise_std', self.noise_std)
            self.learning_rate = hyperparams.get('learning_rate', self.learning_rate)

        self.param_dim = self.state_dim * self.action_dim
        self.elite_size = int(self.population_size * self.elite_ratio)
        self.population = torch.randn(self.population_size, self.param_dim, device=self.device)
