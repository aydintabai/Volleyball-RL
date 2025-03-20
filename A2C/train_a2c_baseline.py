import gym
import slimevolleygym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def decode_action(action_int):
    return [
        (action_int >> 0) & 1,
        (action_int >> 1) & 1,
        (action_int >> 2) & 1
    ]

class A2CNetwork(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64, output_dim=8):
        super(A2CNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value

    def get_action(self, state_np):
        state_t = torch.FloatTensor(state_np).unsqueeze(0).to(device)
        action_probs, _ = self.forward(state_t)
        dist = torch.distributions.Categorical(action_probs)
        action_int = dist.sample()
        log_prob = dist.log_prob(action_int)
        return action_int.item(), log_prob

def compute_advantage(rewards, gamma=0.99):
    returns, advantages = [], []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = returns - returns.mean()
    return advantages, returns

def train_vs_baseline(num_episodes=1000, gamma=0.99, lr=1e-3):
    env = gym.make("SlimeVolley-v0")
    policy = A2CNetwork().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    reward_history = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        log_probs, rewards, values = [], [], []
        episode_reward = 0
        timesteps = 0

        while not done:
            action_int, log_prob = policy.get_action(state)
            action_vec = decode_action(action_int)
            
            next_state, reward, done, info = env.step(action_vec)
            episode_reward += reward

            survival_reward = 0.01
            adjusted_reward = reward + survival_reward

            log_probs.append(log_prob)
            rewards.append(adjusted_reward)

            state = next_state
            timesteps += 1

        reward_history.append(episode_reward)

        advantages, returns = compute_advantage(rewards, gamma=gamma)

        entropy_bonus = -0.05 * sum(torch.exp(log_p) * log_p for log_p in log_probs)
        policy_loss = -sum(log_p * adv for log_p, adv in zip(log_probs, advantages)) + entropy_bonus

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if (episode + 1) % 100 == 0:
            avg_score_100 = np.mean(reward_history[-100:])
            print(f"Episode {episode+1}/{num_episodes}, Average Reward(100): {avg_score_100:.2f}")

    env.close()

    torch.save(policy.state_dict(), "a2c_baseline.pth")

    plt.figure()
    plt.plot(reward_history, label="Episode Reward")
    plt.plot(np.convolve(reward_history, np.ones(100)/100, mode='valid'), label="100-ep moving avg", linewidth=2)
    plt.title("A2C Training")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig("a2c_baseline_training_plot.png")
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_vs_baseline(num_episodes=1000, gamma=0.99, lr=1e-3)
