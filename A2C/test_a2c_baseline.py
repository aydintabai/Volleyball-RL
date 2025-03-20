import gym
import slimevolleygym
import torch
import torch.nn as nn
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_size = 12
action_size = 8
hidden_size = 64

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
        action_int = torch.argmax(action_probs).cpu().numpy()
        return action_int

model = A2CNetwork(state_size, hidden_size, action_size).to(device)
model.load_state_dict(torch.load("a2c_baseline.pth", map_location=device))
model.eval()

def decode_action(action_int):
    return [
        (action_int >> 0) & 1,
        (action_int >> 1) & 1,
        (action_int >> 2) & 1
    ]

env = gym.make("SlimeVolley-v0")
test_episodes = 10

total_rewards = []

for episode in range(test_episodes):
    state = env.reset()
    done = False
    episode_reward = 0

    print(f"Episode {episode + 1}")

    while not done:
        env.render()
        action_int = model.get_action(state)
        action = decode_action(action_int)
        state, reward, done, info = env.step(action)
        episode_reward += reward

        time.sleep(0.02)

    total_rewards.append(episode_reward)
    print(f"Episode {episode + 1} Reward: {episode_reward}")

env.close()

average_reward = np.mean(total_rewards)
print(f"Average reward over {test_episodes} episodes: {average_reward:.2f}")
