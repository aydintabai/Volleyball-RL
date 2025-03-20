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
        state_t = torch.FloatTensor(state_np).unsqueeze(0)
        action_probs, state_value = self.forward(state_t)
        dist = torch.distributions.Categorical(action_probs)
        action_int = dist.sample()
        log_prob = dist.log_prob(action_int)
        return action_int.item(), log_prob, state_value

def run_selfplay_episode(env, policyA, policyB):
    done = False
    obs1 = env.reset()
    obs2 = obs1

    log_probsA, valuesA, rewardsA = [], [], []
    log_probsB, valuesB, rewardsB = [], [], []
    total_score_B = 0.0

    while not done:
        actionA_int, log_probA, valueA = policyA.get_action(obs2)
        actionA_vec = decode_action(actionA_int)

        actionB_int, log_probB, valueB = policyB.get_action(obs1)
        actionB_vec = decode_action(actionB_int)

        obs1_next, rewardB, done, info = env.step(actionB_vec, actionA_vec)

        rewardA = -rewardB
        obs2_next = info["otherObs"]

        log_probsA.append(log_probA)
        valuesA.append(valueA)
        rewardsA.append(rewardA)

        log_probsB.append(log_probB)
        valuesB.append(valueB)
        rewardsB.append(rewardB)

        total_score_B += rewardB
        obs1, obs2 = obs1_next, obs2_next

    return log_probsA, valuesA, rewardsA, log_probsB, valuesB, rewardsB, total_score_B

def compute_advantage(rewards, values, gamma=0.99):
    returns, advantages = [], []
    G = 0.0
    for r, v in zip(reversed(rewards), reversed(values)):
        G = r + gamma * G
        returns.insert(0, G)
        advantages.insert(0, G - v.item())
    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    return advantages, returns

def train_selfplay(num_episodes=5000, gamma=0.99, lr=1e-3):
    env = gym.make("SlimeVolley-v0")
    
    policyA = A2CNetwork()
    policyB = A2CNetwork()

    optimizerA = optim.Adam(policyA.parameters(), lr=lr)
    optimizerB = optim.Adam(policyB.parameters(), lr=lr)

    all_scores_B = []

    for episode in range(num_episodes):
        (log_probsA, valuesA, rewardsA,
         log_probsB, valuesB, rewardsB,
         ep_score_B) = run_selfplay_episode(env, policyA, policyB)

        all_scores_B.append(ep_score_B)

        advantagesA, returnsA = compute_advantage(rewardsA, valuesA, gamma=gamma)
        advantagesB, returnsB = compute_advantage(rewardsB, valuesB, gamma=gamma)

        policy_lossA = -sum(log_pA * adv for log_pA, adv in zip(log_probsA, advantagesA))
        value_lossA = torch.nn.functional.mse_loss(torch.cat(valuesA).squeeze(), returnsA)

        policy_lossB = -sum(log_pB * adv for log_pB, adv in zip(log_probsB, advantagesB))
        value_lossB = torch.nn.functional.mse_loss(torch.cat(valuesB).squeeze(), returnsB)

        optimizerA.zero_grad()
        (policy_lossA + value_lossA).backward()
        optimizerA.step()

        optimizerB.zero_grad()
        (policy_lossB + value_lossB).backward()
        optimizerB.step()

        if (episode + 1) % 100 == 0:
            avg_score_100 = np.mean(all_scores_B[-100:])
            print(f"Episode {episode+1}/{num_episodes}, Average Reward(100): {avg_score_100:.2f}")
    
    env.close()

    torch.save(policyB.state_dict(), "selfplay_a2c.pth")

    plt.figure()
    plt.plot(all_scores_B, label="Episode Reward")
    plt.plot(np.convolve(all_scores_B, np.ones(100)/100, mode='valid'), label="100-ep moving avg", linewidth=2)
    plt.title("A2C Self-Play Training")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig("selfplay_a2c_training_plot.png")
    plt.show()

if __name__ == "__main__":
    train_selfplay(num_episodes=5000, gamma=0.99, lr=1e-3)