#training loop for the CEM agent in this file.
#to be imported into notebook on colab for training
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import os
from typing import List, Tuple, Dict
import copy
import warnings


#VS BASELINE
def train_cem(
    env,
    agent,
    num_episodes: int,
    device=None,
    batch_size: int = 20,
    elite_frac: float = 0.2,
    eval_interval: int = 20,
    save_interval: int = 50,
    save_dir: str = "saved_models",
    render: bool = False
) -> Dict[str, List[float]]:

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "cem_volleyball.pt")
    best_model_path = os.path.join(save_dir, "cem_volleyball_best.pt")

    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")

    agent.device = device
    agent.mean = agent.mean.to(device)
    agent.std = agent.std.to(device)
    agent.population = agent.population.to(device)

    metrics = {
        "episode_rewards": [],
        "mean_rewards": [],
        "std_rewards": [],
        "elite_rewards": [],
        "best_reward": float("-inf"),
        "training_duration": 0
    }

    start_time = time.time()

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    rewards_line, = ax.plot([], [], 'b-', alpha=0.3, label='Episode Reward')
    mean_line, = ax.plot([], [], 'r-', linewidth=2, label='Mean Reward (10 episodes)')
    std_upper, = ax.plot([], [], 'g:', alpha=0.5)
    std_lower, = ax.plot([], [], 'g:', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Training Progress')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    pbar = tqdm(total=num_episodes, desc="Training")

    best_mean_reward = float('-inf')
    best_params = None

    try:
        for batch_num in range(num_episodes):
            batch_states, batch_actions, batch_rewards = [], [], []
            batch_episode_rewards = []

            for _ in range(batch_size):
                states, actions, rewards = [], [], []
                episode_reward = 0
                state = env.reset()
                done = False

                prev_ball_x = None
                prev_ball_y = None
                hit_ball = False
                crossed_net = False

                while not done:
                    if isinstance(state, (int, float)):
                        state = np.array([state], dtype=np.float32)
                    elif not isinstance(state, np.ndarray):
                        state = np.array(state, dtype=np.float32)

                    action = agent.select_action(state)
                    next_state, reward, done, info = env.step(action)

                    shaped_reward = reward
                    if len(state) >= 12:
                        agent_x = state[0]
                        agent_y = state[1]
                        ball_x = state[4]
                        ball_y = state[5]
                        ball_vx = state[6]
                        ball_vy = state[7]
                        #adding reward shaping 
                        #reward being near the ball horizontally
                        if abs(agent_x - ball_x) < 2.0:
                            shaped_reward += 0.01

                        #reward for hitting the ball upward
                        if prev_ball_y is not None:
                            if (ball_y > prev_ball_y and ball_vy > 0
                                    and abs(agent_x - ball_x) < 2.0):
                                shaped_reward += 0.3
                                hit_ball = True

                        #reward for getting ball across the net
                        if prev_ball_x is not None:
                            if prev_ball_x < 0 < ball_x:
                                shaped_reward += 1.0
                                crossed_net = True

                        #small reward for moving toward the ball
                        if prev_ball_x is not None and ball_x < 0 and ball_vx < 0:
                            if ((agent_x < ball_x and action[0] > 0)
                                    or (agent_x > ball_x and action[1] > 0)):
                                shaped_reward += 0.05

                        #small reward for jumping when the ball is above
                        if ball_y > agent_y and action[2] > 0:
                            shaped_reward += 0.05

                        prev_ball_x = ball_x
                        prev_ball_y = ball_y

                    if reward != 0:
                        if reward > 0:
                            shaped_reward = reward
                        else:
                            if not hit_ball:
                                shaped_reward = reward * 1.5
                            elif not crossed_net:
                                shaped_reward = reward * 1.2

                        hit_ball = False
                        crossed_net = False

                    states.append(state)
                    actions.append(action)
                    rewards.append(shaped_reward)
                    episode_reward += reward

                    state = next_state

                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_rewards.extend(rewards)
                batch_episode_rewards.append(episode_reward)

            batch_actions_np = np.array(batch_actions, dtype=np.float32)

            update_info = agent.update(batch_states, batch_actions_np, batch_rewards, elite_frac)

            metrics['episode_rewards'].extend(batch_episode_rewards)

            start_idx = max(0, len(metrics['episode_rewards']) - 10)
            recent_rewards = metrics['episode_rewards'][start_idx:]
            current_mean = np.mean(recent_rewards)
            current_std = np.std(recent_rewards)

            metrics['mean_rewards'].append(current_mean)
            metrics['std_rewards'].append(current_std)
            metrics['elite_rewards'].append(update_info.get('elite_reward', 0))

            if current_mean > best_mean_reward:
                best_mean_reward = current_mean
                best_params = agent.mean.clone()
                torch.save({
                    'mean': best_params,
                    'std': agent.std,
                    'reward': best_mean_reward
                }, best_model_path)
                print(f"\nNew best model saved with mean reward: {best_mean_reward:.2f}")

            metrics['best_reward'] = max(metrics['best_reward'], max(batch_episode_rewards))

            batch_mean = np.mean(batch_episode_rewards)
            batch_std = np.std(batch_episode_rewards)
            pbar.set_postfix({
                'mean_reward': f"{batch_mean:.2f}",
                'std': f"{batch_std:.2f}",
                'best': f"{metrics['best_reward']:.2f}"
            })
            pbar.update(1)

            episodes_x = list(range(len(metrics['episode_rewards'])))
            rewards_line.set_data(episodes_x, metrics['episode_rewards'])

            mean_episodes = list(range(0, len(metrics['episode_rewards']), batch_size))
            if len(mean_episodes) < len(metrics['mean_rewards']):
                mean_episodes.append(len(metrics['episode_rewards']) - 1)
            mean_line.set_data(mean_episodes[:len(metrics['mean_rewards'])],
                               metrics['mean_rewards'])

            if len(metrics['std_rewards']) > 0:
                std_upper.set_data(
                    mean_episodes[:len(metrics['mean_rewards'])],
                    [m + s for m, s in zip(metrics['mean_rewards'], metrics['std_rewards'])]
                )
                std_lower.set_data(
                    mean_episodes[:len(metrics['mean_rewards'])],
                    [m - s for m, s in zip(metrics['mean_rewards'], metrics['std_rewards'])]
                )

            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            if (batch_num + 1) % save_interval == 0:
                agent.save(save_path)
                print(f"\nModel saved to {save_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        agent.save(save_path)
        print(f"Model saved to {save_path}")
    except Exception as e:
        print(f"\nTraining interrupted due to error: {str(e)}")
        agent.save(save_path)
        print(f"Model saved to {save_path}")
        raise e
    finally:
        pbar.close()
        plt.ioff()
        metrics['training_duration'] = time.time() - start_time

        if best_params is not None:
            agent.mean = best_params

        env.close()

    return metrics




## FOR SELFPLAY
def train_selfplay_cem(
    env,
    agent,
    num_episodes: int,
    device=None,
    batch_size: int = 60,
    elite_frac: float = 0.2,
    eval_interval: int = 60,
    save_interval: int = 100,
    save_dir: str = "saved_models",
    render: bool = False
) -> Dict[str, List[float]]:

    #train the CEM agent in a selfplay  

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "cem_volleyball.pt")
    best_model_path = os.path.join(save_dir, "cem_volleyball_best.pt")

    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")

    # clone agent for opponent
    opponent_agent = copy.deepcopy(agent)

    agent.device = device
    agent.mean = agent.mean.to(device)
    agent.std = agent.std.to(device)
    agent.population = agent.population.to(device)

    opponent_agent.device = device
    opponent_agent.mean = opponent_agent.mean.to(device)
    opponent_agent.std = opponent_agent.std.to(device)
    opponent_agent.population = opponent_agent.population.to(device)


    hall_of_fame = []

    metrics = {
        "episode_rewards": [],       # Right agent
        "opponent_rewards": [],      # Left agent 
        "mean_rewards": [],
        "opponent_mean_rewards": [],
        "vs_baseline_rewards": [],
        "std_rewards": [],
        "best_reward": float("-inf"),
        "training_duration": 0
    }

    start_time = time.time()
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    rewards_line, = ax.plot([], [], 'b-', alpha=0.3, label='Agent Reward')
    opponent_line, = ax.plot([], [], 'r-', alpha=0.3, label='Opponent Reward')
    mean_line, = ax.plot([], [], 'b-', linewidth=2, label='Agent Mean Reward')
    op_mean_line, = ax.plot([], [], 'r-', linewidth=2, label='Opponent Mean Reward')
    eval_line, = ax.plot([], [], 'g*', markersize=8, label='Vs Built-in Policy')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Self-Play Training Progress')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    pbar = tqdm(total=num_episodes, desc="Self-Play Training")

    best_mean_reward = float('-inf')
    best_params = None

    try:
        for episode in range(num_episodes):

            use_random_opponent = False
            if len(hall_of_fame) > 0 and np.random.random() < 0.3:
                use_random_opponent = True
                historical_opponent = np.random.choice(hall_of_fame)
                opponent_agent.mean = historical_opponent['mean'].clone()

            states_right, actions_right, rewards_right = [], [], []
            states_left, actions_left, rewards_left = [], [], []

            state_right = env.reset()

            _, _, _, info = env.step(np.zeros(3))  # dummy step
            state_left = info['otherObs']

            done = False
            episode_reward_right = 0
            episode_reward_left = 0

            prev_ball_x, prev_ball_y, prev_ball_vy = None, None, None
            hit_count = 0
            successful_hit = False
            crossed_net = False
            agent_move_counter = 0
            good_position = False

            while not done:
                if isinstance(state_right, (int, float)):
                    state_right = np.array([state_right], dtype=np.float32)
                elif not isinstance(state_right, np.ndarray):
                    state_right = np.array(state_right, dtype=np.float32)

                if isinstance(state_left, (int, float)):
                    state_left = np.array([state_left], dtype=np.float32)
                elif not isinstance(state_left, np.ndarray):
                    state_left = np.array(state_left, dtype=np.float32)

                # actions 
                action_right = agent.select_action(state_right)
                action_left = opponent_agent.select_action(state_left)

                next_state_right, reward_right_val, done, info = env.step(action_right, action_left)
                next_state_left = info['otherObs']
                reward_left_val = -reward_right_val  # Zero-sum

                states_right.append(state_right)
                actions_right.append(action_right)
                rewards_right.append(reward_right_val)

                states_left.append(state_left)
                actions_left.append(action_left)
                rewards_left.append(reward_left_val)

                #add more reward system 
                episode_reward_right += reward_right_val
                episode_reward_left += reward_left_val

                state_right = next_state_right
                state_left = next_state_left

                if render:
                    env.render()

            metrics["episode_rewards"].append(episode_reward_right)
            metrics["opponent_rewards"].append(episode_reward_left)

            if (episode + 1) % batch_size == 0:
                actions_right_np = np.array(actions_right, dtype=np.float32)
                actions_left_np = np.array(actions_left, dtype=np.float32)

                update_info_right = agent.update(states_right, actions_right_np, rewards_right, elite_frac)
                if not use_random_opponent:
                    update_info_left = opponent_agent.update(states_left, actions_left_np, rewards_left, elite_frac)

                # very slow decay of noise std
                if hasattr(agent, 'noise_std'):
                    agent.noise_std *= 0.998
                    agent.noise_std = max(agent.noise_std, 0.05)
                if hasattr(opponent_agent, 'noise_std') and not use_random_opponent:
                    opponent_agent.noise_std *= 0.998
                    opponent_agent.noise_std = max(opponent_agent.noise_std, 0.05)

                # calculate stats
                start_idx = max(0, len(metrics['episode_rewards']) - 10)
                recent_rewards = metrics['episode_rewards'][start_idx:]
                recent_op_rewards = metrics['opponent_rewards'][start_idx:]
                current_mean = np.mean(recent_rewards)
                current_op_mean = np.mean(recent_op_rewards)
                current_std = np.std(recent_rewards)

                metrics["mean_rewards"].append(current_mean)
                metrics["opponent_mean_rewards"].append(current_op_mean)
                metrics["std_rewards"].append(current_std)

                # save best model
                if current_mean > best_mean_reward:
                    best_mean_reward = current_mean
                    best_params = agent.mean.clone()
                    torch.save({
                        'mean': best_params,
                        'std': agent.std,
                        'reward': best_mean_reward
                    }, best_model_path)
                    print(f"\nNew best model saved with mean reward: {best_mean_reward:.2f}")

                if len(hall_of_fame) < 5 or np.random.random() < 0.1:
                    hall_of_fame.append({
                        'mean': agent.mean.clone(),
                        'performance': current_mean
                    })
                    print(f"Added new model to hall of fame (size: {len(hall_of_fame)})")

                if len(hall_of_fame) > 10:
                    hall_of_fame.sort(key=lambda x: x['performance'])
                    hall_of_fame.pop(0)

                pbar.set_postfix({
                    'agent_reward': f"{current_mean:.2f}",
                    'opp_reward': f"{current_op_mean:.2f}",
                    'best': f"{best_mean_reward:.2f}"
                })

            pbar.update(1)

            # plotting 
            if (episode + 1) % 5 == 0:
                episodes_x = list(range(len(metrics["episode_rewards"])))
                rewards_line.set_data(episodes_x, metrics["episode_rewards"])
                opponent_line.set_data(episodes_x, metrics["opponent_rewards"])

                mean_x = list(range(0, len(metrics["episode_rewards"]), batch_size))
                if len(mean_x) < len(metrics["mean_rewards"]):
                    mean_x.append(len(metrics["episode_rewards"]) - 1)

                mean_line.set_data(mean_x[:len(metrics["mean_rewards"])],
                                   metrics["mean_rewards"])
                op_mean_line.set_data(mean_x[:len(metrics["opponent_mean_rewards"])],
                                      metrics["opponent_mean_rewards"])

                if len(metrics["vs_baseline_rewards"]) > 0:
                    eval_x = list(range(0, len(metrics["episode_rewards"]), eval_interval))[:len(metrics["vs_baseline_rewards"])]
                    eval_line.set_data(eval_x, metrics["vs_baseline_rewards"])

                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            # evaluate vs baseline 
            if (episode + 1) % eval_interval == 0:
                env.reset()
                eval_rewards = []
                for _ in range(5):
                    eval_state = env.reset()
                    eval_done = False
                    eval_episode_reward = 0

                    if best_params is not None:
                        agent.mean = best_params.clone()

                    while not eval_done:
                        if isinstance(eval_state, (int, float)):
                            eval_state = np.array([eval_state], dtype=np.float32)
                        elif not isinstance(eval_state, np.ndarray):
                            eval_state = np.array(eval_state, dtype=np.float32)

                        with torch.no_grad():
                            eval_action = agent.select_action(eval_state)

                        eval_next_state, eval_reward, eval_done, _ = env.step(eval_action)
                        eval_episode_reward += eval_reward
                        eval_state = eval_next_state

                    eval_rewards.append(eval_episode_reward)

                eval_mean = np.mean(eval_rewards)
                metrics["vs_baseline_rewards"].append(eval_mean)
                print(f"\nEvaluation vs baseline at episode {episode + 1}: Mean reward = {eval_mean:.2f}")

            # Save model periodically
            if (episode + 1) % save_interval == 0:
                if best_params is not None:
                    agent.mean = best_params.clone()
                agent.save(save_path)
                print(f"\nModel saved to {save_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        if best_params is not None:
            agent.mean = best_params.clone()
        agent.save(save_path)
        print(f"Model saved to {save_path}")
    except Exception as e:
        print(f"\nTraining interrupted due to error: {str(e)}")
        if best_params is not None:
            agent.mean = best_params.clone()
        agent.save(save_path)
        print(f"Model saved to {save_path}")
        raise e
    finally:
        pbar.close()
        plt.ioff()
        metrics["training_duration"] = time.time() - start_time
        if best_params is not None:
            agent.mean = best_params.clone()
        env.close()

    return metrics
