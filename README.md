# Reinforcement Learning for Volleyball: A Comprehensive Analysis of Different RL Algorithms

<img width="100%" src="A2C/a2c_selfplay_video.gif" width="500">

## Group members

- Caleb Galdston
- Aden Harris
- Aydin Tabatabai
- Sia Khorsand

# Abstract 

In this project, we aim to compare the performance of different reinforcement learning algorithms at playing slime volleyball- a 2-dimensional volleyball game in which the objective is to get the ball to land on the opponent's side of the net. To train our agents, we used the SliveVolleyGym environment which allowed us to test the performance of multi-agent RL algorithms and have different agents play in the same environment. This environment has a simple reward structure: the agent receives \+1 if the ball lands on the opponent side of the net, and \-1 if it lands on the agent’s side. To compare performance of our algorithms, we will focus on the average reward and win rate over \~1000+ episodes against the baseline model defined in the environment. Additionally, we will see the average reward they receive when competing against one another. Each episode has a maximum score of 5 as it ends after one of the agents wins 5 rounds. This paper discusses the key aspects of the implementations, evaluates learning progress, and highlights potential reasons for the limitations observed.

# Background

Reinforcement Learning has seen widespread success in complex environments such as robotics, gaming, and multi-agent systems. In this project, we apply different RL algorithms to train a volleyball agent. The goal is to evaluate whether these methods could effectively learn optimal strategies through direct training with a baseline agent and self-play. We implemented and compared the performances of Deep Q-Networks (DQN), Cross-Entropy Method (CEM), Advantage Actor-Critic (A2C), and Proximal Policy Optimization (PPO). 

### DQN
Deep Q-Learning eliminates the need for a Q-table, which an agent traditionally uses to maximize future rewards. Implementing a Q-table becomes impractical in large or complex environments, such as the SlimeVolley environment, which operates in a continuous space with 12-dimensional features. Instead, Deep Q-Learning employs a deep neural network to approximate Q-values, making it a more scalable and feasible approach for such tasks. This method was first introduced by researchers at Google DeepMind in their 2013 paper, *Playing Atari with Deep Reinforcement Learning* by Mnih et al. <a name="mnih"></a>[<sup>[1]</sup>](#mnihnote) The paper's goal was to merge deep learning with traditional reinforcement learning techniques, such as Q-Learning, to create an algorithm capable of learning to play Atari games using pixelated images as inputs. A key innovation in their approach was the use of experience replay, where the agent stores past experiences in a memory pool and randomly samples from this pool to perform Q-Learning updates. This technique helps stabilize training by breaking the correlation between consecutive experiences. While our SlimeVolley environment doesn't rely on pixelated images, we believe this strategy will still be effective in helping our agent learn optimal volleyball strategies.

In addition to the foundational DQN approach, we explored more recent advancements that have demonstrated significant improvements. One such advancement is Double Q-Learning, introduced by van Hasselt et al. in their paper *Deep Reinforcement Learning with Double Q-learning* <a name="hasselt"></a>[<sup>[2]</sup>](#hasseltnote). This method addresses a known issue in DQN: the tendency to overestimate Q-values in certain scenarios. Double Q-Learning extends the original tabular Double Q-Learning concept to the DQN framework. Traditional Q-Learning and DQN use the same values to both select and evaluate actions, which can lead to compounded overestimations. Van Hasselt's solution decouples action selection from evaluation, mitigating this issue. For our project, we aim to investigate whether implementing Double Q-Learning can enhance the performance of our algorithm in the SlimeVolley environment.

### CEM
Cross-Entropy Method is a population-based evolutionary approach that iteratively updates over parameters based on elite samples of each population. Our implemented agent follows this process, employing a similar approach to Rubinstein's adaptive CEM framework <a name="mnih"></a>[<sup>[5]</sup>](#mnihnote):

- Generate a population of parametrized policies.  
- Evaluate their performance in the volleyball environment.  
- Select elite policies based on rewards and update the parameter distributions.  
- Repeat the process for a fixed number of episodes.

In this implementation, policy updates follow the CE minimization principle through iterative distribution tightening, where elite selection acts as our (1-ϱ)-quantile threshold γ_t from Algorithm 2.1"  <a name="mnih"></a>[<sup>[5]</sup>](#mnihnote)

The agent was implemented using PyTorch and trained on CUDA using google colab. The key parameters of the CEM algorithm are:

- Population size : \[300,600,1000\]  
- Elite Ratio: \[15%, 20%\]  
- Initial Noise Std: \[0.65, 0.85, 1.2\]  
- Learning Rate: \[0.01, 0.0075, 0.1\]  
- Experience Buffer: \[1000,2000\]  
- Episodes: \[500,1000,2000,5000\]
- Introduce a curriculum progress approach to improve learning by gradually increasing difficulty.(implemented only in a few iterations of the process)

CEM is inherently a stochastic optimization technique that relies on selecting elite samples for parameter updates. However, because the volleyball environment involves long-term dependencies and requires strategic decision-making, CEM often struggles. It is more suited for tasks with direct reward feedback rather than delayed rewards over extended interactions<a name="mnih"></a>[<sup>[6]</sup>](#mnihnote). Additionally, because the method lacks a value function or policy gradient updates, it is prone to inefficient exploration and overfitting to short-term rewards rather than developing meaningful gameplay strategies.


### A2C



Advantage Actor-Critic (A2C) is a reinforcement learning algorithm that combines policy-based and value-based learning to improve training efficiency. In traditional reinforcement learning, policy gradient methods often suffer from high variance, making learning unstable. On the other hand, value-based approaches like Q-learning struggle in environments with continuous state spaces, where discretization can limit learning. A2C addresses these issues by using an actor network, which selects actions, and a critic network, which estimates the expected reward, allowing for more stable policy updates<a name="sutton"></a>[<sup>[3]</sup>](#sutton).

A2C builds upon the actor-critic framework, first introduced by Konda & Tsitsiklis (2002), which demonstrated that using a Critic to estimate future rewards helps stabilize learning<a name="konda"></a>[<sup>[4]</sup>](#konda). A2C improves upon this by using synchronous updates, where multiple agents collect experiences in parallel, leading to faster and more efficient learning. This approach makes A2C well-suited for dynamic, multi-agent environments like SlimeVolley.

Further advancements in actor-critic methods have included improvements such as Asynchronous Advantage Actor-Critic (A3C), which asynchronously updates multiple agents to refine learning. While A3C has advantages in scalability, A2C remains a strong alternative due to its more stable gradient updates. Given the competitive and fast paced nature of the environment, A2C provides a balance between exploration and learning stability, making it a better choice for this task.

### PPO



Proximal Policy Optimization (PPO) is a reinforcement learning algorithm that improves upon traditional policy gradient methods by limiting policy updates to stay within a "trust region." It achieves this by using a clipped objective function that prevents drastic policy changes between updates <a name="schulman"></a>[<sup>[1]</sup>](#schulmannote). 

Our implementation of PPO for the Slime Volleyball Environment differed from the original PPO implementation in several ways:

- **Curriculum Learning:** We employed curriculum learning, which gradually increased the difficulty for the agent by adjusting the ball speed from 70 percent to 100 percent over the training episodes, allowing the agent to learn more easily at the beginning of the training <a name="narvekar"></a>[<sup>[2]</sup>](#narvekarnote).

- **Reward Shaping:** We added small survival bonuses and rewards for beneficial actions like hitting the ball, addressing the sparse reward problem in the original implementation.

- **Architecture:** The neural network we used in our PPO utilized a shared feature extraction followed by separate actor and critic heads, while maintaining the same PPO algorithm.


### PPO with intrinsic reward mechanism

We then implemented a PPO algorithm with an intrinsic reward mechanism, to further encourage exploration and skill acquisition. It provided additional rewards for novel state-action pairs, and reducing reliance on sparse external rewards. Specifically:
- Curiosity-driven exploration: An intrinsic reward was assigned based on the agent's ability to predict state transitions, encouraging it to explore unfamiliar states.
- State novelty bonuses: The agent received rewards for visiting novel states, promoting diverse gameplay strategies.
- Intrinsic-extrinsic reward balancing: The total reward signal combined both extrinsic rewards from game performance and intrinsic rewards from exploration, ensuring meaningful skill development.

# Problem Statement

How well do different reinforcement learning algorithms compare when playing a zero-sum game like volleyball? Is it possible to optimize learning for a game with only a few rules to follow? What key distinctions between these algorithms make them more suitabele for this task. 

# Data

This project did not include any datasets, however, the agents did generate data about the rewards they had received and what actions were optimal. Our environment had a continuous state space, and a discrete action space (stay, left, right, jump, left and jump, right and jump). 


# Proposed Solution
Our framework trains an agent in the slimevolleygym environment—using its baseline model as a benchmark—by implementing multiple reinforcement learning algorithms (DQN, CEM, PPO, and A2C) in Python. We leverage PyTorch for model building, Gym and slimevolleygym for the simulation environment, NumPy for numerical operations, and matplotlib for visualization. By training the agent both against the baseline model and in self-play, we expose it to diverse strategies and skill levels, helping it adapt and improve over time. A structured reward function—balancing immediate successes (like ball returns) with overall match outcomes—guides the agent toward increasingly efficient and competitive gameplay. This dual training approach of baseline-vs.-agent and self-play offers a comprehensive evaluation of each algorithm’s learning curve, robustness, and eventual performance against the benchmark model.

# Evaluation Metrics

For our evaluation metric, we wanted to compare how our agents performed against the baseline model, so after training each of the models individually, we started by having them play 500+ games against the baseline to see how the average reward across these games compared.

# Results


### Deep Q-Network

Our first implementation of the DQN algorithm was very simple. We discretized all possible actions, and used a four layer neural network to estimate q-values. It followed very closely to what we did for the cartpole balance task in assignment four. Even with such a basic architecture, it was able to beat the baseline model a few times. A few issues that we quickly noticed were that we didn't need to include all of the actions, since a few possible options such as going left and right at the same time were likely not going to be useful in this environment. Additionally, we noticed that the epsilon decay rate may have been a bit too aggressive, as early on it didn't seem like the agent was exploring many different actions as for the first few hundred iterations, it made almost no progress at all. The reward function that the environment uses by default was also a bit sparse as the agent only receives positive or negative rewards after the ball hits the ground, which doesn't encourage potentially positive actions such as hitting the ball over the net. However, to keep our benchmarks the same across the different algorithms, we decided that it would be best to keep it this way. After playing 500 test games, it had an average score of -4.4 with a standard deviation of 0.674. 

Here is a plot of the reward from the first 1000 training episodes of our dqn training. 

<img src="plots/dqn_one.png" width="500" height="300">


As we can see towards then last few hundred episodes the agent definately showed signs of improvement, however, we wanted to see if we could do better. To attempt to improve performaces, we decreased the rate of epsilon decay so that the agent would explore more states early on. Additionally, we increased the complexity of the model by adding more layers to the neural network and modified the evaluation and selction of actions so it followed the Double Q-Learning approach introduced by van Hesselt. These changes resulted in a slight improvement in training, but also caused the agent train signficantly more slowly due to the complexity of the network. 

Here is a plot of the reward from the first 1000 training episodes of the algorithm with these improvements. 

<img src="plots/dqn_two.png" width="500" height="300">

Based on this plot we can see that this algorithm started to explore good strategies early, however, it did slightly worse than our first implementation at improving more in later training episodes. It seems that our original network wasn't having too significant of an issue in terms of overestimating Q-values. 

Overall, Deep Q-Network's were not the most suitable approach to this task as they do not handle continuous state spaces very well. However, it was interesting to see that these agents did show signs up improvement, and if given more training episodes, it is likely that we would have seen even more improvement. The baseline model for this environment was fairly strong, so it was promising to see that this agent actually won a few games to five against it. 

Here is an video of our DQN agent playing against the baseline. It is the yellow agent on the right. 

<img src="plots/dqn_video.gif" width="500" height="300">

As we can see, it learned that jumping often could lead to better performance, but it doesn't seem to track the ball very well. 



### CEM 

Training Against a Baseline Agent: 
Initially, the agent was trained against a pre-defined baseline opponent. The expectation was that the agent would gradually learn strategies to outperform the baseline. However, despite multiple training runs, the agent failed to show meaningful improvement in win rates or reward accumulation. One key limitation was that CEM’s updates were overly influenced by short-term fluctuations rather than a structured learning path.

<img src="plots/CEMFail.png" width="500" height="300">


Self-Play Training: 
To assess whether self-play could yield better learning, the agent was trained against a clone of itself. This approach allowed it to gradually adapt to a dynamic opponent rather than a fixed strategy. Self-play resulted in a more stable learning process compared to training against a static baseline, showing that the agent was able to adjust to its opponent more effectively over time. However, despite this improvement, the final win rate remained close to 50%, as shown below, indicating that while the agent could compete at an equal level with itself, it did not develop a consistently dominant strategy. One reason could be that the training dynamics between two identical agents did not push the learning boundaries as effectively as adversarial training with a more advanced opponent, leading to limited strategic progression.

<img src="plots/LearningCEMSP.png" width="700" height="300">

This figure illustrates the reward progression over episodes for CEM. The moving average suggests minor improvements, but overall learning remained minimal. Despite high exploration via noise injection, the algorithm failed to converge to a stable policy.

<img src="plots/Win-RateCEM.png" width="500" height="300">

Overall, in terms of training stability, CEM exhibited variance in rewards, leading to inconsistent learning. Because its updates relied on selecting elite samples from a noisy distribution, it struggled to refine a stable policy and instead oscillated between different suboptimal behaviors.


Finally, as expected from the graphs, the agent did not do well while playing with the baseline model.

<img src="CEM/CEM.gif" width="700" height="300">


### A2C

The first approach taken was training the A2C agent against a built-in baseline opponent for 1,000 episodes. The idea was that by repeatedly playing against the same opponent, the agent would gradually learn strategies to win more points. However, the results were disappointing, the agent showed little to no improvement over time, with reward values staying close to -5, meaning it was losing almost every game. A major issue with this approach was that the baseline opponent was already strong, making it difficult for the agent to get meaningful feedback. Since it started off losing almost every round, the learning process was unstable, and the policy updates were not effective. The graph below shows that rewards fluctuated but remained mostly negative, indicating no real progress in learning a competitive strategy. When tested against the baseline after training, as seen in the video below, the agent learned no real strategy, only jumping up and down.

<img src="A2C/a2c_baseline_training_plot.png" width="500" height="300">

<img src="A2C/a2c_baseline_video.gif" width="500">

To see if a different approach would work better, the agent was then trained using self-play for 5,000 episodes. Instead of always facing the same opponent, the agent played against a past version of itself, allowing it to learn dynamically as both sides improved over time. This method led to better gameplay behavior, the agent moved more efficiently and reacted more intelligently to the ball when rendering the matches. Unlike the baseline training, where the agent barely improved, self-play allowed it to develop some level of strategy. However, despite these improvements, it was still not good enough to consistently beat the baseline opponent. The reward plot for self-play training showed higher reward values compared to the baseline training, but this was most likely due to the fact that the opponent was weak. While the agent learned to play the game better than before, it never reached a point where it could consistently win matches against the baseline opponent. The video below demonstrates its improvements.

<img src="A2C/selfplay_a2c_training_plot.png" width="500" height="300">

<img src="A2C/a2c_selfplay_video.gif" width="500">

### PPO 
<img src="plots/PPO_winrate.png" width="500" height="300">

The initial win rate for PPO was high in the first 100 episodes but gradually declined as training progressd, after 1000 episodes the final win rate stablized at about 78% to 80$, indicating strong generalization against the baseline agent.

<img src="plots/PPO_intrisinic_rewards.png" width="500" height="300">

for Entrisic rewaards the running average initially increased rapidlly, reading a peak of around 5-6 rewards per episodes, as training continued, rewards flucuated signigicantly with an overall average of 3.88 in the final 100 episodes. The max recorded level was 24.45, while the min reward dropped to -3.15 showing variability in game performance

### PPO with Intrisic Motivators

<img src="plots/PPO_with_IM_winrate.png" width="500" height="300">

The inital winrate was high about 90 percent, but then stabilized at about 80 percent towards the end of training, there were high flucuations, likely do to the exploration-driven nature of the intrinsic rewards. Compared to standard PPO, PPO-IM showed greater stability in long term winrates, suggesting improved adaptability.

<img src="plots/PPO_with_IM_intrisicrewards.png" width="500" height="300">

The running average for extrenisc rewards increased steaility, reaching a final average of about 4 to 5 per episodes. The reward distribution remaied stable despite flucuations in the episode level performance. The max recorded reward was over 20, indicating some high reward stategies learned by the agent. 
For intrisic rewards, in the begining it dropped sharply in the early training, showing that the agent found many novel states but quickly exhausted novelty. The long term intrisinic reward stabilized close to zero, indicating that the agent converged to an optimal policy and no longer relied on exploration

Comparing PPO to PPO with IM
| Metric                  | PPO      | PPO-IM  |
|-------------------------|---------|--------|
| **Final Win Rate**      | ~78%    | **80%** |
| **Final Avg Reward**    | ~3.9    | **4-5** |
| **Exploration Stability** | Decreased over time | **More stable, thanks to IM** |
| **Entropy Trends**      | Decreased sharply | **Gradual decrease (better exploration balance)** |
| **KL Divergence**      | Consistent but fluctuating | **More controlled updates** |


# Discussion

### Interpreting the result

The main takeaway from our experiments is that none of the algorithms except PPO were able to consistently outperform the baseline model. While approaches like A2C, DQN, and CEM showed some learning progress, they struggled to develop strategies strong enough to win games reliably. The baseline model was already well-optimized, making it difficult for most algorithms to gain an advantage. PPO, on the other hand, demonstrated the most effective learning process, gradually improving its decision-making and maintaining a higher win rate. Even though PPO performed the best, it was not perfect. There were still inconsistencies in gameplay, suggesting that further improvements could help reinforcement learning agents reach a more dominant level of play.

The CEM agent failed to significantly improve when facing the baseline opponent. CEM’s reliance on stochastic parameter updates made it difficult for the agent to develop a structured strategy, especially in such environment with sparse rewards. The reward function had high variance due to the stochastic nature of the game, leading to unstable updates in CEM. In contrast, DQN leveraged a structured learning process, which allowed it to smooth out reward fluctuations over time.

After experimenting with different versions of DQN including adding more layers to the neural network and implementing a Double Q-Learning approach we conclude that increasing the complexity of our model didn’t improve performance in a significant way. When competing against such a strong baseline, and only receiving words for beating its opponent, DQN struggled to learn an optimal policy. However, despite computational limitations, we did see indications of it learning as it was trained on more episodes. Since it was challenging to encode a good representation of the state space, it was defeated by the superior agent on most episodes, however, it did tie the baseline on multiple occasions.

The A2C experiment showed that training against the baseline for 1,000 episodes led to almost no improvement. The agent consistently lost games without showing meaningful learning progress. The main issue was that the baseline was already strong, making it difficult for A2C to gain any early momentum. In contrast, switching to self-play for 5,000 episodes resulted in better overall gameplay. Movements became more natural and the agent showed better ball control. However, despite this improvement, the self-play model still couldn’t consistently beat the baseline. This suggests that while self-play helps an agent refine its strategy, it may not be enough to prepare it for an already strong opponent.

After experimenting with implementations for both PPO and PPO-IM, PPO-IM demonstrated greater stability and adaptability due to its intrinsic reward mechanism. The addition of intrinsic motivators helped maintain balanced exploration, leading to more consistent policy updates and improved long-term performance.


### Limitations and Challenges

#### CEM Limitations
These findings underscore that while evolutionary methods like CEM can excel in certain domains, the combination of sparse rewards and volatile gameplay in this slimevolleygym environment makes CEM far less effective than approaches with structured learning and memory mechanisms. This insight informs future algorithm selection, suggesting that methods robust to high-variance environments, like DQN and PPO, are better suited to play volleyball. 


##### DQN Limitations

One of the main challenges we faced when implementing the DQN was discretizing the state space. Since DQN's aren't optimized for continuous state spaces this hindered our agents performance. Another limitation was the reward structure. Since the rewards were binary (either you won or didn't), our agent struggled to learn what moves were actually helping the ball get over the net. Lastly, since DQN's are a very computationally expensive model, we were limited by the run time as each episode took quite some time to run. 

##### A2C Limitations
One limitation of the approach was training time. While we trained for 1,000 episodes against the baseline and 5,000 episodes with self-play, reinforcement learning algorithms like A2C often require more episodes to fully optimize performance. Another challenge was opponent variety. Self-play helped the agent improve movement, but since it was always training against a copy of itself, it may have overfit to a single playstyle rather than developing a generalized strategy. Finally, the simplistic reward function, which only provided +1 for scoring and -1 for losing, also may have limited learning by focusing on short-term gains rather than encouraging long-term strategies.

### PPO and PPO with IM
For both the PPO and PPO with IM while they both performed well against the baseline agent in training, when they were evaluated against it after training they both performed strangely poorly, never seeming able to win. After numerous attempts to bug fix this they still couldn't perform well against it. I think it's something to do with the way that the trained agent interacts with the testing action space, but I'm not sure

### Future work
While we saw some progress with our RL agents, there’s still a lot of room for improvement. I think that some of the algorithms we implemented didn't translate well to this environment. The creator of the environment already had implemented some effective algorithms, so in the future we would like to see how we could improve on the algorithms that they implemented. 

Additionally, for the algorithms that we tried, there were modifications that we could have implemented that could have enhanced performance. For example, we read some literature about Dueling Deep Q-Networks and Rainbow DQN's which are more recent advancements that could have improved our performance. Adjusting the reward system could be another area worth investigating, since adding smaller rewards for good in-game behaviors could encourage smarter play. Finally, training against a variety of opponents instead of just a single baseline or selfplay model could help agents develop more adaptable and well-rounded strategies.


### Ethics & Privacy

Ensuring ethical AI development is critical, and while our project did not involve personal data, we did consider potential biases, unintended consequences, and responsible AI usage.

**1. Bias and Fairness:** Reinforcement learning models can sometimes develop biases based on how they are trained. To avoid this, we tested them in different conditions to make sure no single strategy has an unfair advantage. Additionally, we will analyzed performances across different training scenarios to identify and mitigate any emerging biases.

**2. Transparency and Interpretability:** AI decision-making can sometimes be difficult to understand. To improve transparency, we documented agent behaviors, provided visualizations of their learning process, and conducted detailed evaluations on how different factors influence agent decisions. This helps to ensure that the AI’s reasoning can be explained and justified.

**3. Unintended Advantages:** AI agents can come up with unexpected ways to win, like taking advantage of flaws in the simulation instead of actually playing well. We closely monitored training to make sure they develop fair and realistic volleyball skills. Additionally, we paid close attention to agent decisions and gameplay to ensure that the strategies developed aligned with human expectations.

By addressing these ethical considerations, we tried to develop a fair, responsible, and transparent AI system that contributes positively to the reinforcement learning research community.

### Conclusion

Through our exploration of RL applied to a 2D volleyball gameplay, we demonstrated that an agent trained via policy-based or value-based methods can gradually learn to perform fundamental volleyball actions by iterating over numerous simulated matches. Our experiments showed that careful reward shaping and state representation is significant in guiding the agent towards better learning. 
Additionally, the results highlighted that hyperparameter tuining and the choice of RL models significantly influence stability in training. Through evaluation of agents versus static and adaptive opponents, we found that the learned policies were not capable of consistently outperfoming simpler, rule-based strategies. Our PPO and PPO IM implementation performed the best out of all of the models we tested, getting to a 80% win rate against the baseline agent in the end of training.

In summary, this project highlights the feasibility and promise of reinforcement learning in complex and fast paced sport domains like volleyball. Future research diredctions include refining the simulation environment for better state representations, exploring multi-agent coordination for team-based volleyball scenarios, and using transfer learning to quickly adapt policies trained in simulation to real world conditions. These steps could further enhance both the training efficiency and the tactical depth of RL driven volleyball agents, ultimately bringing them closer to real competitive performance.

# Footnotes
<a name='mnihnote'></a>1.[^](#mnihnote):  Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D. & Riedmiller, M. (2013). Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602<br>
<a name="hasselt"></a>2.[^](#hasseltnot): van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double Q-Learning. Proceedings of the AAAI Conference on Artificial Intelligence, 30(1). https://doi.org/10.1609/aaai.v30i1.1029<br> 
<a name="sutton"></a>3.[^](#sutton): Sutton, R., & Barto, A. (2005). Reinforcement Learning: An Introduction. IEEE Transactions on Neural Networks, 16(1), 285–286. https://doi.org/10.1109/tnn.2004.842673<br>
<a name="konda"></a>4.[^](#konda): Konda, V. R., & Tsitsiklis, J. N. (2002). Actor-critic algorithms. https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf<br>
<a name="deboer"></a>5.[^](#deboer): de Boer, Pieter-Tjerk, et al. “A Tutorial on the Cross-Entropy Method.” Annals of Operations Research, vol. 134, no. 1, Feb. 2005, pp. 19–67, https://doi.org/10.1007/s10479-005-5724-z.<br>
<a name="schulmannote"></a> [1] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv preprint arXiv:1707.06347*.
<a name="narvekarnote"></a> [2] Narvekar, S., & Stone, P. (2019). Learning Curriculum Policies for Reinforcement Learning. *International Joint Conference on Artificial Intelligence (IJCAI)*.



