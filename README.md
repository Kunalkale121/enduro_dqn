# enduro_dqn
Deep Q-Learning Agent trained on the Atari Enduro environment using PyTorch and Gymnasium

# 📘 Deep Q-Learning Agent for Atari Enduro  
**Author:** Kunal Kale  

---

## 1. Project Overview  
This project demonstrates the implementation of a **Deep Q-Learning (DQN)** agent trained to play the **Atari Enduro-v5** game using the **OpenAI Gymnasium** and **Arcade Learning Environment (ALE)** frameworks.  

The objective of the project was to build a reinforcement learning agent capable of learning optimal driving behavior in the Enduro environment — overtaking cars, managing acceleration, and adapting to day-night visual conditions — purely through interaction and experience.  

This work integrates principles of **value-based reinforcement learning**, **neural function approximation**, and **experience replay** to develop an agent that learns to maximize long-term cumulative reward.

---

## 2. Training Setup  

### Hardware & Runtime  
- Environment: Google Colab  
- GPU: T4 GPU  
- CUDA: Version 12.4  
- Framework: PyTorch  

### Libraries Used  
- `gymnasium[atari]` and `ale-py` for the Atari environment  
- `torch`, `torchvision`, `torchaudio` for deep learning  
- `opencv-python` for frame preprocessing  
- `tensorboard` for performance tracking  

### Environment  
- Game: **ALE/Enduro-v5**  
- Observation space: `(210, 160, 3)` (RGB frames)  
- Action space: 9 discrete actions  
- Observation preprocessing: converted to grayscale, resized to `(84 × 84)`, and stacked over 4 frames for temporal context.

---

## 3. Environment Analysis  

**States:**  
Each state is represented by 4 consecutive grayscale frames of size 84×84, giving the agent spatial and temporal awareness of motion and position.  

**Actions:**  
The agent has 9 discrete actions, including acceleration, braking, and steering directions.  

**Q-Table Size:**  
Instead of a traditional table, the DQN uses a **neural network approximator**. The Q-network outputs 9 Q-values per state (one per action).  

---

## 4. Reward Structure  

**Reward Definition:**  
The environment’s native reward structure was used, where the agent earns positive rewards for overtaking cars.  

**Rationale:**  
This aligns directly with the game’s objective — maximizing cars overtaken — making the reward signal both interpretable and stable for training.  

**Reward Clipping:**  
Rewards were clipped to `[-1, +1]` to stabilize learning and reduce the impact of large reward spikes on gradient updates.

---

## 5. Bellman Equation Parameters  

**Alpha (Learning Rate):** `2.5 × 10⁻⁴`  
**Gamma (Discount Factor):** `0.99`  

The learning rate was chosen to balance stability and convergence speed, while a high gamma ensured that the agent prioritized long-term success over immediate gains.  

**Effect of Parameter Changes:**  
Adjusting gamma to lower values caused the agent to favor short-term actions (e.g., immediate acceleration) instead of sustained optimal driving.  
Higher gamma produced smoother, more strategic performance.

---

## 6. Policy Exploration  

The baseline policy used was **ε-greedy**, where ε decayed from **1.0 to 0.01** over 200,000 steps.  
The agent began with high exploration and gradually shifted toward exploitation as learning progressed.  

Alternative policies such as Boltzmann exploration were considered, but ε-greedy offered simpler control and faster convergence for this environment.

---

## 7. Exploration Parameters  

**Decay Rate:** Linear decay over 200,000 steps  
**Initial ε:** 1.0  
**Final ε:** 0.01  

At around **episode 60**, ε reached **0.012**, indicating that the agent had transitioned to primarily exploiting learned behavior.  

---

## 8. Baseline Performance  

**Training Configuration:**  
- Episodes: 500  
- Max steps per episode: 18,000  
- Replay buffer: 100,000 transitions  
- Batch size: 32  
- Target network update: every 10,000 steps  
- Optimizer: Adam  
- Learning rate: 2.5e-4  
- Reward clipping: Enabled  

---

## 9. Performance Metrics  

| Metric | Value |
|:--|:--|
| Episodes logged | 500 |
| Last-100 avg return | **222.51** |
| Last-100 avg steps | **4592.64** |
| Overall mean return | **59.49** |
| Overall mean steps | **3580.93** |
| Final avgR@10 | **391.2** |

**Interpretation:**  
The agent demonstrated a clear improvement throughout training, progressing from random movement (avgR@10 ≈ 0) to strategic driving behavior (avgR@10 ≈ 391).  
Longer episode durations and higher average returns indicate successful learning of sustained control.

---

## 10. Q-Learning Classification  

Q-Learning is a **value-based** reinforcement learning method.  
It estimates the **expected cumulative reward (Q-value)** for each state-action pair and uses the **Bellman optimality equation** to iteratively update estimates toward their true values.

---

## 11. Bellman Equation Concept  

The Bellman equation models the **expected lifetime value** of an action as the immediate reward plus the discounted value of future rewards.  
It enables the agent to learn optimal policies by maximizing long-term outcomes instead of short-term payoffs.

> **Q(s, a) ← Q(s, a) + α [ r + γ maxₐ′ Q(s′, a′) − Q(s, a) ]**

---

## 12. Q-Learning vs. LLM-Based Agents  

| Aspect | Deep Q-Learning Agent | LLM-Based Agent |
|:--|:--|:--|
| **Learning Method** | Trial and error via environment interaction | Pre-trained on vast datasets |
| **Knowledge Source** | Experience (rewards, states, actions) | Contextual text and memory |
| **Goal** | Maximize cumulative rewards | Generate coherent responses or reasoning |
| **Adaptation** | Real-time feedback learning | Fine-tuning or reinforcement from human feedback |

DQN agents optimize numeric rewards through interaction; LLMs optimize textual coherence through probabilistic modeling.

---

## 13. Reinforcement Learning for LLM Agents  

Concepts from reinforcement learning — particularly **reward-based optimization** and **policy improvement** — can be extended to **LLM-based agents**.  
Examples include **Reinforcement Learning from Human Feedback (RLHF)**, which fine-tunes LLMs using reward models derived from human preference signals.

---

## 14. Planning in RL vs. LLM Agents  

In classical RL, planning involves **predicting future states and rewards** through simulation or experience replay.  
For LLMs, planning refers to **multi-step reasoning and goal decomposition**.  

**Example:**  
- RL agent plans by simulating actions and rewards.  
- LLM agent plans by chaining reasoning steps in text form to reach a logical outcome.

Both approaches use sequential decision-making, but their representations (state-action vs. token-sequence) differ fundamentally.

---

## 15. Q-Learning Algorithm Explanation  

**Algorithm Summary:**  
1. Initialize replay buffer and Q-networks (main and target).  
2. For each episode:
   - Select actions using ε-greedy policy.  
   - Observe rewards and next states.  
   - Store transitions in the replay buffer.  
   - Periodically sample batches to update Q-network using Bellman updates.  
   - Sync target network weights.  

This loop continues until convergence, where the Q-network consistently predicts optimal actions.

---

## 16. Integration with LLM Agents  

Deep Q-Learning agents could be integrated with LLM systems for **decision-making tasks** where language and environment interact — such as autonomous dialog control or dynamic strategy games.  

An LLM could act as a **high-level planner**, while a DQN agent executes **low-level environment actions** based on learned control policies.

---

## 17. Code Attribution  

All code was authored by **Kunal Kale**, with structural guidance adapted from **ChatGPT’s PyTorch-based DQN framework**.  
The agent design follows the standard OpenAI Gymnasium and PyTorch DQN conventions, restructured for clarity and reproducibility.

---

## 18. Licensing  

This project includes the following license acknowledgments:  
- **MIT License** – for open educational use.  
- **Apache 2.0 License** – for compatibility and reuse.  
- **GNU GPLv3 License** – for derivative and academic work.  
