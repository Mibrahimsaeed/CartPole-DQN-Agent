# CartPole-DQN-Agent
Warehouse Robot Path Optimization using Deep Q-Learning (DQN)

## 📌 Project Overview

This project demonstrates how a robot can learn to navigate a warehouse environment using **Deep Q-Learning (DQN)**, a reinforcement learning technique.

The robot learns through trial and error to find the **shortest and safest path** from a starting point to a goal while avoiding obstacles.

Instead of following pre-programmed rules, the agent **learns optimal behavior automatically using rewards and penalties**.

---

## 🎯 Objective

The main goal of this project is:

- To train an AI agent that can navigate a grid-based warehouse efficiently  
- To minimize the number of steps required to reach the target  
- To avoid obstacles (shelves/walls)  
- To demonstrate Deep Reinforcement Learning in a simple environment  

---

## 🧠 What is Deep Q-Learning (DQN)?

Deep Q-Learning is a reinforcement learning algorithm where:

- An **agent learns by interacting with an environment**  
- A **neural network predicts the best action (Q-values)**  
- The agent improves over time using rewards  

Instead of storing values in a table, DQN uses a **neural network to approximate Q-values**.

---

## 🏗️ Environment Description

The environment is a **2D grid warehouse**:

- `S` → Start position (Robot)  
- `G` → Goal position (Target item)  
- `X` → Obstacles (Shelves/Walls)  
- `.` → Empty space  

### Example Grid:

```
S . . X .
. X . . .
. . X . .
X . . . .
. . . X G
```

---

## 🎮 Agent Actions

The robot can perform 4 actions:

- Move Up ⬆️  
- Move Down ⬇️  
- Move Left ⬅️  
- Move Right ➡️  

---

## 🏆 Reward System

| Action            | Reward |
|------------------|--------|
| Reach goal       | +100   |
| Hit obstacle     | -10    |
| Each step taken  | -1     |

This encourages the agent to:
- Reach the goal faster  
- Avoid obstacles  
- Minimize unnecessary movement  

---

## 🧠 Key Concepts Used

- Deep Q-Network (DQN)  
- Experience Replay  
- Target Network  
- Epsilon-Greedy Exploration  
- Reinforcement Learning (Markov Decision Process)  

---

## ⚙️ Tech Stack

- Python 🐍  
- PyTorch 🔥  
- NumPy  
- Matplotlib (for visualization)  

---

## 📊 Training Process

1. Initialize environment and agent  
2. Agent observes current state  
3. Chooses action using epsilon-greedy policy  
4. Receives reward and next state  
5. Stores experience in replay buffer  
6. Trains neural network using sampled experiences  
7. Updates target network periodically  

This process repeats for multiple episodes until the agent learns optimal behavior.

---

## 📈 Results

After training, the agent learns to:

- Reach the goal efficiently  
- Avoid obstacles  
- Reduce unnecessary movements  
- Improve performance over time  

A reward graph is used to visualize learning progress.

---

## 🎥 Demo

```
Before Training: Random movement
After Training: Optimal path to goal
```

(Add your GIF or video here)

---

## 📁 Project Structure

```
warehouse-dqn/
│
├── environment/        # Grid world environment
├── agent/              # DQN agent implementation
├── model/              # Neural network
├── train.py            # Training script
├── test.py             # Testing script
├── utils.py            # Helper functions
└── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install numpy torch matplotlib
```

### 2. Run training
```bash
python train.py
```

### 3. Test trained agent
```bash
python test.py
```

---

## 📌 Future Improvements

- Add multiple robots  
- Add moving obstacles  
- Use Double DQN or Dueling DQN  
- Scale to larger warehouse grids  
- Add real-time visualization (GUI)  

---

## 💡 Key Learning Outcome

This project helps understand:

- How reinforcement learning works in real-world problems  
- How agents learn from rewards and penalties  
- How Deep Q-Networks replace Q-tables  
- How AI can be used for path planning and automation  


