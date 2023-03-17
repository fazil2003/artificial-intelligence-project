import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from gym import spaces

# Load the CSV dataset
df = pd.read_csv("malware.csv")

# Extract the payoffs for each player
defender_payoffs = df[["cm1", "cm2", "ci1", "ci2"]].values
attacker_payoffs = df[["l", "a", "b", "a0", "a1", "a2"]].values

# Define the NashDQN class
class NashDQN:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.memory = []
        self.model = self.build_model()

    def build_model(self):
        input_layer = Input(shape=(self.state_dim,))
        x = Dense(24, activation="relu")(input_layer)
        x = Dense(24, activation="relu")(x)
        output_layer = Dense(self.action_dim, activation="softmax")(x)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        action_probs = self.model.predict(state)[0]
        return np.random.choice(self.action_dim, p=action_probs)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = np.array(self.memory[-batch_size:])
        states = np.array(batch[:, 0].tolist())
        actions = np.array(batch[:, 1].tolist())
        rewards = np.array(batch[:, 2].tolist())
        next_states = np.array(batch[:, 3].tolist())
        done = np.array(batch[:, 4].tolist())
        q_values = self.model.predict(states)
        q_next_values = self.model.predict(next_states)
        targets = np.zeros((batch_size, self.action_dim))
        for i in range(batch_size):
            targets[i] = q_values[i]
            targets[i][actions[i]] = rewards[i] + self.gamma * np.max(q_next_values[i]) * (1 - done[i])
        self.model.fit(states, targets, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, env, num_episodes=1000, batch_size=32):
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done = env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                self.replay(batch_size)

# Define the malware environment
class MalwareEnv:
    def __init__(self, defender_payoffs, attacker_payoffs, cm1, cm2, ci1, ci2, l, a, b, a0, a1, a2, a3):
        self.defender_payoffs = defender_payoffs
        self.attacker_payoffs = attacker_payoffs.T
        self.num_defender_actions = defender_payoffs.shape[0]
        self.num_attacker_actions = attacker_payoffs.shape[1]
        self.defender_action = np.random.choice(self.num_defender_actions)
        self.attacker_action = None
        self.attacker_payoff = 0
        self.defender_payoff = 0
        self.done = False
        self.state = self.get_state()

        # Define action and observation spaces
        self.action_space = spaces.MultiDiscrete([2, 2])
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,))
        

        self.a = a
        self.b = b
        self.c = a0
        self.d = a1
        self.l = l
        self.ci1 = ci1
        self.cm1 = cm1
        self.cm2 = cm2
        self.steps = 0
        self.max_steps = 100  # Set maximum number of steps

    def reset(self):
        self.defender_action = None
        self.attacker_action = None
        self.attacker_payoff = 0
        self.defender_payoff = np.array([[self.a, self.b], [self.c, self.d]])
        self.done = False
        self.state = self.get_state()

    def get_state(self):
        return np.concatenate((self.defender_payoffs.flatten(), self.attacker_payoffs.flatten()))

    def step(self, action):
        if isinstance(action, int):
            action = (action,)

        attacker_action, defender_action = action
        assert self.action_space.contains(action)

        self.defender_action = defender_action
        self.attacker_action = attacker_action

        self.attacker_payoff = self.attacker_payoffs[attacker_action].dot(self.defender_action)
        self.defender_payoff = self.defender_payoffs[defender_action][attacker_action]

        self.state = self.get_state()

        self.steps += 1
        done = self.steps >= self.max_steps

        return self.state, self.defender_payoff - self.attacker_payoff, done, {}


    
    def reset_defender(self):
        self.defender_action = np.random.choice(self.num_defender_actions)
        self.attacker_payoff = self.attacker_payoffs[self.attacker_action].dot(self.defender_action)
        self.defender_payoff = self.defender_payoffs.dot(self.attacker_action)[self.defender_action]
        self.done = False
        self.state = self.get_state()
        return self.state

# Train the NashDQN agent
state_dim = defender_payoffs.size + attacker_payoffs.size
action_dim = attacker_payoffs.shape[0]
nash_dqn = NashDQN(state_dim, action_dim)
env = MalwareEnv(defender_payoffs, attacker_payoffs, cm1=0.15, cm2=0.16, ci1=0.17, ci2=0.48, l=0.25, a=0, b=0.2, a0=0, a1=0, a2=0, a3=0)
nash_dqn.train(env)

# Print the Nash equilibrium strategies and payoffs
defender_strategy = np.zeros_like(defender_payoffs)
attacker_strategy = np.zeros_like(attacker_payoffs)
for i in range(defender_payoffs.shape[0]):
    state = np.concatenate((defender_payoffs[i], attacker_payoffs.flatten()))
    defender_strategy[i] = np.argmax(nash_dqn.model.predict(state.reshape(1, -1))[0])
for i in range(attacker_payoffs.shape[0]):
    state = np.concatenate((defender_payoffs.flatten(), attacker_payoffs[i]))
    attacker_strategy[i] = np.argmax(nash_dqn.model.predict(state.reshape(1, -1))[0])
defender_payoff = defender_payoffs.dot(attacker_strategy)[defender_strategy]
attacker_payoff = attacker_payoffs[attacker_strategy].dot(defender_strategy)
print("Defender strategy:", defender_strategy)
print("Attacker strategy:", attacker_strategy)
print("Defender payoff:", defender_payoff)
print("Attacker payoff:", attacker_payoff)

