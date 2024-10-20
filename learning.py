import random
import numpy as np
from keras.src.layers import Dense
from keras.src.models import Sequential


class DQN:
    """Deep Q-Learning model for the agent.

    Attributes:
        state_dim (int): The dimension of the state space.
        action_dim (int): The dimension of the action space.
        memory (list): Memory for storing experiences.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Exploration rate for epsilon-greedy action selection.
        epsilon_min (float): Minimum exploration rate.
        epsilon_decay (float): Decay rate for exploration.
        model (Sequential): The neural network model for Q-value prediction.
        target_model (Sequential): The target model for stability in training.
    """

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = []
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.update_target_model()
        self.rewards = []
        self.actions_taken = []
        self.q_values = []

    def create_model(self):
        """Create and compile the neural network model.

        Returns:
            Sequential: The compiled Keras model.
        """
        model = Sequential()
        model.add(Dense(64, activation="relu", input_dim=self.state_dim))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(self.action_dim))
        model.compile(loss="mse", optimizer="adam")
        return model

    def update_target_model(self):
        """Update the target model with weights from the primary model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store the experience in memory.

        Args:
            state (np.array): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (np.array): The next state after action.
            done (bool): Whether the episode is finished.
        """
        self.memory.append((state, action, reward, next_state, done))
        self.rewards.append(reward)
        self.actions_taken.append(action)

    def replay(self, batch_size):
        """Train the model using a batch of experiences from memory.

        Args:
            batch_size (int): Number of experiences to sample from memory.
        """
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.target_model.predict(next_state)[0]
                )
            target_f = self.model.predict(state)
            self.q_values.append(np.max(target_f))
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def act(self, state):
        """Choose an action based on the current state using epsilon-greedy policy.

        Args:
            state (np.array): The current state.

        Returns:
            int: The chosen action.
        """
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def decay_epsilon(self):
        """Decay the exploration rate to reduce exploration over time."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
