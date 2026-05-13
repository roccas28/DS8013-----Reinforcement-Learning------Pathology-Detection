import numpy as np
import tensorflow as tf
from collections import deque
import random
import model as mdl

class DQNTrainer:   # Handles the Replay Buffer and Training Loop for the Agent -> Epsilon-Greedy Exploration Strat -> TD -> Huber Loss optimization
    
    def __init__(self, action_size=7):
        self.action_size = action_size
        self.memory = deque(maxlen=10000) # Experience Replay Buffer
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9998 # 0.1 @ ~ 11500 ep -> Exploit then
        self.batch_size = 32
        
        self.main_network = mdl.build_dqn_agent(action_size=self.action_size)
        self.target_network = mdl.build_dqn_agent(action_size=self.action_size)
        self.target_network.set_weights(self.main_network.get_weights())
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
        self.loss_fn = tf.keras.losses.Huber() # Huber loss for stability
        
    def remember(self, state, action, reward, next_state, done): # Store memory in buffer
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, explore=True): # Epsilon-greedy action selection
        # Exploring allowed -> roll dice with ep
        if explore and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Otherwise, exploit best known A
        q_values = self.main_network.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])
        
    def replay(self): # Double Q-Learning Update
        if len(self.memory) < self.batch_size:
            return
            
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([m[0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])
        
        # Double Q-Learning logic
        next_actions = np.argmax(self.main_network.predict(next_states, verbose=0), axis=1)
        target_q_next = self.target_network.predict(next_states, verbose=0)
        
        targets = rewards + self.gamma * target_q_next[np.arange(self.batch_size), next_actions] * (1 - dones)
        
        with tf.GradientTape() as tape:
            q_values = self.main_network(states)
            action_masks = tf.one_hot(actions, self.action_size)
            q_values_for_actions_taken = tf.reduce_sum(tf.multiply(q_values, action_masks), axis=1)
            loss = self.loss_fn(targets, q_values_for_actions_taken)
            
        grads = tape.gradient(loss, self.main_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.main_network.trainable_variables))

        return float(loss)

    def decay_epsilon(self): # Exponential decay -> called within main to ensure target met ( ~ 11500 for 15000)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_network(self): # Sync target network weights
        self.target_network.set_weights(self.main_network.get_weights())
