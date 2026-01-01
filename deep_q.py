import os
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DeepQModel(tf.keras.Model):

    def __init__(self, state_size, num_actions):
        super(DeepQModel, self).__init__()
        self.num_actions = num_actions
        self.state_size = state_size

        # TODO: Define network parameters and optimizer

        # We require that you use tf.keras.Sequential to define the model and call it self.model
        #   (This is for auto-grading purposes)
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(shape=(state_size,)))
        self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.model.add(tf.keras.layers.Dense(num_actions))

        learning_rate = 1e-3
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        
        # We require that you call your target model self.target_model
        #    (This is for auto-grading purposes)
        #    Hints: You can clone the model using tf.keras.models.clone_model
        #           You can get the weights of model using get_weights
        #           You can set the weights of target_model using set_weights

        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        

    def call(self, states):
        return self.target_model(states)
    

    def loss_func(self, batch, discount_factor = 0.99):
        # Compute the loss for the agent
        
        batch_states = batch[0]
        batch_actions = batch[1]
        batch_rewards = batch[2]
        batch_next_states = batch[3]
        batch_done = batch[4]
        
        Q = self.model(batch_states)
        Q_prime = self.target_model(batch_next_states)
        max_Q_prime = tf.reduce_max(Q_prime, axis=1)

        target_Q = batch_rewards + (1-tf.cast(batch_done, tf.float32)) * discount_factor * max_Q_prime
        
        action_mask = tf.one_hot(batch_actions, self.num_actions)
        chosen_Q = tf.reduce_sum(Q * action_mask, axis=1)
        
        loss = tf.reduce_mean(tf.square(chosen_Q - target_Q))
        return loss
