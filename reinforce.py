import os
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class Reinforce(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The Reinforce class that inherits from tf.keras.Model
        The forward pass calculates the policy for the agent given a batch of states.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but having this parameter may streamline your implementation.
        :param num_actions: number of actions in an environment. You do need to use this in your implementation.
        """
        super(Reinforce, self).__init__()
        self.num_actions = num_actions
        self.state_size = state_size
        # TODO: Define network parameters and optimizer
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(shape=(state_size,)))
        self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.model.add(tf.keras.layers.Dense(num_actions))

        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-5)


    def call(self, states):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.

        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [episode_length,num_actions] matrix representing the probability distribution over actions
        for each state in the episode
        """
        # TODO: implement this ~
        if len(states.shape) == 1:
            states = tf.expand_dims(states, 0)

        logits =self.model(states)
        return tf.nn.softmax(logits, axis=-1)

    def loss_func(self, states, actions, discounted_rewards):
        """
        Computes the loss for the agent. Make sure to understand the handout clearly when implementing this.

        :param states: A batch of states of shape [episode_length, state_size]
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a Tensorflow scalar
        """
        # TODO: implement this
        # Hint: Use gather_nd to get the probability of each action that was actually taken in the episode

        action_probs = self.call(states) 

        timesteps = tf.range(len(actions))
        indices = tf.stack([timesteps, actions], axis=1)  # [[0,a0], [1,a1], ...]
        taken_action_probs = tf.gather_nd(action_probs, indices)
        log_probs = tf.math.log(taken_action_probs + 1e-8)

        return -tf.reduce_sum(log_probs * discounted_rewards)
