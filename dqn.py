import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

tf.compat.v1.enable_v2_behavior()


class DQN:

    def __init__(
        self,
        environment: str = 'Seaquest-ram-v0',
        num_iterations: int = 20000,
        init_collect_steps: int = 1000,
        collect_steps_per_iteration: int = 1,
        replay_buffer_max_length: int = 100000,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        log_interval: int = 200,
        num_eval_episodes: int = 10,
        eval_interval: int = 1000
        ) -> None:

        # Initialize hyperparameters

        self.num_iterations = num_iterations
        
        self.init_collect_steps = init_collect_steps
        self.collect_steps_per_iteration = collect_steps_per_iteration
        self.replay_buffer_max_length = replay_buffer_max_length

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.log_interval = log_interval

        self.num_eval_episodes = num_eval_episodes
        self.eval_interval = eval_interval

        # create the OpenAI Gym training/evaluation environments

        self.env_train = suite_gym.load(environment)
        self.env_eval = suite_gym.load(environment)

        self.train_env = tf_py_environment.TFPyEnvironment(self.env_train)
        self.eval_env = tf_py_environment.TFPyEnvironment(self.env_eval)

        # Instantiate a Deep Q Network using TensorFlow DQN Agent

        fc_layer_params = (100,)

        self.q_net = q_network.QNetwork(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            fc_layer_params=fc_layer_params
        )

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.train_step_counter = tf.Variable(0)

        self.agent = dqn_agent.DqnAgent(
            self.train_env.time_step_spec(),
            self.train_env.action_spec(),
            q_network=self.q_net,
            optimizer=self.optimizer,
            epsilon_greedy = 0.8,
            td_errors_loss_fn=common.element_wise_squared_loss,
            gamma=0.95,
            train_step_counter=self.train_step_counter
        )

        self.agent.initialize()

        self.eval_policy = self.agent.policy
        self.collect_policy = self.agent.collect_policy

        self.random_policy = random_tf_policy.RandomTFPolicy(self.train_env.time_step_spec(),
                                                             self.train_env.action_spec())

        # Collect initial data from training environment

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=self.replay_buffer_max_length
        )

        self._collect_data(self.train_env, 
                           self.random_policy,
                           self.replay_buffer,
                           steps=100)

        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self.batch_size,
            num_steps=2
        ).prefetch(3)

        self.iterator = iter(dataset)

    def _compute_avg_reward(self, env, policy, num_episodes: int = 10) -> float:
        """
        Apply the Policy on the evaluation environment.
        Collect rewards after each action, and compute
        the average value over multiple episodes.
        """

        total_reward = 0.0
        for i in range(num_episodes):
            time_step = env.reset()
            ep_reward = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = env.step(action_step.action)
                if np.abs(time_step.reward) > 0:
                    ep_reward += np.divide(time_step.reward, np.abs(time_step.reward))

            total_reward += ep_reward

        avg_reward = total_reward / num_episodes
        return avg_reward

    def _collect_step(self, env, policy, buffer):
        """
        Gather trajectory data for the Deep Q-Learning 
        process. These data points are used solely for
        the training of the Deep Q Network.
        """

        time_step = env.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = env.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        buffer.add_batch(traj)

    def _collect_data(self, env, policy, buffer, steps):
        for i in range(steps):
            self._collect_step(env, policy, buffer)

    def train(self):
        """
        Formulate the main training loop for the Deep
        Q Network. At each iteration, the DQN collects
        data from the environment to estimate a Q value.
        These Q values are used in the evaluation
        environment after a given number of iterations
        to calculate the performance (reward) of the 
        current algorithm.
        """

        self.agent.train = common.function(self.agent.train)

        self.agent.train_step_counter.assign(0)

        avg_reward = self._compute_avg_reward(self.eval_env,
                                              self.agent.policy,
                                              self.num_eval_episodes)
        print(f"Initial Return: {avg_reward}")
        
        self.returns = [avg_reward]

        # self.display_policy_eval_video()

        for i in range(self.num_iterations):

            for j in range(self.collect_steps_per_iteration):
                self._collect_step(self.train_env,
                                   self.agent.collect_policy,
                                   self.replay_buffer)
                
            exp, info = next(self.iterator)
            train_loss = self.agent.train(exp).loss

            step = self.agent.train_step_counter.numpy()

            if step % self.log_interval == 0:
                print(f"Step {step}: loss = {train_loss}")

            if step % self.eval_interval == 0:
                avg_reward = self._compute_avg_reward(self.eval_env,
                                                     self.agent.policy,
                                                     self.num_eval_episodes)
                print(f"Average Return ({step}): {avg_reward}")
                self.returns.append(avg_reward)
                # self.display_policy_eval_video()

    def plot_return(self):
        iterations = range(0, self.num_iterations + 1, self.eval_interval)

        plt.plot(iterations, self.returns)
        plt.xlabel('Iterations')
        plt.ylabel('Average Return')
        plt.show()

    def _embed_mp4(self, fname):
        video = open(fname, 'rb').read()
        b64 = base64.b64encode(video)
        tag = '''
        <video width="640" height="480" controls>
            <source src="data:video/mp4;base64,{0}" type="video/mp4">
        Your browser does not support the video tag.
        <video/>'''.format(b64.decode())

        return IPython.display.HTML(tag)

    def create_policy_eval_video(self, filename, num_episodes=5, fps=30):
        filename = filename + ".mp4"
        with imageio.get_writer(filename, fps=fps) as video:
            for i in range(num_episodes):
                time_step = self.eval_env.reset()
                video.append_data(self.env_eval.render())

                while not time_step.is_last():
                    action_step = self.agent.policy.action(time_step)
                    time_step = self.eval_env.step(action_step.action)
                    video.append_data(self.env_eval.render())
        return self._embed_mp4(filename)

    def display_policy_eval_video(self, num_episodes=5):
        for i in range(num_episodes):
            time_step = self.eval_env.reset()

            while not time_step.is_last():
                action_step = self.agent.policy.action(time_step)
                time_step = self.eval_env.step(action_step.action)
                self.env_eval.render()
