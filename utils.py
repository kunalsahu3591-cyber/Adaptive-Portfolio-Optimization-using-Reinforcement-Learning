"""
Utility functions for RL environment interaction, evaluation, and data collection.
"""

import numpy as np
from tf_agents.trajectories import trajectory


def compute_avg_return(environment, policy, num_episodes=10):
    """
    Computes the average cumulative return across a number of evaluation episodes.
    
    Args:
        environment: TFPyEnvironment or PyEnvironment instance.
        policy: TFPolicy to evaluate.
        num_episodes: Number of episodes to run.
        
    Returns:
        float: Mean cumulative episode return.
    """
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    if hasattr(avg_return, 'numpy'):
        return float(avg_return.numpy()[0])
    return float(avg_return)


def collect_step(environment, policy, buffer):
    """
    Executes a single step in the environment and appends the trajectory transition to the replay buffer.
    
    Args:
        environment: TFPyEnvironment instance.
        policy: Exploration/collection policy.
        buffer: TFUniformReplayBuffer instance.
    """
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    buffer.add_batch(traj)


def collect_data(environment, policy, buffer, steps):
    """
    Runs the collection loop for a fixed number of steps.
    
    Args:
        environment: TFPyEnvironment instance.
        policy: Exploration/collection policy.
        buffer: TFUniformReplayBuffer instance.
        steps: Total steps to collect.
    """
    for _ in range(steps):
        collect_step(environment, policy, buffer)
