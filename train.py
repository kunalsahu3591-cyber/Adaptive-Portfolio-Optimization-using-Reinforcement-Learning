"""
Training Pipeline for DDPG Reinforcement Learning Agent on Adaptive Portfolio Allocation.
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf

from tf_agents.environments import tf_py_environment
from tf_agents.agents.ddpg import actor_network, critic_network, ddpg_agent
from tf_agents.policies import random_tf_policy
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

import config
from environments import PortfolioEnv
from utils import compute_avg_return, collect_step, collect_data

tf.compat.v1.enable_v2_behavior()


def setup_directories():
    os.makedirs(config.LOGDIR, exist_ok=True)
    os.makedirs(config.MODEL_SAVE, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(config.LOGDIR, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )


def train_ddpg(num_iterations=None, eval_interval=None, log_interval=None):
    num_iterations = num_iterations or config.NUM_ITERATIONS
    eval_interval = eval_interval or config.EVAL_INTERVAL
    log_interval = log_interval or config.LOG_INTERVAL

    setup_directories()

    # Check data availability
    if not os.path.exists(config.FILE):
        print(f"Preprocessed file '{config.FILE}' not found.")
        if os.path.exists(config.RAW_FILE):
            print("Found raw data file. Running pre_process.py...")
            from pre_process import preprocess_data
            preprocess_data()
        else:
            print("Generating synthetic sample data and preprocessing...")
            from generate_sample_data import generate_market_data
            from pre_process import preprocess_data
            generate_market_data()
            preprocess_data()

    print("Initializing RL Environments...")
    train_py_env = PortfolioEnv()
    eval_py_env = PortfolioEnv()

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # Actor & Critic Networks
    actor_net = actor_network.ActorNetwork(
        train_env.time_step_spec().observation,
        train_env.action_spec(),
        fc_layer_params=config.actor_fc_layers,
    )

    critic_net_input_specs = (
        train_env.time_step_spec().observation,
        train_env.action_spec()
    )

    critic_net = critic_network.CriticNetwork(
        critic_net_input_specs,
        observation_fc_layer_params=config.critic_obs_fc_layers,
        action_fc_layer_params=config.critic_action_fc_layers,
        joint_fc_layer_params=config.critic_joint_fc_layers,
    )

    global_step = tf.compat.v1.train.get_or_create_global_step()

    tf_agent = ddpg_agent.DdpgAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=config.actor_learning_rate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=config.critic_learning_rate),
        ou_stddev=config.ou_stddev,
        ou_damping=config.ou_damping,
        target_update_tau=config.target_update_tau,
        target_update_period=config.target_update_period,
        dqda_clipping=config.dqda_clipping,
        td_errors_loss_fn=config.td_errors_loss_fn,
        gamma=config.gamma,
        reward_scale_factor=config.reward_scale_factor,
        gradient_clipping=config.gradient_clipping,
        debug_summaries=config.debug_summaries,
        summarize_grads_and_vars=config.summarize_grads_and_vars,
        train_step_counter=global_step
    )
    tf_agent.initialize()

    # Initial exploration & replay buffer
    random_policy = random_tf_policy.RandomTFPolicy(
        train_env.time_step_spec(),
        train_env.action_spec()
    )

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=config.REPLAY_BUFFER_MAX_LENGTH
    )

    print("Collecting initial warmup transitions with random policy...")
    collect_data(train_env, random_policy, replay_buffer, steps=100)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=config.BATCH_SIZE,
        num_steps=2
    ).prefetch(3)

    saver = PolicySaver(tf_agent.collect_policy, batch_size=None)
    iterator = iter(dataset)
    tf_agent.train = common.function(tf_agent.train)
    tf_agent.train_step_counter.assign(0)

    # Pre-training baseline evaluation
    avg_return = compute_avg_return(eval_env, tf_agent.policy, config.NUM_EVAL_EPISODES)
    returns = [avg_return]
    iterations = [0]
    print(f"Initial Average Return: {avg_return:.4f}")

    print(f"Starting training loop for {num_iterations} iterations...")
    for _ in tqdm(range(num_iterations), total=num_iterations, desc="Training DDPG"):
        # Step collection
        for _ in range(config.COLLECT_STEPS_PER_ITERATION):
            collect_step(train_env, tf_agent.collect_policy, replay_buffer)

        # Optimize agent networks
        experience, _ = next(iterator)
        train_loss = tf_agent.train(experience).loss
        step = int(tf_agent.train_step_counter.numpy())

        if step % log_interval == 0:
            logging.info(f"Step {step} | Loss: {train_loss:.6f}")

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, tf_agent.policy, config.NUM_EVAL_EPISODES)
            print(f"Step {step} | Avg Return: {avg_return:.4f} | Loss: {train_loss:.6f}")
            logging.info(f"Step {step} | Avg Return: {avg_return:.4f}")
            returns.append(avg_return)
            iterations.append(step)

        if step % config.MODEL_SAVE_FREQ == 0:
            model_path = os.path.join(config.MODEL_SAVE, f'policy_step_{step}')
            saver.save(model_path)

    # Save outputs & training curves
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, returns, label="DDPG Average Return", color="#2563eb", linewidth=2)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.title('DDPG Agent Portfolio Performance')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig("output_img_gamma.png", bbox_inches='tight')
    plt.close()

    result_df = pd.DataFrame({"iterations": iterations, "return": returns})
    result_df.to_csv(os.path.join(config.LOGDIR, "output_ar_gamma.csv"), index=False)
    print("Training finished! Results saved to LOGDIR/output_ar_gamma.csv and output_img_gamma.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DDPG agent on portfolio allocation.")
    parser.add_argument("--iterations", type=int, default=config.NUM_ITERATIONS, help="Number of training iterations.")
    parser.add_argument("--eval-interval", type=int, default=config.EVAL_INTERVAL, help="Evaluation interval.")
    parser.add_argument("--log-interval", type=int, default=config.LOG_INTERVAL, help="Logging interval.")
    args = parser.parse_args()

    train_ddpg(args.iterations, args.eval_interval, args.log_interval)
