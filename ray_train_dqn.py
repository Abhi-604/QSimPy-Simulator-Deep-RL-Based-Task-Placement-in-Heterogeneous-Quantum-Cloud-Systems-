
import argparse
import os

import ray
from ray import tune, air
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn import DQNConfig

from ray.air import CheckpointConfig

from env_creator import qsimpy_env_creator


parser = argparse.ArgumentParser()

parser.add_argument("--num-cpus", type=int, default=0)

parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)

parser.add_argument("--stop-iters", type=int, default=100)
parser.add_argument("--stop-timesteps", type=int, default=100000)

if __name__ == "__main__":
    args = parser.parse_args()

    # Optional
    # ray.init(num_cpus=args.num_cpus or None)

    # Register env
    register_env("QSimPyEnv", qsimpy_env_creator)

    # Replay buffer config
    replay_config = {
        "type": "MultiAgentPrioritizedReplayBuffer",
        "capacity": 60000,
        "prioritized_replay_alpha": 0.5,
        "prioritized_replay_beta": 0.5,
        "prioritized_replay_eps": 3e-6,
    }

    # RLlib DQN config
    config = (
        DQNConfig()
        .framework(args.framework)
        .environment(
            env="QSimPyEnv",
            env_config={
                "obs_filter": "rescale_-1_1",
                "reward_filter": None,
                "dataset": "qdataset/qsimpyds_1000_sub_26.csv",
            },
        )
        .training(
            lr=tune.grid_search([0.01]),
            train_batch_size=tune.grid_search([78]),
            replay_buffer_config=replay_config,
            num_atoms=10,
            n_step=5,
            noisy=True,
            v_min=-10.0,
            v_max=10.0,
        )
        .rollouts(num_rollout_workers=8)
    )

    stop_config = {
        "timesteps_total": args.stop_timesteps,
        "training_iteration": args.stop_iters,
    }

    # Save results
    current_directory = os.getcwd()
    result_directory = os.path.join(current_directory, "results")
    storage_path = f"file://{result_directory}"

    # Main tune call
    results = tune.Tuner(
        "DQN",
        run_config=air.RunConfig(
            stop=stop_config,
            checkpoint_config=CheckpointConfig(
                checkpoint_frequency=10,
                checkpoint_at_end=True,
            ),
            storage_path=storage_path,
            name="DQN_QCE_1000",
        ),
        param_space=config.to_dict(),
    ).fit()

    ray.shutdown()
