from gymenv_qsimpy import QSimPyEnv
from env_wrapper import ScaleQSimPyEnv
from gymnasium.wrappers import TransformObservation
import numpy as np


def qsimpy_env_creator(env_config):
    dataset = env_config.pop("dataset", None)
    config = env_config.pop("config", None)
    config = config if config is not None else {}

    if dataset is None:
        raise ValueError("Dataset is not specified")

    env = QSimPyEnv(dataset=dataset, config=config)

    obs_filter = env_config.pop("obs_filter", None)
    reward_filter = env_config.pop("reward_filter", None)

    # -----------------------------
    # OBSERVATION FILTER
    # -----------------------------
    if obs_filter is not None:
        if obs_filter == "rescale_-1_1":
            # first cast to float32
            env = TransformObservation(env, lambda obs: obs.astype(np.float32))

            # rescale to [-1, 1]
            def _scale(obs):
                max_abs = np.max(np.abs(obs)) if np.max(np.abs(obs)) != 0 else 1
                return (obs / max_abs).astype(np.float32)

            env = TransformObservation(env, _scale)

    # -----------------------------
    # REWARD FILTER
    # -----------------------------
    if reward_filter is not None:
        if reward_filter == "scale_2x":
            env = ScaleQSimPyEnv(env, scale=env_config.pop("reward_scale", 2))

    return env
