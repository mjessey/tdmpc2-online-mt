import numpy as np
import gymnasium as gym
from envs.wrappers.timeout import Timeout

from sai_rl import SAIClient

SAI_TASKS = {
    "sai_goalie_penalty_kick": 0,
    "sai_obstical_penalty_kick": 1,
    "sai_kick_to_target": 2,
}


class Preprocessor:
    def modify_state(self, obs, info):
        return obs


class SAIWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.pre_proc = Preprocessor()

    def reset(self):
        return self.env.reset()[0]

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(
            self.action_function(action)
        )

        done = terminated or truncated
        info["terminated"] = terminated

        obs = self.pre_proc.modify_state(obs, info)

        return obs, reward, done, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def action_function(self, action):
        action = action.clamp(-1, 1)

        mi = self.env.action_space.low
        ma = self.env.action_space.high

        normalized = (action + 1) / (2) * (ma - mi) + mi

        return normalized


def make_env(cfg):
    """
    Make classic/MuJoCo environment.
    """
    if cfg.task not in SAI_TASKS:
        raise ValueError("Unknown task:", cfg.task)

    sai = SAIClient(comp_id="lower-t1-penalty-kick-goalie")

    env = sai.make_env(SAI_TASKS[cfg.task], render_mode="rgb_array")

    env = SAIWrapper(env, cfg)

    env = Timeout(
        env,
        max_episode_steps=5000,
    )

    return env
