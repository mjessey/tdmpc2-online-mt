import argparse
import torch
from pathlib import Path
from omegaconf import OmegaConf

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from common.logger import Logger
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.online_trainer import OnlineTrainer
from trainer.online_trainer_multitask import OnlineTrainerMultitask


def load_checkpoint(path, agent, buffer):
    ckpt = torch.load(path, map_location="cpu")
    agent.model.load_state_dict(ckpt["model"])
    agent.optim.load_state_dict(ckpt["optim"])
    agent.pi_optim.load_state_dict(ckpt["pi_optim"])
    buffer.load_from_tensordict(ckpt["buffer"])
    return ckpt["step"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    saved_cfg = OmegaConf.load(args.cfg)

    # Workaround Hydra absence:
    saved_cfg.work_dir = str(Path(args.cfg).parents[1])

    cfg = parse_cfg(saved_cfg)

    set_seed(cfg.seed)
    env = make_env(cfg)
    agent = TDMPC2(cfg)
    buffer = Buffer(cfg)
    logger = Logger(cfg)

    trainer_cls = OnlineTrainerMultitask if cfg.multitask else OnlineTrainer
    trainer = trainer_cls(cfg, env, agent, buffer, logger)

    trainer._step = load_checkpoint(args.checkpoint, agent, buffer)

    trainer.train(pretrain=False)


if __name__ == "__main__":
    main()
