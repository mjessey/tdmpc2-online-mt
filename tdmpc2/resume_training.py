import argparse
import torch
from omegaconf import OmegaConf
from pathlib import Path

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

	# No Hydra. No overrides. Simple argparse.
	parser = argparse.ArgumentParser()
	parser.add_argument("--cfg", required=True)
	parser.add_argument("--checkpoint", required=True)
	args = parser.parse_args()

	# Load saved training config
	cfg_path = Path(args.cfg)
	saved_cfg = OmegaConf.load(cfg_path)

	# --- The key fix ---
	# Use the directory containing cfg_path as original working directory
	# This replaces hydra.utils.get_original_cwd()
	saved_cfg.work_dir = cfg_path.parent.parent  # outputs/<job>_<seed>
	saved_cfg.original_cwd = str(cfg_path.parent.parent)

	# Now parse the TD-MPC2 config
	cfg = parse_cfg(saved_cfg)

	# Rebuild system exactly as in training
	set_seed(cfg.seed)
	env = make_env(cfg)
	agent = TDMPC2(cfg)
	buffer = Buffer(cfg)
	logger = Logger(cfg)

	trainer_cls = OnlineTrainerMultitask if cfg.multitask else OnlineTrainer
	trainer = trainer_cls(cfg=cfg, env=env, agent=agent, buffer=buffer, logger=logger)

	# Load checkpoint
	step = load_checkpoint(args.checkpoint, agent, buffer)
	trainer._step = step

	print(f"[Resume] Resuming at step {step}...")
	trainer.train(pretrain=False)


if __name__ == "__main__":
	main()
