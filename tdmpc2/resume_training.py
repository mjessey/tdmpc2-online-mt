import argparse
import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.core.global_hydra import GlobalHydra
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


def hydra_initialize_for_resume():
	"""Initialize Hydra manually so get_original_cwd() works."""
	if not GlobalHydra().is_initialized():
		GlobalHydra().clear()
		hydra.initialize(config_path=".")
		# Required to avoid HydraConfig errors
		HydraConfig().set_config(OmegaConf.create())


def load_checkpoint(path, agent, buffer):
	ckpt = torch.load(path, map_location="cpu")
	agent.model.load_state_dict(ckpt["model"])
	agent.optim.load_state_dict(ckpt["optim"])
	agent.pi_optim.load_state_dict(ckpt["pi_optim"])
	buffer.load_from_tensordict(ckpt["buffer"])
	return ckpt["step"]


def main():
	parser = argparse.ArgumentParser(description="Resume TD-MPC2 training")
	parser.add_argument("--cfg", required=True)
	parser.add_argument("--checkpoint", required=True)
	args = parser.parse_args()

	# --------------------------
	# Initialize Hydra manually
	# --------------------------
	hydra_initialize_for_resume()

	# --------------------------
	# Load saved Hydra config
	# --------------------------
	saved_cfg = OmegaConf.load(args.cfg)
	cfg = parse_cfg(saved_cfg)

	# --------------------------
	# Load training components
	# --------------------------
	set_seed(cfg.seed)
	env = make_env(cfg)
	agent = TDMPC2(cfg)
	buffer = Buffer(cfg)
	logger = Logger(cfg)

	trainer_cls = OnlineTrainerMultitask if cfg.multitask else OnlineTrainer
	trainer = trainer_cls(cfg, env, agent, buffer, logger)

	# --------------------------
	# Load checkpoint
	# --------------------------
	step = load_checkpoint(args.checkpoint, agent, buffer)
	trainer._step = step

	print(f"[Resume] Resuming at step {step}")
	trainer.train(pretrain=False)


if __name__ == "__main__":
	main()
