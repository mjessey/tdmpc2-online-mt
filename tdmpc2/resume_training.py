import argparse
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
from pathlib import Path
import torch

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
	parser.add_argument("--run-dir", required=True)
	parser.add_argument("--checkpoint", required=True)
	args = parser.parse_args()

	run_dir = Path(args.run_dir)
	overrides_path = run_dir / ".hydra" / "overrides.yaml"

	# Load the overrides Hydra saved during training
	overrides = OmegaConf.load(overrides_path)

	# Initialize Hydra with original config directory
	with initialize(version_base=None, config_path="."):
		# Compose a fresh, resolved config using saved overrides
		cfg = compose(config_name="config", overrides=overrides)

	# Now cfg has real values (no ???)
	cfg = parse_cfg(cfg)

	# Recreate full training state
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
