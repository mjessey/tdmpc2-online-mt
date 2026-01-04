import argparse
import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import torch

from common.seed import set_seed
from common.buffer import Buffer
from common.logger import Logger
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.online_trainer_multitask import OnlineTrainerMultitask
from trainer.online_trainer import OnlineTrainer


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

	run_dir = args.run_dir

	# Load overrides.yaml
	overrides = OmegaConf.load(f"{run_dir}/.hydra/overrides.yaml")

	# Initialize Hydra in the original working directory
	with initialize(version_base=None, config_path="."):
		cfg = compose(config_name="config", overrides=overrides)

	# Now cfg is fully resolved (no ???)
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
