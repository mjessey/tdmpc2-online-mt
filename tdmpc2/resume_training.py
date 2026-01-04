import argparse
import hydra
from pathlib import Path
from omegaconf import OmegaConf
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
	print(f"[Resume] Loading checkpoint from: {path}")
	ckpt = torch.load(path, map_location="cpu")

	agent.model.load_state_dict(ckpt["model"])
	agent.optim.load_state_dict(ckpt["optim"])
	agent.pi_optim.load_state_dict(ckpt["pi_optim"])
	buffer.load_from_tensordict(ckpt["buffer"])

	print(f"[Resume] Restored {ckpt['num_eps']} episodes")
	print(f"[Resume] Restored step to {ckpt['step']}")
	return ckpt["step"]


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(_hydra_cfg):
	parser = argparse.ArgumentParser()
	parser.add_argument("--cfg", required=True)
	parser.add_argument("--checkpoint", required=True)
	args, unknown = parser.parse_known_args()

	# Load the REAL training config (not Hydra's)
	saved_cfg = OmegaConf.load(args.cfg)
	cfg = parse_cfg(saved_cfg)

	# Set seed
	set_seed(cfg.seed)

	env = make_env(cfg)
	agent = TDMPC2(cfg)
	buffer = Buffer(cfg)
	logger = Logger(cfg)

	trainer_cls = OnlineTrainerMultitask if cfg.multitask else OnlineTrainer
	trainer = trainer_cls(cfg, env, agent, buffer, logger)

	start_step = load_checkpoint(args.checkpoint, agent, buffer)
	trainer._step = start_step

	trainer.train(pretrain=False)


if __name__ == "__main__":
	main()
