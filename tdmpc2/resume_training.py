import argparse
import torch
import hydra
from pathlib import Path
from omegaconf import OmegaConf

# TD-MPC2 imports
from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from common.logger import Logger
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.online_trainer import OnlineTrainer
from trainer.online_trainer_multitask import OnlineTrainerMultitask
from trainer.offline_trainer import OfflineTrainer


# -----------------------------------------------------------------------------
# Checkpoint loading
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Hydra entry point
# -----------------------------------------------------------------------------
@hydra.main(version_base=None, config_path=".", config_name="config")
def main(dummy_cfg):
	parser = argparse.ArgumentParser()
	parser.add_argument("--cfg", required=True)
	parser.add_argument("--checkpoint", required=True)
	args = parser.parse_args()

	# Load saved Hydra config (original training config)
	saved_cfg = OmegaConf.load(args.cfg)
	cfg = parse_cfg(saved_cfg)

	set_seed(cfg.seed)

	print("[Resume] Creating environment...")
	env = make_env(cfg)

	print("[Resume] Creating agent...")
	agent = TDMPC2(cfg)

	print("[Resume] Creating buffer...")
	buffer = Buffer(cfg)

	print("[Resume] Creating logger...")
	logger = Logger(cfg)

	trainer_cls = OnlineTrainerMultitask if cfg.multitask else OnlineTrainer

	print("[Resume] Creating trainer...")
	trainer = trainer_cls(cfg=cfg, env=env, agent=agent, buffer=buffer, logger=logger)

	# Load checkpoint
	start_step = load_checkpoint(args.checkpoint, agent, buffer)
	trainer._step = start_step

	print(f"[Resume] Resuming training at step {start_step}...")
	trainer.train(pretrain=False)


if __name__ == "__main__":
	main()
