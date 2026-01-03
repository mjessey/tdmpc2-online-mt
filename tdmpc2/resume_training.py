import argparse
import torch
from pathlib import Path

# TD-MPC2 imports (adjust paths if needed)
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
# Load checkpoint
# -----------------------------------------------------------------------------
def load_checkpoint(path, agent, buffer):
	print(f"[Resume] Loading checkpoint from: {path}")
	ckpt = torch.load(path, map_location="cpu")

	# Restore model
	agent.model.load_state_dict(ckpt["model"])

	# Restore optimizers
	agent.optim.load_state_dict(ckpt["optim"])
	agent.pi_optim.load_state_dict(ckpt["pi_optim"])

	# Restore replay buffer
	buffer.load_from_tensordict(ckpt["buffer"])

	print(f"[Resume] Restored {ckpt['num_eps']} episodes")
	print(f"[Resume] Restored step to {ckpt['step']}")

	return ckpt["step"]


# -----------------------------------------------------------------------------
# MAIN RESUME LOGIC
# -----------------------------------------------------------------------------
def main():
	parser = argparse.ArgumentParser(description="Resume TD-MPC2 training")
	parser.add_argument("--cfg", type=str, required=True,
						help="Path to the Hydra-generated config file used for training")
	parser.add_argument("--checkpoint", type=str, required=True,
						help="Path to checkpoint_XXXXXX.pt")
	args = parser.parse_args()

	cfg_path = Path(args.cfg)
	ckpt_path = Path(args.checkpoint)

	if not cfg_path.exists():
		raise FileNotFoundError(f"Config not found: {cfg_path}")
	if not ckpt_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

	print(f"[Resume] Loading Hydra config: {cfg_path}")

	# Hydra configs are YAML, but they need parse_cfg()
	import yaml
	with open(cfg_path, "r") as f:
		raw_cfg = yaml.safe_load(f)

	cfg = parse_cfg(raw_cfg)

	# Must set seed here to ensure determinism
	set_seed(cfg.seed)

	# --- Recreate all components exactly like train.py ---
	print("[Resume] Creating environment...")
	env = make_env(cfg)

	print("[Resume] Creating agent...")
	agent = TDMPC2(cfg)

	print("[Resume] Creating buffer...")
	buffer = Buffer(cfg)

	print("[Resume] Creating logger...")
	logger = Logger(cfg)

	# Pick correct trainer class
	if cfg.multitask:
		trainer_cls = OnlineTrainerMultitask
	else:
		trainer_cls = OnlineTrainer

	print("[Resume] Creating trainer...")
	trainer = trainer_cls(
		cfg=cfg,
		env=env,
		agent=agent,
		buffer=buffer,
		logger=logger,
	)

	# --- Load checkpoint ---
	start_step = load_checkpoint(ckpt_path, agent, buffer)

	# --- Set step count in trainer ---
	trainer._step = start_step

	print(f"[Resume] Resuming training at step {start_step}...")
	trainer.train()


if __name__ == "__main__":
	main()
