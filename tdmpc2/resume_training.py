import argparse
from pathlib import Path
import torch
from omegaconf import OmegaConf

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

	# Restore model + optimizers
	agent.model.load_state_dict(ckpt["model"])
	agent.optim.load_state_dict(ckpt["optim"])
	agent.pi_optim.load_state_dict(ckpt["pi_optim"])

	# Restore replay buffer
	buffer.load_from_tensordict(ckpt["buffer"])

	print(f"[Resume] Restored {ckpt['num_eps']} episodes")
	print(f"[Resume] Will resume from step {ckpt['step']}")
	return ckpt["step"]


def main():
	parser = argparse.ArgumentParser(description="Resume TD-MPC2 training")
	parser.add_argument("--run-dir", required=True,
						help="Original Hydra run directory (e.g., outputs/<job>_<seed>)")
	parser.add_argument("--checkpoint", required=True,
						help="Path to checkpoint_XXXXXX.pt")
	args = parser.parse_args()

	run_dir = Path(args.run_dir)
	cfg_path = run_dir / "resolved_config.yaml"

	if not cfg_path.exists():
		raise FileNotFoundError(
			f"Resolved config not found: {cfg_path}\n"
			"Make sure you saved it during training via OmegaConf.save."
		)

	if not Path(args.checkpoint).exists():
		raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

	print(f"[Resume] Loading resolved config: {cfg_path}")
	cfg = OmegaConf.load(cfg_path)

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

	start_step = load_checkpoint(args.checkpoint, agent, buffer)
	trainer._step = start_step

	print(f"[Resume] Resuming training at step {start_step}...")
	trainer.train(pretrain=False)


if __name__ == "__main__":
	main()
