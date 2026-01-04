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
def main(dummy_cfg):
	# Use custom argument names to avoid Hydra conflicts
	parser = argparse.ArgumentParser()
	parser.add_argument("--resume-cfg", required=True)
	parser.add_argument("--resume-checkpoint", required=True)
	args, unknown = parser.parse_known_args()

	cfg_path = Path(args.resume_cfg)
	ckpt_path = Path(args.resume_checkpoint)

	print(f"[Resume] Loading Hydra config: {cfg_path}")
	saved_cfg = OmegaConf.load(cfg_path)
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
	trainer = trainer_cls(cfg=cfg, env=env, agent=agent, buffer=buffer, logger=logger)

	start_step = load_checkpoint(ckpt_path, agent, buffer)
	trainer._step = start_step

	print(f"[Resume] Resuming training at step {start_step}...")
	trainer.train(pretrain=False)


if __name__ == "__main__":
	main()
