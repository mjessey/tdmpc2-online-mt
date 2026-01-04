import hydra
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
	print(f"[Resume] Loading checkpoint: {path}")
	ckpt = torch.load(path, map_location="cpu")

	agent.model.load_state_dict(ckpt["model"])
	agent.optim.load_state_dict(ckpt["optim"])
	agent.pi_optim.load_state_dict(ckpt["pi_optim"])
	buffer.load_from_tensordict(ckpt["buffer"])

	print(f"[Resume] Loaded {ckpt['num_eps']} episodes")
	print(f"[Resume] Resuming from step {ckpt['step']}")
	return ckpt["step"]


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(hcfg):
	# Hydra automatically injects fields like resume_cfg, resume_checkpoint
	if "resume_cfg" not in hcfg or "resume_checkpoint" not in hcfg:
		raise ValueError("Must pass resume_cfg=... and resume_checkpoint=...")

	cfg_path = Path(hcfg.resume_cfg)
	ckpt_path = Path(hcfg.resume_checkpoint)

	print(f"[Resume] Loading saved config: {cfg_path}")
	saved_cfg = OmegaConf.load(cfg_path)
	cfg = parse_cfg(saved_cfg)

	set_seed(cfg.seed)

	env = make_env(cfg)
	agent = TDMPC2(cfg)
	buffer = Buffer(cfg)
	logger = Logger(cfg)

	trainer_cls = OnlineTrainerMultitask if cfg.multitask else OnlineTrainer
	trainer = trainer_cls(cfg, env, agent, buffer, logger)

	start_step = load_checkpoint(ckpt_path, agent, buffer)
	trainer._step = start_step

	print(f"[Resume] Training from step {start_step}...")
	trainer.train(pretrain=False)


if __name__ == "__main__":
	main()
