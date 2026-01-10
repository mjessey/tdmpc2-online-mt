import torch
import torch.nn.functional as F

import numpy as np

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from common.layers import api_model_conversion
from tensordict import TensorDict


class TDMPC2(torch.nn.Module):
	"""
	TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments,
	and supports both state and pixel observations.
	"""

	def __init__(self, cfg, num_envs=100):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device('cuda:0')
		self.model = WorldModel(cfg).to(self.device)
		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._termination.parameters() if self.cfg.episodic else []},
			{'params': self.model._Qs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []
			 }
		], lr=self.cfg.lr, capturable=True)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		print('Episode length:', cfg.episode_length)
		print('Discount factor:', self.discount)
		#self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))
		self._prev_mean = torch.nn.Parameter(
			torch.zeros(num_envs, cfg.horizon, cfg.action_dim, device=self.device), requires_grad=False
		)
		if cfg.compile:
			print('Compiling update function with torch.compile...')
			self._update = torch.compile(self._update, mode="default")
		self._num_envs = num_envs

	@property
	def plan(self):
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is not None:
			return _plan_val
		if self.cfg.compile:
			plan = torch.compile(self._plan, mode="default")
		else:
			plan = self._plan
		self._plan_val = plan
		return self._plan_val

	def mark_reset(self, env_ids):
		env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
		self._t0_mask[env_ids] = True

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		if isinstance(fp, dict):
			state_dict = fp
		else:
			state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
		state_dict = state_dict["model"] if "model" in state_dict else state_dict
		state_dict = api_model_conversion(self.model.state_dict(), state_dict)
		self.model.load_state_dict(state_dict)
		return

	@torch.no_grad()
	def forward(self, x):
		if x[0, -3]:
			task_idx = 0
		elif x[0, -2]:
			task_idx = 1
		else:
			task_idx = 2

		x = x[:, :-3]
		x = np.hstack((x, np.zeros((100, 54 - x.shape[1]))))
		x = torch.tensor(x, dtype=torch.float32)
		return self.act(x,
				t0=torch.ones(100, dtype=torch.bool, device=self.device),
				eval_mode=True,
				task=task_idx)
    
	@torch.no_grad()
	def _forward(self, x):
		if isinstance(x, np.ndarray):
			task_idx = torch.from_numpy(x[:, -3:].argmax(1)).to(self.device, torch.long)
			state = x[:, :-3]
		elif isinstance(x, torch.Tensor):
			task_idx = x[:, -3:].argmax(1).to(torch.long)
			state = x[:, :-3].cpu().numpy()
		else:
			raise TypeError

		B, F = state.shape
		if F > 54:
			raise ValueError
		if F < 54:
			state = np.hstack((state, np.zeros((B, 54 - F), dtype=state.dtype)))

		obs = torch.as_tensor(state, dtype=torch.float32, device=self.device)

		if not hasattr(self, "_t0_mask"):
			self._t0_mask = torch.ones(B, dtype=torch.bool, device=self.device)
		elif self._t0_mask.shape[0] != B:
			raise RuntimeError

		action = self.act(obs, t0=self._t0_mask, eval_mode=True, task=task_idx.to(self.device))
		#self._t0_mask[:] = False
		return action

	@torch.no_grad()
	def act(self, obs, t0, eval_mode=False, task=None):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (torch.BoolTensor): mask that is True for envs that just reset.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True)
		if task is not None:
			task = torch.tensor([task], device=self.device)
			task = task.repeat(obs.shape[0]).to(self.device)
		if self.cfg.mpc:
			return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task).cpu()
		z = self.model.encode(obs, task)
		action, info = self.model.pi(z, task)
		if eval_mode:
			action = info["mean"]
		return action.cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions, task):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		termination = torch.zeros(z.shape[0], 1, dtype=torch.float32, device=z.device)
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
			z = self.model.next(z, actions[t], task)
			G = G + discount * (1-termination) * reward
			discount_update = self.discount[task] if self.cfg.multitask else self.discount
			discount = discount * discount_update
			if self.cfg.episodic:
				termination = torch.clip(termination + (self.model.termination(z, task) > 0.5).float(), max=1.)
		action, _ = self.model.pi(z, task)
		return G + discount * (1-termination) * self.model.Q(z, action, task, return_type='avg')

	@torch.no_grad()
	def _plan(self, obs, t0, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
			z (torch.Tensor): Latent state from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		# Sample policy trajectories
		z = self.model.encode(obs, task)
		if self.cfg.num_pi_trajs:
			pi_actions = torch.empty(self.cfg.horizon, self._num_envs, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)

			task_pi = task.unsqueeze(1)
			task_pi = task_pi.repeat(1, self.cfg.num_pi_trajs).flatten(0)

			_z = z.unsqueeze(1).repeat(1, self.cfg.num_pi_trajs, 1).flatten(0, 1)
			for t in range(self.cfg.horizon - 1):
				a, _ = self.model.pi(_z, task)
				pi_actions[t] = a.view(self._num_envs, self.cfg.num_pi_trajs, self.cfg.action_dim)
				_z = self.model.next(_z, a, task)
			a, _ = self.model.pi(_z, task)
			pi_actions[-1] = a.view(self._num_envs, self.cfg.num_pi_trajs, self.cfg.action_dim)
			pi_actions = pi_actions.permute(1,0,2,3).contiguous()
		else:
			pi_actions = None

		# Initialize state and parameters
		_z = z.unsqueeze(1).repeat(1, self.cfg.num_samples, 1).flatten(0, 1)
		mean = torch.zeros(self._num_envs, self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = torch.full_like(mean, self.cfg.max_std)
		
		mean[~t0] = self._prev_mean[~t0]

		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample actions
			r = torch.randn(
				self._num_envs, self.cfg.horizon, self.cfg.num_samples - self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device
			)
			actions_sample = (mean.unsqueeze(2) + std.unsqueeze(2) * r).clamp(-1, 1)
			actions = torch.cat((pi_actions, actions_sample), 2) if pi_actions is not None else actions_sample
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# Compute elite actions
			actions_flat = actions.permute(1,0,2,3).reshape(
				self.cfg.horizon,
				self._num_envs * self.cfg.num_samples,
				self.cfg.action_dim
			)
			task_rep = None
			if task is not None:
			    task_rep = task.unsqueeze(1)
			    task_rep = task_rep.repeat(1, self.cfg.num_samples)
			    task_rep = task_rep.flatten(0)

			value = self._estimate_value(_z, actions_flat, task_rep).view(self._num_envs, self.cfg.num_samples, 1)
			elite_idx = value.squeeze(-1).topk(self.cfg.num_elites, 1).indices
			actions_perm = actions.permute(0,2,1,3).contiguous()
			elite_actions = torch.gather(
				actions_perm,
				1,
				elite_idx[:, :, None, None].expand(-1, -1, self.cfg.horizon, self.cfg.action_dim)
			)
			elite_value = torch.gather(value, 1, elite_idx.unsqueeze(-1))

			# Update parameters
			max_value = elite_value.max(1, keepdim=True).values
			score = torch.exp(self.cfg.temperature * (elite_value - max_value))
			score = score / (score.sum(1, keepdim=True) + 1e-9)
			mean = (score[..., None] * elite_actions).sum(1)
			std = ((score[..., None] * (elite_actions - mean[:, None])**2).sum(1) / (score.sum(1, keepdim=True) + 1e-9)).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		# Select action
		rand_idx = math.gumbel_softmax_sample(score.squeeze(-1))
		first_a = elite_actions[torch.arange(self._num_envs), rand_idx, 0]
		if not eval_mode:
			first_a = first_a + std[:, 0] * torch.randn_like(first_a)
		
		self._prev_mean.copy_(mean)
		return first_a.clamp(-1, 1)

	def update_pi(self, zs, task):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		action, info = self.model.pi(zs, task)
		qs = self.model.Q(zs, action, task, return_type='avg', detach=True)
		self.scale.update(qs[0])
		qs = self.scale(qs)

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		info = TensorDict({
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"pi_entropy": info["entropy"],
			"pi_scaled_entropy": info["scaled_entropy"],
			"pi_scale": self.scale.value,
		})
		return info

	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			terminated (torch.Tensor): Termination signal at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: TD-target.
		"""
		action, _ = self.model.pi(next_z, task)
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * (1-terminated) * self.model.Q(next_z, action, task, return_type='min', target=True)

	def _update(self, obs, action, reward, terminated, task=None):
		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, terminated, task)

		# Prepare for update
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_loss = 0
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			z = self.model.next(z, _action, task)
			consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
			zs[t+1] = z

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)
		if self.cfg.episodic:
			termination_pred = self.model.termination(zs[1:], task, unnormalized=True)

		# Compute losses
		reward_loss, value_loss = 0, 0
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

		consistency_loss = consistency_loss / self.cfg.horizon
		reward_loss = reward_loss / self.cfg.horizon
		if self.cfg.episodic:
			termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
		else:
			termination_loss = 0.
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.termination_coef * termination_loss +
			self.cfg.value_coef * value_loss
		)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		# Update policy
		pi_info = self.update_pi(zs.detach(), task)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		info = TensorDict({
			"consistency_loss": consistency_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"termination_loss": termination_loss,
			"total_loss": total_loss,
			"grad_norm": grad_norm,
		})
		if self.cfg.episodic:
			info.update(math.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]))
		info.update(pi_info)
		return info.detach().mean()

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, reward, terminated, task = buffer.sample()
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		torch.compiler.cudagraph_mark_step_begin()
		return self._update(obs, action, reward, terminated, **kwargs)
