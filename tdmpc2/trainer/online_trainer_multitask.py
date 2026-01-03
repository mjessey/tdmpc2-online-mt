from time import time

from tqdm import tqdm
import random
import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer


class OnlineTrainerMultitask(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._task_step = [0] * len(self.cfg.tasks)
		self._ep_idx = 0
		self._task_idx = 0
		self._task_avg_reward = [0.0] * len(self.cfg.tasks)
		self._start_time = time()

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		elapsed_time = time() - self._start_time
		return dict(
			step=self._step,
			episode=self._ep_idx,
			elapsed_time=elapsed_time,
			steps_per_second=self._step / elapsed_time
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		results = dict()
		for task_idx in tqdm(range(len(self.cfg.tasks)), desc='Evaluating'):
			ep_rewards, ep_successes = [], []
			for i in range(self.cfg.eval_episodes):
				obs, done, ep_reward, t = self.env.reset(task_idx), False, 0, 0
				if self.cfg.save_video:
					self.logger.video.init(self.env, enabled=(i==0))
				while not done:
					torch.compiler.cudagraph_mark_step_begin()
					action = self.agent.act(obs, t0=t==0, eval_mode=True, task=task_idx)
					obs, reward, done, info = self.env.step(action)
					ep_reward += reward
					t += 1
					if self.cfg.save_video:
						self.logger.video.record(self.env)
				ep_rewards.append(ep_reward)
				ep_successes.append(info['success'])
				if self.cfg.save_video:
					self.logger.video.save(self._step, key=f"videos/{self.cfg.tasks[task_idx]}_eval_video")
			results.update({
				f'episode_reward+{self.cfg.tasks[task_idx]}': np.nanmean(ep_rewards),
				#f'episode_success+{self.cfg.tasks[task_idx]}': np.nanmean(ep_successes),
				})

		self.logger.save_checkpoint(self.agent, self.buffer, self._step)
		return results

	def to_td(self, obs, action=None, reward=None, terminated=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		if terminated is None:
			terminated = torch.tensor(float('nan'))
		td = TensorDict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
			terminated=terminated.unsqueeze(0),
			task=torch.tensor(self._task_idx).unsqueeze(0),
		batch_size=(1,))
		return td

	def pretrain(self):
		print("Sampling seed samples...")
		for task_idx in range(len(self.cfg.tasks)):
			done = True
			self._task_idx = task_idx
			while self._task_step[task_idx] <= self.cfg.seed_steps:
				if done:
					if self._task_step[task_idx] > 0:
						self._ep_idx = self.buffer.add(torch.cat(self._tds))

					obs = self.env.reset(task_idx)
					self._tds = [self.to_td(obs)]

				action = self.env.rand_act()
				obs, reward, done, info = self.env.step(action)
				self._tds.append(self.to_td(obs, action, reward, info['terminated']))

				self._step += 1
				self._task_step[task_idx] += 1

		print("Pretraining...")
		for _ in range(self.cfg.seed_steps):
			self.agent.update(self.buffer)
		print("Finished pretraining")

	def train(self):
		"""Train a TD-MPC2 agent."""
		self.pretrain()

		is_first_iter = True
		train_metrics, done, eval_next = {}, True, True
		#avail_tasks = [i for i in range(len(self.cfg.tasks))]

		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Reset environment
			if done:
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if not is_first_iter:
					if info['terminated'] and not self.cfg.episodic:
						raise ValueError('Termination detected but you are not in episodic mode. ' \
						'Set `episodic=true` to enable support for terminations.')
					episode_reward = torch.tensor([td['reward'] for td in self._tds[1:]]).sum()
					train_metrics.update(
						episode_reward=episode_reward,
						episode_success=info['success'],
						episode_length=len(self._tds),
						episode_terminated=info['terminated'])
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(torch.cat(self._tds))

					# Update moving average of task reward
					self._task_avg_reward[self._task_idx] *= 0.9
					self._task_avg_reward[self._task_idx] += 0.1 * episode_reward

					#task_weights = [1000.0 - self._task_avg_reward[i] for i in avail_tasks]
					#self._task_idx = random.choices(avail_tasks, weights=task_weights)[0]

				if random.random() < 0.5:
					# Set the next task to the one with the lowest average reward
					self._task_idx = self._task_avg_reward.index(min(self._task_avg_reward))
				else:
					# Set the next task to the one where the fewest experience was collected
					#self._task_idx = self._task_step.index(min(self._task_step))
					# Set the next task to a random one
					self._task_idx = random.randint(0, len(self.cfg.tasks) - 1)

				obs = self.env.reset(self._task_idx)
				self._tds = [self.to_td(obs)]

			# Collect experience
			action = self.agent.act(obs, t0=len(self._tds)==1, task=self._task_idx)

			obs, reward, done, info = self.env.step(action)
			self._tds.append(self.to_td(obs, action, reward, info['terminated']))

			# Update agent
			_train_metrics = self.agent.update(self.buffer)
			train_metrics.update(_train_metrics)

			self._step += 1
			self._task_step[self._task_idx] += 1
			is_first_iter = False

		eval_metrics = self.eval()
		eval_metrics.update(self.common_metrics())
		self.logger.log(eval_metrics, 'eval')

		self.logger.finish(self.agent)
