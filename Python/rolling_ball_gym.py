from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
import numpy as np
import rolling_ball_DQN
import matplotlib.pyplot as plt
import torch

def main():

	#Parameters
	input_size = 9
	output_size = 5
	batch_size = 256
	gamma = 0.99
	tau = 0.005
	lr = 1e-4
	eps_start = 0.9
	eps_end = 0.05
	eps_decay = 2000
	is_train = True
	load_brain = False

	# Init environment 
	print("Starting simulation. Press the Play button in Unity editor")
	unity_env = UnityEnvironment(file_name=None)
	gym = UnityToGymWrapper(unity_env=unity_env, allow_multiple_obs=True)
	
	try:
		rewards, reward_episode, episode_duration = [], [], []
		step = 0
		total_steps = 0

		# Init Deep Q-learning 
		brain = rolling_ball_DQN.Dqn(input_size=input_size,
				 output_size=output_size,
				 batch_size=batch_size,
				 gamma=gamma,
				 tau=tau,
				 lr=lr,
				 eps_start=eps_start,
				 eps_end=eps_end,
				 eps_decay=eps_decay)
		if load_brain:
			brain.load()
		
		#Start simulation
		reset_gym(gym,brain)
		while(True):
			step += 1

			# Advance one step in the environment
			action = brain.select_actions(brain.state, is_train)
			observation, reward, terminated, _ = gym.step(action.item())
			observation = observation[0]
			#print_gym_status(action, observation, reward, terminated, truncated)

			# Update DQN with new environment state
			brain.update(reward,observation,terminated,is_train)
			rewards.append(reward)

			# If agent reached end of episode
			if(terminated):
				reset_gym(gym,brain)
				reward_episode.append(reward)
				episode_duration.append(step)
				nb_episode = len(reward_episode)
				total_steps += step
				print(f"Episode: {nb_episode} / Step: {total_steps} / Reward: {reward} / Duration: {step}")
				step = 0
				if(nb_episode > 20):
					if(np.mean(reward_episode[nb_episode-20:nb_episode])>0.9):
						break

	finally:
		gym.close()
		brain.save()
		plot_reward_history(reward_episode,episode_duration, total_steps)

def reset_gym(gym, brain):
	""" Reset gym environment (required after the end of an episode)
	Update the brain internal state with the current gym state

	Args:
		gym (gym): Gym environment
		brain (DQN): Reinforcement model
	"""
	state = gym.reset()
	state = torch.tensor(state[0], dtype=torch.float32, device=brain.device).unsqueeze(0)
	brain.state = state


def plot_reward_history(rewards, durations, steps):
	"""End of simulation plot

	Args:
		rewards (list of float): Rewards throughout the simulation
		durations (list of int): Duration of episodes throughout the simulation
		steps (int): Number of total steps during training
	"""
	average_window = 20

	if len(rewards) < average_window:
		print("Not enough episode to plot results")
		return
	
	mean_rewards, mean_durations = [],[]
	for i in range(0,average_window):
		mean_rewards.append(np.mean(rewards[0:i]))
		mean_durations.append(np.mean(durations[0:i]))
	for i in range(average_window,len(rewards)):
		mean_rewards.append(np.mean(rewards[i-average_window:i]))
		mean_durations.append(np.mean(durations[i-average_window:i]))

	fig, axs = plt.subplots(2,1, figsize=(16, 9))
	fig.suptitle(f"Rolling ball info for {len(rewards)} episodes and {steps} steps")
	axs[0].plot(rewards)
	axs[0].plot(mean_rewards)
	axs[0].set(ylabel=" Last reward", xlabel="Episode")
	axs[0].grid()
	axs[0].set
	axs[1].plot(durations)
	axs[1].plot(mean_durations)
	axs[1].set(ylabel="Duration (steps)", xlabel="Episode")
	axs[1].grid()
	plt.savefig("rolling_ball_reward.png")
	plt.show()


def print_gym_status(action, observation, reward, terminated):
	""" Print state information.
	Parameters are given by gym.step()
	"""
	print("[Gym infos]")
	print(f"Action: {action}")
	print(f"Obs: {observation}")
	print(f"Reward: {reward}")
	print(f"Terminated: {terminated}")
	
	
if __name__ == '__main__':
  main()