import sys
import os
import numpy as np
from .. import config
from ..Environments import GridWorld
from ..Environments import MountainCar

np.random.seed(0)

class ResidualAlgorithm(object):
	def __init__(self, epsilon, alpha, feature_size, environment_type, training_mode, mu):
		super(ResidualAlgorithm, self).__init__()
		self.epsilon = epsilon # for epsilon greedy action selection
		self.alpha = alpha
		self.training_mode = training_mode
		self.mu = mu
		self.weights = np.zeros(config.FEATURE_SIZE[environment_type])
		self.episode_count = episode_count
		self.trials = trials
		self.environment_type = environment_type


	def getEnvironment(self):
		if self.environment_type == config.EnvironmentType.GRIDWORLD:
			return GridWorld.GridWorld(config.GRIDWORLD_START_STATE)
		elif self.environment_type == config.EnvironmentType.MOUNTAIN_CAR:
			return MountainCar.MountainCar(config.FOURIER_ORDER)

	'''
	Picks the action using the epsilon greedy exploration method
	'''
	def getAction(self, state, environment):
		action_set = environment.getAction()
		best_action = None
		best_score = None
		for action in action_set:
			features = environment.getFeatures(state, action)
			score =  np.dot(self.weights, np.array(features))
			if best_score == None:
				best_score = score
			elif best_score < score:
				best_score = score

		best_action_list = []
		for action in action_set:
			features = environment.getFeatures(state, action)
			score =  np.dot(self.weights, np.array(features))
			if score == best_action:
				best_action_list.append(action)

		best_action = np.random.randint(len(best_action_list), size = 1)[0]

		best_action_probability = 1.0 - self.epsilon

		random_draw = np.random.random_sample()

		if best_action_probability >= random_draw:
			return best_action

		best_action = np.random.randint(len(action_set), size = 1)[0]
		return best_action


	def perform_incremental_update(self, current_state, current_action, next_state, next_action_list):
		# compute the value of phi if incremental method is used
		# perform the weight update for the q function if in incremental mode
		pass


	def perform_epoch_update(self, transition_list):
		# compute the value of phi if epoch mode
		#perform the weight update for the q function if in epoch mode
		pass
		
	def runTrial(self):
		rewards_list = []
		for episode_index in xrange(self.episode_count):
			# fetch the environment
			environment = self.getEnvironment()
			current_state = environment.getState() # state s
			# draw action from the epsilon greedy method
			current_action = self.getAction(current_state, environment) 
			total_reward = 0
			transition_list = []

			while environment.isEpisodeOver() == False:
				# obtain the next state using the current action
				next_state = environment.updateState(current_action)

				# obtain the next action using the epsilon greedy method (draw the action twice if the MDP is stochastic to get a better estimate)
				next_action_list = [self.getAction(next_state, environment), self.getAction(next_state, environment)]

				# sum the reward variable with R(s, a, s')
				reward = environment.getReward(current_state, current_action, next_state)
				total_reward += reward

				if self.training_mode == config.TrainingMode.INCREMENTAL:
					self.perform_incremental_update(current_state, current_action, next_state, next_action_list)
				else:
					transition_list.append((current_state, current_action, reward, next_state, next_action_list))
				
				current_state = next_state
				current_action = next_action_list[0]
			
			if  self.training_mode == config.TrainingMode.EPOCH:
				self.perform_epoch_update(transition_list)
			
			rewards_list.append(total_reward)
		return rewards_list



def runTrials(trials, episode_count):
	trial_history = []
	for trial_index in xrange(self.trials):
		agent = ResidualAlgorithm(config.EPSILON, config.ALPHA, config.ENVIRONMENT, config.TRAINING_TYPE, config.MU)
		rewards_list = 
		



def main():
	runTrials(trials, episode_count)

if __name__ == '__main__':
	main()