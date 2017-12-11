import sys
import os
import numpy as np
from utilities import config
from Environments import GridWorld
from Environments import MountainCar

np.random.seed(0)

class ResidualAlgorithm(object):
	def __init__(self, epsilon, alpha, feature_size, environment_type, training_mode, mu):
		super(ResidualAlgorithm, self).__init__()
		self.epsilon = epsilon # for epsilon greedy action selection
		self.alpha = alpha
		self.training_mode = training_mode
		self.mu = mu
		self.weights = np.zeros(feature_size)
		self.feature_size = feature_size
		self.environment_type = environment_type
		self.direct_weights_trace = None
		self.residual_gradient_weights_trace = None
		self.reset_trace_variables()


	def getEnvironment(self):
		if self.environment_type == config.EnvironmentType.GRIDWORLD:
			return GridWorld.GridWorld(config.GRIDWORLD_START_STATE, config.MAX_TIME_STEP)
		elif self.environment_type == config.EnvironmentType.MOUNTAIN_CAR:
			return MountainCar.MountainCar(config.FOURIER_ORDER, config.MAX_TIME_STEP)


	def reset_trace_variables(self):
		self.direct_weights_trace = np.zeros(self.feature_size)
		self.residual_gradient_weights_trace = np.zeros(self.feature_size)
		
	'''
	Picks the action using the epsilon greedy exploration method
	'''
	def getAction(self, state, environment):
		action_set = environment.getActions()
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
			if score == best_score:
				best_action_list.append(action)

		best_action_index = np.random.randint(len(best_action_list), size = 1)[0]
		best_action = best_action_list[best_action_index]
		random_draw = np.random.random()

		if random_draw > self.epsilon:
			return best_action

		best_action_index = np.random.randint(len(action_set), size = 1)[0]
		return action_set[best_action_index]


	def compute_phi_incremental_mode(self, current_features, reward, next_features_list):
		delta = reward + config.GAMMA * np.dot(self.weights, next_features_list[0]) - np.dot(self.weights, current_features)
		self.direct_weights_trace = (1 - self.mu) * self.direct_weights_trace + (self.mu * delta * current_features)
		self.residual_gradient_weights_trace = (1 - self.mu) * self.residual_gradient_weights_trace + (-self.mu * delta * (config.GAMMA * next_features_list[1] - current_features))
		# compute dot product of the denominator
		term1 = np.dot(self.direct_weights_trace, self.residual_gradient_weights_trace)
		denominator = term1 - np.dot(self.residual_gradient_weights_trace, self.residual_gradient_weights_trace)
		numerator = term1
		phi = -1
		# if value equals zero then phi = small constant
		if denominator == 0:
			phi = config.PHI_CONSTANT
		else:
			phi = (numerator / denominator) + self.mu
			# if value of phi outpside [0, 1] then set phi to 0
			if phi < 0 or phi > 1:
				phi = 0
		return phi
		

	def compute_phi_epoch_mode(self, transition_list):
		direct_weights_grad = np.zeros(self.weights.shape)
		residual_gradients_weights_grad = np.zeros(self.weights.shape) 
		# compute the value of phi if epoch mode
		for instance in transition_list:
			current_features = instance[0]
			reward = instance[1]
			next_features_list = instance[2]

			delta = reward + config.GAMMA * np.dot(self.weights, next_features_list[0]) - np.dot(self.weights, current_features)
			direct_weights_grad += self.alpha * delta * current_features
			residual_gradients_weights_grad += -self.alpha * delta * (config.GAMMA * next_features_list[1] - current_features)

		term1 = np.dot(direct_weights_grad, residual_gradients_weights_grad)
		term2 = np.dot(residual_gradients_weights_grad, residual_gradients_weights_grad)
		denominator =  term1 - term2 
		numerator = term1
		phi = -1
		# if value equals zero then phi = small constant
		if denominator == 0:
			phi = config.PHI_CONSTANT
		else:
			phi = (numerator / denominator)
			# if value of phi outpside [0, 1] then set phi to 0
			if phi < 0 or phi > 1:
				phi = 0
		return phi
		

	def perform_weight_update(self, current_features, reward, next_features_list, phi):
		delta = reward + config.GAMMA * np.dot(self.weights, next_features_list[0]) - np.dot(self.weights, current_features)
		self.weights = self.weights - (self.alpha * delta * (config.GAMMA * phi * next_features_list[1] - current_features))

		
	def runTrial(self, episode_count):
		rewards_list = []
		for episode_index in xrange(episode_count):
			environment = self.getEnvironment()
			current_state = environment.getState()
			current_action = self.getAction(current_state, environment) 
			total_reward = 0
			transition_list = []
			if self.training_mode == config.TrainingMode.INCREMENTAL:
				self.reset_trace_variables()
			while environment.isEpisodeOver() == False and environment.isTerminalState(current_state) == False:
				next_state = environment.updateState(current_action)
				next_action_list = [self.getAction(next_state, environment), self.getAction(next_state, environment)]
				reward = environment.getReward(current_state, current_action, next_state)
				total_reward += reward
				current_features = environment.getFeatures(current_state, current_action)
				next_features_list = [environment.getFeatures(next_state, next_action_list[0]), environment.getFeatures(next_state, next_action_list[1])]
				
				if self.training_mode == config.TrainingMode.INCREMENTAL:
					phi = self.compute_phi_incremental_mode(current_features, reward, next_features_list)
					#phi = 0 # to test the algorithm in direct gradient and residual gradient scenarios in incremental mode
					self.perform_weight_update(current_features, reward, next_features_list, phi)
				else:
					transition_list.append([current_features, reward, next_features_list])
				
				current_state = next_state
				current_action = next_action_list[0]
			
			if  self.training_mode == config.TrainingMode.EPOCH:
				phi = self.compute_phi_epoch_mode(transition_list)
				#phi = 0 # to test the algorithm in direct gradient and residual gradient scenarios in epoch mode
				for instance in transition_list:
					current_features = instance[0]
					reward = instance[1]
					next_features_list = instance[2]
					self.perform_weight_update(current_features, reward, next_features_list, phi)
			rewards_list.append(total_reward)
		return rewards_list


