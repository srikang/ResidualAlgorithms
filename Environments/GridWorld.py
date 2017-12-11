import os
import sys
import numpy as np

class GridWorld(object):
	def __init__(self, start_state, max_time_step):
		self.layout = np.zeros((5,5))
		self.layout[2,2] = 1
		self.layout[3,2] = 1

		self.reward = np.zeros((5,5))
		self.reward[4,4] = 10
		self.reward[4,2] = -10

		self.max_grid_size = 4
		self.current_state = start_state

		self.probability_distribution_vector = [0.8, 0.1, 0.05, 0.05]
		self.action_set = ['top', 'bottom', 'left', 'right']

		self.current_time_step = 0
		self.max_time_step = max_time_step

	'''
	Returns if the episode has terminated or not
	'''
	def isEpisodeOver(self):
		if self.current_time_step >= self.max_time_step:
			return True
		return False

	def isTerminalState(self, state):
		if self.current_state[0] == self.max_grid_size and self.current_state[1] == self.max_grid_size:
			return True
		return False

	def getActions(self):
		return self.action_set

	'''
	Returns the current state of the agent
	'''
	def getState(self):
		return self.current_state

	def getReward(self, current_state, action, next_state):
		if current_state[0] == self.max_grid_size and current_state[1] == self.max_grid_size:
			return 0
		return self.reward[next_state[0], next_state[1]]

	def updateState(self, action):
		self.current_state = self.update_state_internal(self.current_state[0], self.current_state[1], action)
		self.current_time_step += 1
		return self.current_state


	def update_state_internal(self, row_index, column_index, action):
		if row_index == self.max_grid_size and column_index == self.max_grid_size:
			return self.current_state

		pos_dictionary = {}
		if row_index + 1 >= 0 and row_index + 1 <= self.max_grid_size:
			pos_dictionary['bottom'] = (row_index + 1, column_index)
		if row_index - 1 >= 0 and row_index - 1 <= self.max_grid_size:
			pos_dictionary['top'] = (row_index - 1, column_index)
		if column_index + 1 >= 0 and column_index + 1 <= self.max_grid_size:
			pos_dictionary['right'] = (row_index, column_index + 1)
		if column_index - 1 >= 0 and column_index -1 <= self.max_grid_size:
			pos_dictionary['left'] = (row_index, column_index - 1)


		for key in pos_dictionary.keys():
			temp1, temp2 = pos_dictionary[key]
			if self.layout[temp1, temp2] == 1:
				pos_dictionary[key] = (row_index, column_index)

		for key in ['top', 'bottom', 'right', 'left']:
			if pos_dictionary.has_key(key) == False:
				pos_dictionary[key] = (row_index, column_index)

		effective_action = np.random.choice(4, 1, p = self.probability_distribution_vector)[0]

		if action == 'right':
			if effective_action == 0:
				return pos_dictionary['right']
			elif effective_action == 1:
				return (row_index, column_index)
			elif effective_action == 2:
				return pos_dictionary['top']
			else:
				return pos_dictionary['bottom']
			
		if action == 'left':
			if effective_action == 0:
				return pos_dictionary['left']
			elif effective_action == 1:
				return (row_index, column_index)
			elif effective_action == 2:
				return pos_dictionary['top']
			else:
				return pos_dictionary['bottom']
			
		if action == 'top':
			if effective_action == 0:
				return pos_dictionary['top']
			elif effective_action == 1:
				return (row_index, column_index)
			elif effective_action == 2:
				return pos_dictionary['right']
			else:
				return pos_dictionary['left']
			
		if action == 'bottom':
			if effective_action == 0:
				return pos_dictionary['bottom']
			elif effective_action == 1:
				return (row_index, column_index)
			elif effective_action == 2:
				return pos_dictionary['left']
			else:
				return pos_dictionary['right']

	def getFeatures(self, state, action):
		features = np.zeros(len(self.action_set) * (self.max_grid_size + 1) * (self.max_grid_size + 1))
		# for the terminal state the feature vector will be zero
		if state[0] == self.max_grid_size and state[1] == self.max_grid_size:
			return features
		index = 0
		if action == self.action_set[1]:
			index = (self.max_grid_size + 1) * (self.max_grid_size + 1)
		elif action == self.action_set[2]:
			index = 2 * (self.max_grid_size + 1) * (self.max_grid_size + 1)
		elif action == self.action_set[3]:
			index = 3 * (self.max_grid_size + 1) * (self.max_grid_size + 1)

		index = index + state[0] * (self.max_grid_size + 1) + state[1]
		features[index] = 1
		return features


'''
if __name__ == '__main__':
	environment = GridWorld((4,3))
	# running few tests to ensure that the environment matches the specified criteria
	#print environment.update_state_internal(4, 2, 'left') # test for movement

	#print environment.isEpisodeOver()

	#print environment.update_state_internal(4, 3, 'right') # test for episode termination

	print environment.updateState('right')

	print environment.isEpisodeOver()

	#print environment.update_state_internal(2, 1, 'right') # test for avoiding invalid states
'''
