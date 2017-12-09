import os
import sys
import numpy as np

class GridWorld(object):
	"""docstring for ValueIteration"""
	def __init__(self, start_state):
		self.layout = np.zeros((5,5))
		self.layout[2,2] = 1
		self.layout[3,2] = 1

		self.reward = np.zeros((5,5))
		self.reward[4,4] = 10
		self.reward[4,2] = -10


		self.value_matrix = np.zeros((5,5))

		self.max_grid_size = 4
		self.current_state = start_state
		self.is_episode_over = False

		self.probability_distribution_vector = [0.8, 0.1, 0.05, 0.05]

	'''
	Returns if the episode has terminated or not
	'''
	def isEpisodeOver(self):
		return self.is_episode_over

	'''
	Returns the current state of the agent
	'''
	def getState(self):
		return self.current_state

	def update_state(self, action):
		self.current_state = self.update_state_internal(self.current_state[0], self.current_state[1], action)
		if self.current_state[0] == self.max_grid_size and self.current_state[1] == self.max_grid_size:
			self.is_episode_over = True
		return self.current_state


	def update_state_internal(self, row_index, column_index, action):
		if self.layout[row_index,column_index] == 1:
			return (-1, -1) #invalid state

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

		print(pos_dictionary)


		effective_action = np.random.choice(4, 1, p = self.probability_distribution_vector)[0]
		print effective_action

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



if __name__ == '__main__':
	environment = GridWorld((4,3))
	# running few tests to ensure that the environment matches the specified criteria
	#print environment.update_state_internal(4, 2, 'left') # test for movement

	#print environment.isEpisodeOver()

	#print environment.update_state_internal(4, 3, 'right') # test for episode termination

	print environment.update_state('right')

	print environment.isEpisodeOver()

	#print environment.update_state_internal(2, 1, 'right') # test for avoiding invalid states

