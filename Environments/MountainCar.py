import os
import sys
import math
import numpy as np

class MountainCar(object):
	def __init__(self, fourier_order, max_time_step):
		self.car_position = -0.5
		self.car_velocity = 0
		self.velocity_lower_bound = -0.07
		self.velocity_upper_bound = 0.07
		self.position_lower_bound = -1.2
		self.position_upper_bound = 0.5
		self.terminal_position = self.position_upper_bound
		self.max_time_steps = max_time_step
		self.current_time_step = 0
		self.action_set = [1, 0, -1]
		self.fourier_order = fourier_order

	def isEpisodeOver(self):
		if self.current_time_step >= self.max_time_steps:
			return True
		return False

	def isTerminalState(self, state):
		if state[0] == self.terminal_position:
			return True
		return False

	def getState(self):
		return (self.car_position, self.car_velocity)

	def getActions(self):
		return self.action_set

	def getReward(self, current_state, action, next_state):
		if current_state[0] == self.terminal_position:
			return 0
		return -1


	def updateState(self, current_action):
		#compute the velocity at new time step
		temp_velocity = self.car_velocity + 0.001 * current_action - 0.0025 * math.cos(3 * self.car_position)
		if temp_velocity < self.velocity_lower_bound:
			temp_velocity = self.velocity_lower_bound
		elif temp_velocity > self.velocity_upper_bound:
			temp_velocity = self.velocity_upper_bound
		
		temp_position = self.car_position + temp_velocity

		if temp_position < self.position_lower_bound:
			temp_position = self.position_lower_bound
			temp_velocity = 0
		elif temp_position > self.position_upper_bound:
			temp_position = self.position_upper_bound

		#compute the updated car position
		self.car_position = temp_position
		self.car_velocity = temp_velocity

		#update the time step count for the episode
		self.current_time_step += 1

		#return the state of the environment
		return (self.car_position, self.car_velocity)


	def getFeatures(self, state, action):
		if state[0] == self.terminal_position:
			return np.zeros(len(self.action_set) * (self.fourier_order + 1) * (self.fourier_order + 1))
		features = np.zeros(len(self.action_set) * (self.fourier_order + 1) * (self.fourier_order + 1))
		index = 0
		if action == 0:
			index = (self.fourier_order + 1) * (self.fourier_order + 1)
		elif action == -1:
			index = 2 * (self.fourier_order + 1) * (self.fourier_order + 1)

		new_pos = (1.2 + state[0])/ 1.7
		new_vel = (0.07 + state[1]) / 0.14
		
		for c_1 in xrange(self.fourier_order + 1):
			for c_2 in xrange(self.fourier_order + 1):
				features[index] = math.cos(math.pi * ( c_1 * new_pos + c_2 * new_vel))
				index = index + 1
		
		return features


def main():
	enviroment = MountainCar(1)
	print(enviroment.getState())


if __name__ == '__main__':
	main()
