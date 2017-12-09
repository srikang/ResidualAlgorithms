import os
import sys
import math
import numpy as np
from threading import Thread
from matplotlib import pyplot as plt
import time

class MountainCar(object):
	"""docstring for Environment"""
	def __init__(self):
		self.car_position = -0.5
		self.car_velocity = 0
		self.velocity_lower_bound = -0.07
		self.velocity_upper_bound = 0.07
		self.position_lower_bound = -1.2
		self.position_upper_bound = 0.5

		self.max_time_steps = 20000
		self.current_time_step = 0

	def isEpisodeOver(self):
		if self.current_time_step >= self.max_time_steps:
			return True
		return False

	def getState(self):
		return (self.car_position, self.car_velocity)


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