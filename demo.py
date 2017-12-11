import sys
from utilities import config
import os
from Agent import residual_algorithm as RA
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time


def runTrials(trials, episode_count):
	trial_history = []
	start_time = time.time()
	for trial_index in xrange(trials):
		print('starting trial ' + str(trial_index + 1))
		agent = RA.ResidualAlgorithm(config.EPSILON, config.ALPHA, config.FEATURE_SIZE[config.ENVIRONMENT], config.ENVIRONMENT, config.TRAINING_TYPE, config.MU)
		rewards_list = agent.runTrial(episode_count)
		trial_history.append(rewards_list)

	print('Running time =' + str(time.time() - start_time) + ' seconds')
	# compute the mean returns and  standard deviation numpy arrays
	returns = np.array(trial_history) #(trial_count, episode_count)
	mean_returns = np.mean(returns, axis = 0)
	std_dev = np.std(returns, axis = 0)
	# run the plotting code
	episode_list = [i+1 for i in xrange(episode_count)]
	plt.plot(episode_list, mean_returns)
	plt.xlabel('episode count')
	plt.ylabel('mean returns')
	plt.errorbar(episode_list, mean_returns, std_dev)
	environment_string = 'Grid World' if config.ENVIRONMENT == config.EnvironmentType.GRIDWORLD else 'Mountain Car'
	training_string = 'epoch mode' if config.TRAINING_TYPE == config.TrainingMode.EPOCH else 'incremental mode'
	plt.title('Residual Algorithm mean returns plot with standard deviation as error bars for ' + environment_string + ' domain in ' + training_string)
	plt.show()

		
def main():
	runTrials(config.TRIALS_COUNT, config.EPISODE_COUNT)

if __name__ == '__main__':
	main()