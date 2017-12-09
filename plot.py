import os
import sys
from matplotlib import pyplot as plt
import numpy as np

def main():
	episode_count = 200
	max_file_count = 1
	mean_return = np.zeros(episode_count)
	file_indicies = [x+1 for x in xrange(max_file_count)]
	count = 0
	returns_array = []
	for file_index in file_indicies:
		file = open('mean_returns_' + str(file_index) + '.txt', 'r')
		for line in file.readlines():
			count += 1
			#print line
			line = line.split(',')
			returns = []
			for item in line:
				#print item
				returns.append(int(item))
			mean_return = mean_return + np.array(returns[0:episode_count])
			returns_array.append(returns)
		file.close()

	returns_array = np.array(returns_array)
	returns_array = returns_array.T
	std_dev = np.std(returns_array, axis = 1)
	print std_dev.shape
	mean_return = mean_return / count
	plt.plot(xrange(episode_count), mean_return)
	plt.xlabel('episode count')
	plt.ylabel('mean returns')
	plt.ylim((-1000,0))
	plt.errorbar(xrange(episode_count), mean_return[0:episode_count], std_dev[0:episode_count])
	plt.title('Q-Learning mean returns plot with standard deviation as error bars')
	plt.show()





if __name__ == '__main__':
	main()