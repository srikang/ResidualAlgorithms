# Residual Algorithms

Based on the following source: www.leemon.com/papers/1995b.pdf

An agent is trained using the residual algorithm with epsilon greedy exploration on the grid world and the mountain car environments.

The Mean return vs episode plots for each of the settings are tested:
1. Residual algorithm with epoch training mode on Gridworld
2. Residual algorithm with epoch training mode on Mountain Car
3. Residual algorithm with incremental learning (uses traces) mode on Gridworld
4. Residual algorithm with incremental learning (uses traces) mode on Mountain Car.



Running times:
Mountain car , incremental, adaptive mode , 480.447901011 seconds
Mountain car , incremental, direct grad , 398.989701986 seconds
Mountain car , incremental, residual grad , 4914.64692903 seconds
Mountain car , epoch, adaptive mode , 588.568005085 seconds
Mountain car , epoch, direct grad , 572 seconds
Mountain car , epoch,  residual grad, 7829.71257687 seconds


Grid world, incremental, adaptive mode, 120.229432106 seconds
Grid world, incremental, direct grad (0), 57.5160820484 seconds
Grid world, incremental, residual grad (1), 2339.00636792 seconds
Grid world, epoch, adaptive mode, 63.7803812027 seconds
Grid world, epoch, direct grad, 52.1971981525 seconds
Grid world, epoch, residual grad, 1046.95599103 seconds


Hyperparameters used for Mountain Car:
TRIALS_COUNT = 100
EPISODE_COUNT = 300
MAX_TIME_STEP = 2000
ALPHA = 0.01
MU = 0.05
PHI_CONSTANT = 0.05
FOURIER_ORDER = 1
EPSILON = 0.05
GAMMA = 1


Hyoerpaprameters used for Grid World: 
TRIALS_COUNT = 100
EPISODE_COUNT = 300
MAX_TIME_STEP = 2000
ALPHA = 0.01
MU = 0.05
PHI_CONSTANT = 0.05
GRIDWORLD_START_STATE = (0, 0)
EPSILON = 0.05
GAMMA = 1
