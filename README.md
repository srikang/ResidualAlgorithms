# Residual Algorithms

Instructions to run:

1. Update the config.py file in utilities/ to set the environment using the ENVIRONMENT variable and to set the training mode using the TRAINING_TYPE variable.
2. Set the desired hyperparameters in the config.py file.
3. Run the demo.py file in the main folder to train the agent.


Few details about the algorithm:

Based on the following source: www.leemon.com/papers/1995b.pdf

An agent is trained using the residual algorithm with epsilon greedy exploration on the grid world and the mountain car environments.

The Mean return vs episode plots for each of the settings are tested:

1. Residual algorithm with epoch training mode on Gridworld
2. Residual algorithm with epoch training mode on Mountain Car
3. Residual algorithm with incremental learning (uses traces) mode on Gridworld
4. Residual algorithm with incremental learning (uses traces) mode on Mountain Car.



Running times:

1. Mountain car , incremental, adaptive mode , 480.447901011 seconds
2. Mountain car , incremental, direct grad , 398.989701986 seconds
3. Mountain car , incremental, residual grad , 4914.64692903 seconds
4. Mountain car , epoch, adaptive mode , 588.568005085 seconds
5. Mountain car , epoch, direct grad , 572 seconds
6. Mountain car , epoch,  residual grad, 7829.71257687 seconds
7. Grid world, incremental, adaptive mode, 120.229432106 seconds
8. Grid world, incremental, direct grad (0), 57.5160820484 seconds
9. Grid world, incremental, residual grad (1), 2339.00636792 seconds
10. Grid world, epoch, adaptive mode, 63.7803812027 seconds
11. Grid world, epoch, direct grad, 52.1971981525 seconds
12. Grid world, epoch, residual grad, 1046.95599103 seconds


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