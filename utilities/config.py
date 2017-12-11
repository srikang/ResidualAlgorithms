import sys
import os

#--------------------------------------------------------------------------------------
class TrainingMode(enumerate):
	EPOCH = 0
	INCREMENTAL = 1	

class EnvironmentType(enumerate):
	GRIDWORLD = 0
	MOUNTAIN_CAR = 1	

#------------------------------------------------------------------------------------------

TRIALS_COUNT = 10
EPISODE_COUNT = 100
MAX_TIME_STEP = 2000
ALPHA = 0.01
MU = 0.01
PHI_CONSTANT = 0.05
GRIDWORLD_START_STATE = (0, 0)
FOURIER_ORDER = 1
EPSILON = 0.05
GAMMA = 1

#--------------------------------------------------------------------------------------------

ENVIRONMENT = EnvironmentType.GRIDWORLD
TRAINING_TYPE = TrainingMode.INCREMENTAL
FEATURE_SIZE = {EnvironmentType.GRIDWORLD : 4 * 5 * 5, EnvironmentType.MOUNTAIN_CAR : 3 * (FOURIER_ORDER + 1) * (FOURIER_ORDER + 1)}

