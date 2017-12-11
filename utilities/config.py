import sys
import os

#------------------------------------------------------------------------------------------
# DO NOT MODIFY THIS BLOCK
#-----------------------------------------------------------------------------------------
class TrainingMode(enumerate):
	EPOCH = 0
	INCREMENTAL = 1	

class EnvironmentType(enumerate):
	GRIDWORLD = 0
	MOUNTAIN_CAR = 1	

#------------------------------------------------------------------------------------------

TRIALS_COUNT = 100
EPISODE_COUNT = 300
MAX_TIME_STEP = 2000
ALPHA = 0.01
MU = 0.05
PHI_CONSTANT = 0.05
GRIDWORLD_START_STATE = (0, 0)
FOURIER_ORDER = 1
EPSILON = 0.05
GAMMA = 1

#--------------------------------------------------------------------------------------------

ENVIRONMENT = EnvironmentType.MOUNTAIN_CAR
TRAINING_TYPE = TrainingMode.EPOCH

#--------------------------------------------------------------------------------------------
# DO NOT MODIFY THIS BLOCK
#--------------------------------------------------------------------------------------------
FEATURE_SIZE = {EnvironmentType.GRIDWORLD : 4 * 5 * 5, EnvironmentType.MOUNTAIN_CAR : 3 * (FOURIER_ORDER + 1) * (FOURIER_ORDER + 1)}


