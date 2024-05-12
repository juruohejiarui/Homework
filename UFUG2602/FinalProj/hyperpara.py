# hyperpara.py

# It works as the Lib of external hyperparameter which can be valued by hand.

from os import path
from func_lib import Trie
from functools import cache

# The base dictionary of your project
BASE_DIR = path.dirname(path.abspath(__file__))

# Threshold values
APPROX_LEN_THRESHOLD = 5
ERROR_THRESHOLD = 2

def set_thresholds(approx_len : int, error : int | float) -> None:
	# Set the threshold values
	global APPROX_LEN_THRESHOLD, ERROR_THRESHOLD
	APPROX_LEN_THRESHOLD = approx_len
	ERROR_THRESHOLD = error
	Trie.findCandidates.cache_clear()

def get_thresholds() -> tuple[int, int | float]:
	# Get the threshold values
	return APPROX_LEN_THRESHOLD, ERROR_THRESHOLD

# PATH of the testing data (string, easy loading)
TEST_DATA_01 = path.join(BASE_DIR, 'data', 'test_data_01.csv')
TEST_DATA_02 = path.join(BASE_DIR, 'data', 'test_data_02.csv')
# YOUR_OWN_TEST_DATA = path.join(BASE_DIR, 'data', 'test.csv')