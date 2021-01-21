import time
from typing import Callable

import numpy as np

from lib import conditional_entropy as c_entropy, conditional_entropy_compact as c_entropy_compact


def benchmark_time(for_function: Callable, with_args: dict):
  """
    Executes a function and returns a tuple containing
    the returned value and the execution time in seconds.
  """

  start = time.time()
  result = for_function(**with_args)
  end = time.time()

  return result, end - start


def test_conditional_entropy():

  n_dash = 500000
  C = 20
  params = {
    'x_dash': np.random.rand(n_dash), # input vectors
    'n_dash': n_dash, # number of unlabeled data
    'C': C, # number of classes
    'P': lambda y, x: 1 / C # probability for P(y_i^m = 1 | x'^m)
  }

  normal_val, normal_time = benchmark_time(for_function=c_entropy, with_args=params)
  compact_val, compact_time = benchmark_time(for_function=c_entropy_compact, with_args=params)

  print()

  print('Executed "conditional_entropy":\n')
  print('H = {} (took {}s)'.format(normal_val, normal_time))

  print('\n')

  print('Executed "conditional_entropy_compact":\n')
  print('H = {} (took {}s)'.format(compact_val, compact_time))

  print()


if __name__ == '__main__':
  test_conditional_entropy()