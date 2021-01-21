from typing import Callable
import numpy as np


def conditional_entropy(x_dash: np.ndarray, n_dash: int, C: int, P: Callable[[int, float], float]) -> float:
  sum_term = 0
  for m in range(1, n_dash):
    for i in range(1, C):
      sum_term += P(i, x_dash[m]) * np.log(P(i, x_dash[m]))

  return -(1 / n_dash) * sum_term

def conditional_entropy_compact(x_dash: np.ndarray, n_dash: int, C: int, P: Callable[[int, float], float]):
  return (-1 / n_dash) * sum(sum(P(i, x_dash[m]) * np.log(P(i, x_dash[m])) for i in range(1, C)) for m in range(1, n_dash))
