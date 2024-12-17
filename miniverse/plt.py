import numpy as np
from matplotlib import pyplot as plt

from miniverse.bin_converter import prob_to_char


def make_plt(matrix: np.ndarray, matrix_names: list) -> plt:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.matshow(matrix, cmap='viridis')
    ax.set_yticks(np.arange(len(matrix)))
    ax.set_yticklabels(matrix_names)
    for (i, j), val in np.ndenumerate(matrix):
        txt = int(val + 0.5) if i == 0 else prob_to_char(val, num_digits=2)
        ax.text(j, i, txt, ha='center', va='center', color='white')
    return plt
