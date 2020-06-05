import os

import numpy as np
import matplotlib.pyplot as plt
from numba import njit


class Ising:
    def __init__(self, config):
        self.config = config
        self.d = config["d"]
        self.L = config["L"]
        self.β = config["β"]
        self.n_sweeps = config["n_sweeps"]
        self.seed = config["seed"]
        self.thermalization_cutoff = config["thermalization_cutoff"]
        self.save_path = config.get("save_path", "/tmp/ising/data")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        np.random.seed(self.config["seed"])
        self._initialize_lattice()

    def _initialize_lattice(self):
        dims = (self.L,) * self.d
        if self.config["start_type"] == "cold":
            self.σ = np.ones(dims)
        elif self.config["start_type"] == "hot":
            self.σ = np.random.choice([-1, 1], dims)

    def _process_observables(self):
        N = self.L ** self.d

        cutoff = int(self.n_sweeps * self.thermalization_cutoff)
        E = self.E[cutoff:]
        M = self.M[cutoff:]

        e = (E).mean() / N
        e_sq = (E ** 2).mean() / N ** 2
        c = self.β ** 2 * E.var() / N

        m = np.abs(M).mean() / N
        m_sq = (M ** 2).mean() / N ** 2
        χ = self.β * np.abs(M).var() / N

        data = {
            "e": e,
            "e_sq": e_sq,
            "c": c,
            "m": m,
            "m_sq": m_sq,
            "χ": χ,
            "β": self.β,
            "N_samples": len(E),
            "L": self.L,
        }
        self.observables = data

    def plot_lattice(self, title=None, save_as=False):
        fig, ax = plt.subplots(dpi=144, figsize=(9, 9))
        ax.matshow(self.σ, cmap="gray_r", aspect="equal")
        ax.set(xticks=[], yticks=[])
        if title is not None:
            ax.set_title(title)
        plt.tight_layout()

        if save_as:
            plt.savefig(os.path.join(self.save_path, save_as))

    def metropolis(self):
        if self.d == 1:
            self.E, self.M = metropolis_1D(self.σ, self.β, self.n_sweeps, self.seed)
        elif self.d == 2:
            self.E, self.M = metropolis_2D(self.σ, self.β, self.n_sweeps, self.seed)
        self._process_observables()


@njit(boundscheck=True)
def metropolis_1D(σ, β, n_sweeps, seed):
    """
    Implementation of the metropolis algorithm for 1D linear lattices with periodic
    boundary conditions.

    Args:
        σ       : Lattice of spins σ[i] ∈ {-1, 1}
        β       : Inverse temperature
        n_sweeps: Number of Monte Carlo update sweeps
        seed    : RNG seed
    Returns:
        E: Array where E[i] is the ith energy observation
        M: Array where M[i] is the ith magnetization observation
    """
    np.random.seed(seed)
    E = np.zeros(n_sweeps)
    M = np.zeros(n_sweeps)
    L = σ.shape[0]

    for sweep in range(n_sweeps):
        for site in range(L):
            i = np.random.randint(0, L)

            sum_of_neighbors = 0
            sum_of_neighbors += σ[i - 1] if i > 0 else σ[L - 1]
            sum_of_neighbors += σ[i + 1] if i < L - 1 else σ[0]
            ΔE = 2 * σ[i] * sum_of_neighbors

            if ΔE <= 0 or np.random.rand() < np.exp(-β * ΔE):
                σ[i] *= -1

        E[sweep] = -np.sum(σ[:-1] * σ[1:]) - σ[0] * σ[L - 1]
        M[sweep] = np.sum(σ)

    return E, M


@njit(boundscheck=True)
def metropolis_2D(σ, β, n_sweeps, seed):
    """
    Implementation of the metropolis algorithm for 2D square lattices with periodic
    boundary conditions.

    Args:
        σ       : Lattice of spins σ[i] ∈ {-1, 1}
        β       : Inverse temperature
        n_sweeps: Number of Monte Carlo update sweeps
        seed    : RNG seed
    Returns:
        E: Array where E[i] is the ith energy observation
        M: Array where M[i] is the ith magnetization observation
    """
    np.random.seed(seed)
    E = np.zeros(n_sweeps)
    M = np.zeros(n_sweeps)
    L = σ.shape[0]

    for sweep in range(n_sweeps):
        for i in range(L):
            for j in range(L):
                sum_of_neighbors = 0
                sum_of_neighbors += σ[i - 1, j] if i > 0 else σ[L - 1, j]
                sum_of_neighbors += σ[i + 1, j] if i < L - 1 else σ[0, j]
                sum_of_neighbors += σ[i, j - 1] if j > 0 else σ[i, L - 1]
                sum_of_neighbors += σ[i, j + 1] if j < L - 1 else σ[i, 0]
                ΔE = 2 * σ[i, j] * sum_of_neighbors

                if ΔE <= 0 or np.random.rand() < np.exp(-β * ΔE):
                    σ[i, j] *= -1

        E[sweep] = (
            -np.sum(σ[:, 1:] * σ[:, :-1])
            - np.sum(σ[1:, :] * σ[:-1, :])
            - np.sum(σ[0, :] * σ[L - 1, :])
            - np.sum(σ[1:, 0] * σ[1:, L - 1])
        )
        M[sweep] = np.sum(σ)

    return E, M
