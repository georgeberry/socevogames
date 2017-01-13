import numpy as np
import scipy as sp
from scipy import integrate
import matplotlib.pyplot as plt

"""
Social replicator simulation
"""

class StageGame(object):
    """
    A particular stage game

    Input:
        strategies: list of str
        payoffs: np.ndarray
        name: str
    """
    def __init__(
        self,
        payoffs,
        name=None,
        reverse=False
        ):
        self.name = name
        self.strategies = ['c', 'd']
        self.payoff_matrix = np.array(payoffs)
        assert self.payoff_matrix.shape == (2,2)

    def strat_idx(self, strat):
        return self.strategies.index(strat)

    def payoff(self, self_strat_str, opp_strat_str):
        self_strat_idx = self.strat_idx(self_strat_str)
        opp_strat_idx = self.strat_idx(opp_strat_str)
        return self.payoff_matrix[self_strat_idx, opp_strat_idx]

    def _column_operation(self, payoff_matrix):
        return payoff_matrix - payoff_matrix.diagonal()

    def __call__(self, self_strat_str, opp_strat_str):
        return self.payoff(self_strat_str, opp_strat_str)

class SocialGroup(object):
    """
    Contains the information for a social group

    Inputs:
        group_name: str
        ingroup_game: StageGame
        outgroup_game: StageGame
        structure: np array
    """
    def __init__(
        self,
        ingroup_game,
        outgroup_game,
        idx=None,
        name=None,
        ):
        self.idx = idx # reindexed later
        self.name = name # for convenience only
        self.ingroup_game = ingroup_game
        self.outgroup_game = outgroup_game

    def payoff(self, strategy, population_state, structure):
        """
        Inputs:
            strategy: str
            structure: matrix
            population_state: vector
        """
        s = strategy
        p1, p2 = population_state[0], population_state[1]
        i1, i2 = structure[self.idx, 0], structure[self.idx, 1]
        ig, og = self.ingroup_game, self.outgroup_game
        ingroup_utility = p1 * ig(s, 'c') + (1 - p1) * ig(s, 'd')
        outgroup_utility = p2 * og(s, 'c') + (1 - p2) * og(s, 'd')
        return i1 * ingroup_utility + i2 * outgroup_utility

    def avg_payoff(self, population_state, structure):
        p = population_state[self.idx]
        c_payoff = self.payoff('c', population_state, structure)
        d_payoff = self.payoff('d', population_state, structure)
        return p * c_payoff + (1 - p) * d_payoff

class SocialGame(object):
    """
    Class for a social evolutionary game
    This should rely on lower-level classes as much as possible

    This class does solving and plotting
    Stuff like iterating through spaces of init coniditions

    Inputs:
        social_groups: list of social groups
        structure: 2 x 2 interaction matrix
        state: 1 x 2 vector [frac_group_1_c, frac_group_2_c]

    We use X to represent the state of the system
    It should be a 2 vector, giving [p(1;c), p(2;c)]
    """
    def __init__(self, social_groups, structure):
        assert len(social_groups) == 2
        self.social_groups = social_groups
        self.reindex_social_groups()
        self.structure = np.array(structure)
        assert self.structure.shape == (2, 2)

    def reindex_social_groups(self):
        idx = 0
        for grp in self.social_groups:
            grp.idx = idx
            idx += 1

    def dXg_dt(self, g_idx, X):
        """
        Change for group g_idx at state X
        """
        Xg = X[g_idx]
        g = self.social_groups[g_idx]
        payoff = g.payoff('c', X, self.structure)
        avg_payoff = g.avg_payoff(X, self.structure)
        return Xg * (payoff - avg_payoff)

    def dX_dt(self, X, t):
        """
        Change in system at state X and time
        """
        return np.array([self.dXg_dt(0, X), self.dXg_dt(1, X)])

    def sim(self, X0):
        """
        Simulates with initial conditions
        """
        t = np.linspace()
        return scipy.integrate(self.dX_dt, X0, t)

    def plot_phase_space(self, title, path, fname):
        """
        We want to cover the
        """
        n_intervals = 20
        s = np.linspace(0, 1, n_intervals)
        x_mat, y_mat = np.meshgrid(s, s)
        # outputs
        u_mat, v_mat = np.zeros(x_mat.shape), np.zeros(y_mat.shape)
        for i in range(n_intervals):
            for j in range(n_intervals):
                x = x_mat[i, j]
                y = y_mat[i, j]
                dX_dt = self.dX_dt(np.array([x, y]), 0)
                u_mat[i, j] = dX_dt[0]
                v_mat[i, j] = dX_dt[1]

        axis_font = {'size':'24'}
        title_font = {'size': '24'}

        p = plt.quiver(x_mat, y_mat, u_mat, v_mat, color='b')
        plt.xlabel('$p^a$', **axis_font)
        plt.ylabel('$p^b$', **axis_font)
        plt.xlim([-.1, 1.1])
        plt.ylim([-.1, 1.1])
        plt.title(title, **title_font)
        plt.savefig(path + fname + '.pdf', format='pdf')
        plt.close()

if __name__ == '__main__':
    def make_example_social_game():
        structure = [[.5,.5],[.5,.5]]
        pd = StageGame([[3,-1],[5,0]])
        sh = StageGame([[3, 0],[0,1]])
        sg1 = SocialGroup(sh, pd)
        sg2 = SocialGroup(pd, sh)
        return SocialGame([sg1, sg2], structure)

    def example_phase_plot():
        g = make_example_social_game()
        g.plot_phase_space()

    def explore_biased_pd():
        """
        pd with ingroup bias
        """
        pd_payoff = np.array([[3,-1],[5,0]])

    example_phase_plot()
