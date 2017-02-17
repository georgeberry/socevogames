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
        strategies: list of strings, e.g. ['c', 'd']
        payoffs: matrix of payoffs as numpy or list, e.g. [[3, -1], [5, 0]]

    Behavior:
        name: set name to the name of the class instance

    """
    def __init__(self, strategies, payoffs):
        self.name = None
        self.strategies = strategies
        self.payoff_matrix = np.array(payoffs)
        self.name_to_idx_dict = {k:v for v,k in enumerate(strategies)}
        self.idx_to_name_dict = {k:v for v,k in self.name_to_idx_dict.items()}
        # checks
        self.check_consistency()

    def check_consistency(self):
        nrow, ncol = self.payoff_matrix.shape
        nstrat = len(self.strategies)
        assert 2 == nrow == ncol == nstrat, 'Dimension mismatch'

    def get_strat_idx(self, strat):
        return self.name_to_idx_dict[strat]

    def get_strat_name(self, idx):
        return self.idx_to_name_dict[idx]

    def payoff_from_idx(self, p1_strat_idx, p2_strat_idx):
        return self.payoff_matrix[p1_strat_idx, p2_strat_idx]

    def payoff_from_name(self, p1_strat_name, p2_strat_name):
        p1_strat_idx = self.get_strat_idx(p1_strat_name)
        p2_strat_idx = self.get_strat_idx(p2_strat_name)
        return self.payoff_matrix[p1_strat_idx, p2_strat_idx]

    def _column_operation(self, payoff_matrix):
        """
        Make Herbert Gintis swell with pride.
        """
        return payoff_matrix - payoff_matrix.diagonal()

    def __call__(self, ingroup_strat_idx, outgroup_strat_idx):
        return self.payoff_from_idx(ingroup_strat_idx, outgroup_strat_idx)

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
        name,
        idx,
        games,
        interactions,
    ):
        self.idx = idx
        self.name = name
        self.games = games
        self.interactions = interactions
        self.check_consistency()

    def check_consistency(self):
        assert len(self.games) == 2, 'Dimension mismatch'
        print(self.interactions.shape)
        assert self.interactions.shape == (2,), 'Dimension mismatch'

    def payoff(self, strat_idx, pop_state):
        """
        Inputs:
            strategy: str
            structure: matrix
            population_state: vector
        """
        s = strat_idx
        # ingroup is 1 or 0: 1 - 1 = 0, 1 - 0 = 1, you get other group
        # i stands for ingroup, o for outgroup
        p_i = pop_state[self.idx]
        p_o = pop_state[1 - self.idx]
        rate_i = self.interactions[self.idx]
        rate_o = self.interactions[1 - self.idx]
        game_i = self.games[self.idx]
        game_o = self.games[1 - self.idx]

        utility_i = rate_i * (p_i * game_i(s, 0) + (1 - p_i) * game_i(s, 1))
        utility_o = rate_o * (p_o * game_o(s, 0) + (1 - p_o) * game_o(s, 1))
        return utility_i + utility_o

    def avg_payoff(self, population_state, structure):
        avg = p * self.payoff(0, pop_state) + (1 - p) * self.payoff(1, pop_state)
        return avg

def make_groups(interaction_matrix, game_matrix, names=None):
    interaction_matrix = np.array(interaction_matrix)
    game_matrix = np.array(game_matrix)
    if not names:
        names = list(range(game_matrix.shape[0]))
    groups = []
    for grp_idx, name in enumerate(names):
        g = SocialGroup(
            name,
            grp_idx,
            game_matrix[grp_idx,:],
            interaction_matrix[grp_idx,:],
        )
        groups.append(g)
    return groups

class SocialGame(object):
    """
    Class for a social evolutionary game
    This class does solving and plotting

    The actual calculus functions are d_grp_dt, d_pop_dt, and sim
    Sim calls scipy integrate, which calls d_pop_dt, which calls d_grp_dt
    d_grp_dt actually calculates stuff based on payoffs

    Inputs:
        social_groups: list of social groups
        structure: 2 x 2 interaction matrix
        state: 1 x 2 vector [frac_group_1_c, frac_group_2_c]
    """
    def __init__(self, social_groups):
        assert len(social_groups) == 2
        self.social_groups = social_groups

    def d_grp_dt(self, grp_idx, pop_state):
        """
        Change for group grp_idx at state X
        """
        grp_state = pop_state[grp_idx]
        g = self.social_groups[grp_idx]
        payoff = g.payoff(0, pop_state)
        avg_payoff = g.avg_payoff(pop_state)
        return grp_state * (payoff - avg_payoff)

    def d_pop_dt(self, pop_state, t):
        """
        Change in system at state X and time
        t is required by scipy but is really pretty useless
        """
        return np.array(
            [self.d_grp_dt(0, pop_state), self.d_grp_dt(1, pop_state)]
        )

    def sim(self, init_conditions, **kwargs):
        """
        This will give you a trajectory for a start condition
        """
        t = np.linspace(**kwargs)
        return scipy.integrate(self.d_pop_dt, init_conditions, t)

    def plot_phase_space(self, title, path, fname):
        """
        We want to cover the
        """
        n_intervals = 20
        s = np.linspace(0, 1, n_intervals)
        x_mat, y_mat = np.meshgrid(s, s)
        # u, v are partial derivatives at each point on the grid
        u_mat, v_mat = np.zeros(x_mat.shape), np.zeros(y_mat.shape)
        for i in range(n_intervals):
            for j in range(n_intervals):
                x = x_mat[i, j]
                y = y_mat[i, j]
                d_pop_dt = self.d_pop_dt(np.array([x, y]), 0)
                u_mat[i, j] = d_pop_dt[0]
                v_mat[i, j] = d_pop_dt[1]

        # some matplolib args
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
    pd = StageGame(
        ['c', 'd'],
        [[3,-1],
         [5, 0]],
    )
    sh = StageGame(
        ['c', 'd'],
        [[3,0],
         [1,1]],
    )
    game_matrix = [[pd, sh],
                   [sh, pd]]
    interaction_matrix = [[0.5,0.5],
                          [0.5,0.5]]

    groups = make_groups(interaction_matrix, game_matrix)
    



    """
    def make_example_social_game():

        pd = StageGame()
        sh = StageGame([[3, 0],[0,1]])
        sg1 = SocialGroup(sh, pd)
        sg2 = SocialGroup(pd, sh)
        return SocialGame([sg1, sg2], structure)

    def example_phase_plot():
        g = make_example_social_game()
        g.plot_phase_space()

    def explore_biased_pd():

        pd with ingroup bias

        pd_payoff = np.array([[3,-1],[5,0]])

    example_phase_plot()
    """
