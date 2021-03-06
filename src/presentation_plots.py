from social_replicator import *

#### Setup params ##############################################################


rand_structure = [[0.5, 0.5],
                  [0.5, 0.5]]
high_h_structure = [[0.8, 0.2],
                    [0.2, 0.8]]

pd_pd_matrix = [[pd, pd],
                [pd, pd]]
sh_pd_matrix = [[sh, pd],
                [pd, sh]]
ck_pd_matrix = [[ck, pd],
                [pd, ck]]
sh_sh_matrix = [[sh, sh],
                [sh, sh]]
sh_ck_matrix = [[sh, ck],
                [ck, sh]]
ck_ck_matrix = [[ck, ck],
                [ck, ck]]


sg = SocialGame(groups)
sg.plot_phase_space()

#### PD ########################################################################

pd_grp1 = SocialGroup(pd, pd)
pd_grp2 = SocialGroup(pd, pd)

# pure pd #
pd_no_h = SocialGame([pd_grp1, pd_grp2], rand_structure)
pd_no_h.plot_phase_space(
    title="Prisoner's dilemma, random mixing",
    path="../plots/",
    fname="pd_no_h",
)

# with homophily #
pd_high_h = SocialGame([pd_grp1, pd_grp2], high_h_structure)
pd_high_h.plot_phase_space(
    title="Prisoner's dilemma, high ingroup association",
    path="../plots/",
    fname="pd_high_h",
)

"""
#### IGPD ######################################################################

## pd with ingroup bias ##
igpd_grp1 = SocialGroup(pd_bias, pd)
igpd_grp2 = SocialGroup(pd, pd_bias)

igpd_no_h_high_p = SocialGame([igpd_grp1, igpd_grp2], no_h_struct)
igpd_no_h_high_p.plot_phase_space(
    title='PD with ingroup bias, No Homophily',
    path="../plots/",
    fname="pd_no_h",
)
igpd_low_h_high_p = SocialGame([igpd_grp1, igpd_grp2], med_h_structure)
igpd_low_h_high_p.plot_phase_space(
    title='PD with ingroup bias, 0.4 Homophily',
    path="../plots/",
    fname="pd_no_h",
)
igpd_high_h_high_p = SocialGame([igpd_grp1, igpd_grp2], high_h_struct)
igpd_high_h_high_p.plot_phase_space(
    title='PD with ingroup bias, 0.8 Homophily',
    path="../plots/",
    fname="pd_no_h",
)
"""

#### SH ########################################################################

sh_grp1 = SocialGroup(sh, sh)
sh_grp2 = SocialGroup(sh, sh)

# pure pd #
sh_no_h = SocialGame([sh_grp1, sh_grp2], rand_structure)
sh_no_h.plot_phase_space(
    title="Stag hunt, random mixing",
    path="../plots/",
    fname="sh_no_h",
)

# with homophily #
sh_high_h = SocialGame([sh_grp1, sh_grp2], high_h_structure)
sh_high_h.plot_phase_space(
    title="Stag hunt, high ingroup association",
    path="../plots/",
    fname="sh_high_h",
)

#### CK ########################################################################

ck_grp1 = SocialGroup(ck, ck)
ck_grp2 = SocialGroup(ck, ck)

# pure pd #
ck_no_h = SocialGame([ck_grp1, ck_grp2], rand_structure)
ck_no_h.plot_phase_space(
    title="Chicken, random mixing",
    path="../plots/",
    fname="ck_no_h",
)

# with homophily #
ck_high_h = SocialGame([ck_grp1, ck_grp2], high_h_structure)
ck_high_h.plot_phase_space(
    title="Chicken, high ingroup association",
    path="../plots/",
    fname="ck_high_h",
)

#### PD v SH ###################################################################

sh_pd_grp1 = SocialGroup(sh, pd)
sh_pd_grp2 = SocialGroup(pd, sh)

# pure pd #
sh_pd_no_h = SocialGame([sh_pd_grp1, sh_pd_grp2], rand_structure)
sh_pd_no_h.plot_phase_space(
    title="SH-PD, random mixing",
    path="../plots/",
    fname="sh_pd_no_h",
)

# with homophily #
sh_pd_high_h = SocialGame([sh_pd_grp1, sh_pd_grp2], high_h_structure)
sh_pd_high_h.plot_phase_space(
    title="SH-PD, high ingroup association",
    path="../plots/",
    fname="sh_pd_high_h",
)

#### PD v CK ###################################################################

ck_pd_grp1 = SocialGroup(ck, pd)
ck_pd_grp2 = SocialGroup(pd, ck)

# pure pd #
ck_pd_no_h = SocialGame([ck_pd_grp1, ck_pd_grp2], rand_structure)
ck_pd_no_h.plot_phase_space(
    title="CK-PD, random mixing",
    path="../plots/",
    fname="ck_pd_no_h",
)

# with homophily #
ck_pd_high_h = SocialGame([ck_pd_grp1, ck_pd_grp2], high_h_structure)
ck_pd_high_h.plot_phase_space(
    title="CK-PD, high ingroup association",
    path="../plots/",
    fname="ck_pd_high_h",
)

#### SH v CK ###################################################################

sh_ck_grp1 = SocialGroup(sh, ck)
sh_ck_grp2 = SocialGroup(ck, sh)

# pure pd #
sh_ck_no_h = SocialGame([sh_ck_grp1, sh_ck_grp2], rand_structure)
sh_ck_no_h.plot_phase_space(
    title="SH-CK, random mixing",
    path="../plots/",
    fname="sh_ck_no_h",
)

# with homophily #
sh_ck_high_h = SocialGame([sh_ck_grp1, sh_ck_grp2], high_h_structure)
sh_ck_high_h.plot_phase_space(
    title="SH-CK, high ingroup association",
    path="../plots/",
    fname="sh_ck_high_h",
)

#### Biased PD #################################################################

pd_bias_grp1 = SocialGroup(pd, pd_bias)
pd_bias_grp2 = SocialGroup(pd_bias, pd)

# pure pd #
sh_ck_no_h = SocialGame([pd_bias_grp1, pd_bias_grp2], rand_structure)
sh_ck_no_h.plot_phase_space(
    title="Bias PD, random mixing",
    path="../plots/",
    fname="bias_pd_no_h",
)

# with homophily #
sh_ck_high_h = SocialGame([pd_bias_grp1, pd_bias_grp2], high_h_structure)
sh_ck_high_h.plot_phase_space(
    title="Bias PD, high ingroup association",
    path="../plots/",
    fname="bias_pd_high_h",
)
