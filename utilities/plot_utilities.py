import os
from os.path import join as os_join
import json
import math

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.ticker as ticker
# ===============================================================================
# custom plot definitions
# return cm from inch
def cm2inch(value):
    return value / 2.54

# paper size for a4 landscape
height_9 = cm2inch(4.5)
width_14 = cm2inch(7)

# options
# for customizing check https://matplotlib.org/users/customizing.html
params = {
            'text.usetex': False,
            'font.size': 6,
            'font.family': 'lmodern',
            # 'text.latex.unicode': True,
            'figure.titlesize': 8,
            'figure.figsize': (width_14, height_9),
            'figure.dpi': 300,
            # 'figure.constrained_layout.use': True,
            # USE with the suplot_tool() to check which settings work the best
            'figure.subplot.left': 0.15,
            'figure.subplot.bottom': 0.15,
            'figure.subplot.right': 0.95,
            'figure.subplot.top': 0.9,
            'figure.subplot.wspace': 0.225,
            'figure.subplot.hspace': 0.35,
            #
            'axes.titlesize': 8,
            'axes.titlepad': 6,
            'axes.labelsize': 6,
            'axes.labelpad': 4,
            'axes.grid': 'True',
            'axes.grid.which': 'both',
            'axes.xmargin': 0.1,
            'axes.ymargin': 0.1,
            'lines.linewidth': 0.5,
            'lines.markersize': 5,
            'xtick.labelsize': 6,
            'ytick.labelsize': 6,
            'ytick.minor.visible': False,
            'xtick.minor.visible': False,
            'grid.linestyle': '-',
            'grid.linewidth': 0.25,
            'grid.alpha': 0.5,
            'legend.fontsize': 6,
            'savefig.dpi': 300,
            'savefig.format': 'pdf',
            'savefig.bbox': 'tight'
        }

# custom set up for the lines in plots 
LINE_TYPE_SETUP = {"color":          ["blue", "darkgreen", "red", "green", "slategrey", "darkmagenta","olive"],
                "linestyle":      ["solid",    "dashed",  "dashdot",    "dotted",   "(offset,on-off-dash-seq)",   ":"],
                "marker":         ["o",    "s",  "^",    "p",   "x", "*", "+"],
                "markeredgecolor": ["blue", "darkgreen", "red", "green", "slategrey", "darkmagenta","olive"],
                "markerfacecolor": ["blue", "darkgreen", "red", "green", "slategrey", "darkmagenta","olive"],
                "markersize":     [0.15,      1,    2,      3,    4,    4]}

class plot_data_general:
    def __init__(self, np_array_1 = None, np_array_2 = None, np_array_3 = None , savefile_name  = None , labels = None , axis_label = None):

        # direct input
        plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
        plt.rcParams.update(params)

        # cp plots 
        fig_cf = plt.figure()
        #fig_cf.suptitle('Cp at 2/3 H')
        n_row = 1
        n_col = 1
        gs = gridspec.GridSpec(n_row, n_col)
        ax_cf = [[fig_cf.add_subplot(gs[i, j]) for i in range(n_row)]
                        for j in range(n_col)]

        # ======================================================

        simulTime = np_array_1
        data_1 = np_array_2
        data_2 = np_array_3

        x_min  = np.amin(simulTime)
        x_max  = np.amax(simulTime)
        y_min  = np.amin(data_1)
        y_max  = np.amax(data_2)

        # plot force coefficients 
        # ax_cf[0][0].set_title("Error comparison")
        ax_cf[0][0].plot(simulTime, data_1 ,label= labels[0],
            color=LINE_TYPE_SETUP["color"][5],
            linestyle=LINE_TYPE_SETUP["linestyle"][0],
            # marker=LINE_TYPE_SETUP["marker"][0],
            markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][5],
            markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][5],
            markersize=LINE_TYPE_SETUP["markersize"][0])
        
        ax_cf[0][0].plot(simulTime, data_2 ,label=labels[1],
            color=LINE_TYPE_SETUP["color"][6],
            linestyle=LINE_TYPE_SETUP["linestyle"][1],
            marker=LINE_TYPE_SETUP["marker"][4],
            markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][6],
            markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][6],
            markersize=LINE_TYPE_SETUP["markersize"][0])
        
        ax_cf[0][0].tick_params(axis="x",direction="in",which = 'both')
        ax_cf[0][0].tick_params(axis="y",direction="in",which = 'both')
        ax_cf[0][0].set_xlabel(axis_label[0])
        ax_cf[0][0].set_ylabel(axis_label[1],labelpad=0.1)

        ax_cf[0][0].set_xlim([-4,1000])
        ax_cf[0][0].set_ylim([y_min-50, 200])
        ax_cf[0][0].grid(b=True, which = 'both')
        ax_cf[0][0].legend(loc = 1)

        # save figure 
        fig_cf.savefig(savefile_name)
        plt.show()
        plt.close(fig=fig_cf)