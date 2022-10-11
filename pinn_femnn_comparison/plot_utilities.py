import pylab as py
import os
from os.path import join as os_join
import json
import math
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
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
params = {  'text.usetex': False,
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
                "markersize":     [0.015,      1,    2,      3,    4,    4]}

def get_rms( series):
    mean = np.mean(series)
    fl_series = series-mean
    return np.sqrt(np.mean(fl_series**2))
        
class plot_data:
    '''
    This class is for making comparison plot of three variables. The variable on x-axis is given as np_array[0] and variables to compare are provided as 
    np_array[1] and np_array[2]. Labels for both the axes and the legend for the variable to compare are given as labels.   
    '''
    def __init__(self, np_array1 = None, np_array2 = None, np_array3 = None,  savefile_name  = None , labels = None  ):
        
        # custom rectangle size for figure layout
        cust_rect = [0.05, 0.05, 0.95, 0.95]

        # direct input
        plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
        plt.rcParams.update(params)

        offset_factor = 1.05
        pad_factor = 0.15

        # cp plots 
        fig_cf = plt.figure()
        #fig_cf.suptitle('Cp at 2/3 H')
        n_row = 1
        n_col = 1
        gs = gridspec.GridSpec(n_row, n_col)
        ax_cf = [[fig_cf.add_subplot(gs[i, j]) for i in range(n_row)]
                        for j in range(n_col)]
        # ======================================================

        Var1X = np_array1[:,0]
        Var1Y = np_array1[:,1]
        
        Var2X = np_array2[:,0]
        Var2Y = np_array2[:,1]
        
        Var3X = np_array3[:,0]
        Var3Y = np_array3[:,1]
              

        x_min  = np.amin(Var3X)
        x_max  = np.amax(Var3X)
        y_min  = np.amin(Var3Y)
        y_max  = np.amax(Var3Y)

        ax_cf[0][0].plot(Var1X, Var1Y ,label=labels[0],
            color=LINE_TYPE_SETUP["color"][0],
            linestyle=LINE_TYPE_SETUP["linestyle"][1],
            marker=LINE_TYPE_SETUP["marker"][1],
            markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][0],
            markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][0],
            markersize=LINE_TYPE_SETUP["markersize"][1])

        ax_cf[0][0].plot(Var2X, Var2Y ,label=labels[1],
            color=LINE_TYPE_SETUP["color"][2],
            linestyle=LINE_TYPE_SETUP["linestyle"][0],
            marker=LINE_TYPE_SETUP["marker"][5],
            markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][2],
            markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][2],
            markersize=LINE_TYPE_SETUP["markersize"][1])
        
        ax_cf[0][0].plot(Var3X, Var3Y ,label=labels[2],
            color=LINE_TYPE_SETUP["color"][3],
            linestyle=LINE_TYPE_SETUP["linestyle"][0],
            marker=LINE_TYPE_SETUP["marker"][3],
            markeredgecolor=LINE_TYPE_SETUP["markeredgecolor"][3],
            markerfacecolor=LINE_TYPE_SETUP["markerfacecolor"][3],
            markersize=LINE_TYPE_SETUP["markersize"][1])
              
        
        ax_cf[0][0].tick_params(axis="x",direction="in",which = 'both')
        ax_cf[0][0].tick_params(axis="y",direction="in",which = 'both')
        ax_cf[0][0].set_xlabel(labels[3])
        ax_cf[0][0].set_ylabel(labels[4],labelpad=0.001)
        print("§§§§§§§§§§§§§§§§§§§")
        # ax_cf[0][0].set_xticks(np.arange(x_min,x_max+1, step= 0.1))
        # ax_cf[0][0].set_yticks(np.arange(y_min,y_max+1, step= (y_max-y_min)/5))
        
        print("Limits are ", x_min, x_max, y_min, y_max)
        ax_cf[0][0].set_xticks(np.arange(-1.2, 1.2, step= 0.2))
        ax_cf[0][0].set_yticks(np.arange(y_min,-0.8, step=0.2))
        
        print("############################# 5 ")
        ax_cf[0][0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_cf[0][0].yaxis.set_minor_locator(ticker.AutoMinorLocator())

        end_time = 1

        ax_cf[0][0].set_xlim([-1.2,1.2])
        ax_cf[0][0].set_ylim([-2, -0.8])
        ax_cf[0][0].grid(b=False, which = 'minor')
        ax_cf[0][0].legend(loc = 1)

        # save figure 
        fig_cf.savefig(savefile_name)
        plt.show()
        plt.close(fig=fig_cf)
        print('plotting Caarc finished!')