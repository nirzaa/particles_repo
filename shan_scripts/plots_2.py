#!/usr/bin/env python3

import sys, os
import numpy, matplotlib
import matplotlib.pyplot as pyplot
# from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# Generate data set
data1 = numpy.random.normal(0,1, 10000)
data2 = numpy.random.normal(0,1, 10000)
yy1, xx = numpy.histogram(data1, bins=100, range=(-3.0,3.0))
yy2, xx = numpy.histogram(data2, bins=100, range=(-3.0,3.0))
residue = yy2-yy1
residue_normalised = residue/yy1

pyplot.style.use("luxe.mplstyle")
# pyplot.rcParams["text.usetex"] = True
# pyplot.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
# pyplot.rcParams["mathtext.fontset"] = "Latin Modern Math"

# pyplot.ion() # for interaction in terminal

fig = pyplot.figure(figsize=(11.025,11.025))
ax1 = pyplot.subplot(7,1,(1,5))
ax2 = pyplot.subplot(7,1,(6,7), sharex=ax1)
ax = ax1 # shared x axis

ax1.stairs(yy1,xx, fill=False, color='b', linestyle='-', label=r"$N_\mathregular{true}$")
ax1.stairs(yy2,xx, fill=False, color='k', linestyle='--', label=r"$N_\mathregular{recon}$")
ax2.stairs(residue_normalised,xx, color='k')
ax.set_xlim(-3,3)
ax2.set_ylim(-1,1)
for label in ax.get_xticklabels(): label.set_visible(False)
ax1.legend(loc=(0.7,0.7))  # defined by left-bottom of legend box; in the ratio of figure size
ax1.set_ylabel(r'd$N$/d$E_e$ [1/GeV]')
ax2.set_ylabel(r'$\Delta N$/$N_\mathregular{true}$', loc="center")
fig.align_ylabels([ax1,ax2])
ax2.set_xlabel(r'$E_e$ [GeV]')
ax2.set_yticks(numpy.arange(-1,1.1,0.5), labels=[None,None,"0",None,None])
ax2.set_yticks(numpy.arange(-1,1.1,0.1), \
    [None,None,r"$-0.8$",None,None,None,r"$-0.4$",None,None,None,"0", \
        None,None,None,"0.4",None,None,None,"0.8",None,None], \
        minor=True)
ax1.text(0.05,0.9,"$LUXE$ TDR", fontweight='bold', \
    transform=ax1.transAxes, verticalalignment='top')
ax1.text(0.05,0.8,"Python & matplotlib", \
    transform=ax1.transAxes, verticalalignment='top')
# ax.xaxis.set_minor_locator(AutoMinorLocator())
# ax.yaxis.set_minor_locator(AutoMinorLocator())
# xleft, xright = ax.get_xlim()
# ybottom, ytop = ax.get_ylim()
# ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*3/4)


pyplot.savefig('luxeMatplotlib_2.pdf')