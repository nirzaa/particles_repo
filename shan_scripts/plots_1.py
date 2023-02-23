#!/usr/bin/env python3

import sys, os
import numpy, matplotlib
import matplotlib.pyplot as pyplot
# from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

pyplot.style.use("./shan_scripts/luxe.mplstyle")


def plotme_hist(xx, yy, fname):
    fig,ax = pyplot.subplots()
    ax.stairs(yy,xx, fill=True, color='b', label="A LUXE Histo")
    ax.stairs(yy,xx, fill=False, color='k') # redraw the outline in black
    ax.legend(loc=(0.625,0.8))  # defined by left-bottom of legend box; in the ratio of figure size
    ax.set_xlim(-3,3)
    ax.set_xlabel(r'$E_e$ [GeV]')
    ax.set_ylabel(r'd$N_e$/d$E_e$ [1/GeV]')
    ax.text(0.05,0.9,"$LUXE$ TDR\nPython & matplotlib", fontweight='bold', \
        transform=ax.transAxes, verticalalignment='top')
    pyplot.savefig(fname)

def plotme_scatter(xx, yy, fname):
    fig,ax = pyplot.subplots()
    ax.scatter(yy,xx, color='k', label="A LUXE Histo")
    ax.legend(loc=(0.625,0.8))  # defined by left-bottom of legend box; in the ratio of figure size
    ax.set_xlim(-3,3)
    ax.set_xlabel(r'$E_e$ [GeV]')
    ax.set_ylabel(r'd$N_e$/d$E_e$ [1/GeV]')
    ax.text(0.05,0.9,"$LUXE$ TDR\nPython & matplotlib", fontweight='bold', \
        transform=ax.transAxes, verticalalignment='top')
    pyplot.savefig(fname)


if __name__ == '__main__':
    # Generate data set
    data1 = numpy.random.normal(0,1, 10000)
    data2 = numpy.random.normal(0,1, 10000)
    yy1, xx = numpy.histogram(data1, bins=100, range=(-3.0,3.0))
    yy2, xx = numpy.histogram(data2, bins=100, range=(-3.0,3.0))
    yy = yy1
    plotme(xx, yy, './shan_scripts/luxeMatplotlib.pdf')