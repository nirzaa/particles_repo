#!/usr/bin/env python3

import sys, os
import numpy, matplotlib
import matplotlib.pyplot as pyplot
# from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

pyplot.style.use("./shan_scripts/luxe.mplstyle")

layers = 20

def hist(xx, yy1, yy2, location):

    residue = numpy.array(yy2)-numpy.array(yy1)
    residue_normalised = residue/yy1

    pyplot.style.use("./shan_scripts/luxe.mplstyle")
    fig = pyplot.figure(num=123, figsize=(14.025,14.025))
    ax1 = pyplot.subplot(7,1,(1,5))
    ax2 = pyplot.subplot(7,1,(6,7), sharex=ax1)
    ax = ax1 # shared x axis

    data1 = numpy.random.normal(0,1, 10000)
    _, xx = numpy.histogram(data1, bins=48, range=(1,13))
    yy1 /= ((13-1) / 48)
    yy2 /= ((13-1) / 48)
    ax1.stairs(yy1,xx, fill=False, color='b', linestyle='-', label=r"$N_\mathregular{true}$")
    ax1.stairs(yy2,xx, fill=False, color='k', linestyle='--', label=r"$N_\mathregular{recon}$")
    ax2.stairs(residue_normalised,xx, color='k')
    ax.set_xlim(xx[0],xx[-1])
    # ax2.set_ylim(-1,1)
    for label in ax.get_xticklabels(): label.set_visible(False)
    ax1.legend(loc=(0.7,0.7))  # defined by left-bottom of legend box; in the ratio of figure size
    ax1.set_ylabel(r'd$N$/d$E_e$ [1/GeV]')
    ax2.set_ylabel(r'$\Delta N$/$N_\mathregular{true}$', loc="center")
    fig.align_ylabels([ax1,ax2])
    ax2.set_xlabel(r'$E_e$ [GeV]')
    ax2.set_yticks(numpy.arange(-2,2.1,1), labels=[None,None,"0",None,None])
    ax2.set_yticks(numpy.arange(-2,2.1,0.2), \
        [None,None,r"$-1.6$",None,None,None,r"$-0.8$",None,None,None,"0", \
            None,None,None,"0.8",None,None,None,"1.6",None,None], \
            minor=True)
    ax1.text(0.05,0.9,"$LUXE$ CNN", \
        transform=ax1.transAxes, verticalalignment='top')
    ax1.text(0.05,0.8,"e-laser IPstrong ECAL", \
        transform=ax1.transAxes, verticalalignment='top')
    ax1.text(0.05,0.6,f"180 BXs {layers} layers", \
        transform=ax1.transAxes, verticalalignment='top')

    pyplot.savefig(location)

def loss(xx, yy, fname):
    fig,ax = pyplot.subplots()
    ax.scatter(xx,yy, color='k', label="Loss function")
    ax.legend(loc=(0.625,0.8))  # defined by left-bottom of legend box; in the ratio of figure size
    # ax.set_xlim(-3,3)
    ax.set_xlabel(r'Epochs')
    ax.set_ylabel(r'MSELoss')
    ax.text(0.05,0.9,"$LUXE$ CNN\ne-laser IPstrong ECAL", \
        transform=ax.transAxes, verticalalignment='top')
    ax.text(0.05,0.7,f"180 BXs {layers} layers", \
        transform=ax.transAxes, verticalalignment='top')
    pyplot.savefig(fname)

def projection(xx, yy, fname):
    fig,ax = pyplot.subplots()
    ax.stairs(yy,xx, fill=True, color='b', label="Projection")
    ax.stairs(yy,xx, fill=False, color='k') # redraw the outline in black
    ax.legend(loc=(0.625,0.8))  # defined by left-bottom of legend box; in the ratio of figure size
    ax.set_xlim(-.5,.5)
    ax.set_xlabel(r'(Nout - Ntrue)/Ntrue [%]')
    ax.set_ylabel(r'Occurences')
    ax.text(0.05,0.9,"$LUXE$ CNN\ne-laser IPstrong ECAL", \
        transform=ax.transAxes, verticalalignment='top')
    ax.text(0.05,0.7,f"180 BXs {layers} layers", \
        transform=ax.transAxes, verticalalignment='top')
    pyplot.savefig(fname)

def rel_error(xx, yy, fname):
    fig,ax = pyplot.subplots()
    ax.scatter(xx,yy, color='k', label="Relative error")
    ax.legend(loc=(0.625,0.8))  # defined by left-bottom of legend box; in the ratio of figure size
    # ax.set_xlim(-3,3)
    ax.set_xlabel(r'Epochs')
    ax.set_ylabel(r'(Nout - Ntarget)/Ntarget [%]')
    ax.text(0.05,0.9,"$LUXE$ CNN\ne-laser IPstrong ECAL", \
        transform=ax.transAxes, verticalalignment='top')
    ax.text(0.05,0.7,f"180 BXs {layers} layers", \
        transform=ax.transAxes, verticalalignment='top')
    pyplot.savefig(fname)

def tot(xx, yy, fname):
    fig,ax = pyplot.subplots()
    ax.scatter(xx,yy, color='k')
    ax.legend(loc=(0.625,0.8))  # defined by left-bottom of legend box; in the ratio of figure size
    # ax.set_xlim(-3,3)
    ax.set_xlabel(r'Multicipies')
    ax.set_ylabel(r'(Nout - Ntrue)/Ntrue [%]')
    ax.text(0.05,0.9,"$LUXE$ CNN\ne-laser IPstrong ECAL", \
        transform=ax.transAxes, verticalalignment='top')
    ax.text(0.05,0.7,f"180 BXs {layers} layers", \
        transform=ax.transAxes, verticalalignment='top')
    pyplot.savefig(fname)



