#!/usr/bin/env python3
import scipy
import sys, os
import numpy, matplotlib
import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
# from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.stats import norm

pyplot.style.use("./shan_scripts/luxe.mplstyle")

layers = 10

def hist(xx, yy1, yy2, location, case=None):
    # yy1 is output, yy2 is target

    print()
    yy1[yy1 == np.inf] = 0
    yy1_mean = yy1[34:].mean()
    yy1[34:] = yy1_mean

    yy2[yy2 == np.inf] = 0
    yy2_mean = yy2[34:].mean()
    yy2[34:] = yy2_mean
    
    yy1 = np.array(yy1)
    yy2 = np.array(yy2)

    output_std = pd.read_csv(f'./shan_scripts/multiple_runs/case_{case}/output_std.csv')
    target_std = pd.read_csv(f'./shan_scripts/multiple_runs/case_{case}/target_std.csv')

    residue = numpy.array(yy2)-numpy.array(yy1)
    residue_normalised = residue/yy2

    std_t = target_std['target_std']
    std_o = output_std['output_std']
    mean_t = target_std['target_mean']
    mean_o = output_std['output_mean']
    multiply = output_std['multiply_mean']


    cov = multiply - mean_o * mean_t
    correlation = cov / (std_t * std_o)
    delta = 1/(yy2**2)*(std_o**2) + ((yy1**2) / (yy2**4)) * (std_t**2) - 2*(yy1/yy2**3)*cov
    delta_root = np.sqrt(delta)

    delta_root[34:] = 0 # there is no meaning to mean of stds

    # ========== Save the Values ============= #

    df = pd.DataFrame({'cov': cov, 'delta': delta, 'correlation' : correlation, 'delta_root': delta_root})
    df.to_csv(f"./shan_scripts/multiple_runs/case_{case}/cov_delta.csv", index=False)
    
    # ======================================== #

    pyplot.style.use("./shan_scripts/luxe.mplstyle")
    fig = pyplot.figure(num=123, figsize=(14.025,14.025))
    ax1 = pyplot.subplot(7,1,(1,5))
    ax2 = pyplot.subplot(7,1,(6,7), sharex=ax1)
    ax = ax1 # shared x axis
    ax.set_ylim(0,50000)
    data1 = numpy.random.normal(0,1, 10000)
    _, xx = numpy.histogram(data1, bins=48, range=(1,13))
    xx_errorbar = np.linspace(1.125, 12.875, 48) # + 12 / 48 / 2 = 0.125
    yy1 /= ((13-1) / yy1.shape[0])
    yy2 /= ((13-1) / yy2.shape[0])
    ax1.stairs(yy1,xx, fill=False, color='b', linestyle='-', label=r"$N_\mathregular{true}$")
    ax1.errorbar(xx_errorbar, yy1, yerr=output_std['output_std'], linestyle='none')
    ax1.errorbar(xx_errorbar, yy2, yerr=target_std['target_std'], linestyle='none')

    ax1.stairs(yy2,xx, fill=False, color='k', linestyle='--', label=r"$N_\mathregular{recon}$")
    ax2.stairs(residue_normalised,xx, color='k')
    ax2.errorbar(xx_errorbar, residue_normalised, yerr=delta_root, linestyle='none')
    ax.set_xlim(xx[0],xx[-1])
    # ax2.set_ylim(-1,1)
    for label in ax.get_xticklabels(): label.set_visible(False)
    ax1.legend(loc=(0.7,0.7))  # defined by left-bottom of legend box; in the ratio of figure size
    ax1.set_ylabel(r'd$N$/d$E_e$ [1/GeV]')
    ax2.set_ylabel(r'$(N_{gen} - N_{rec})$/$N_\mathregular{true}$', loc="center")
    fig.align_ylabels([ax1,ax2])
    ax2.set_xlabel(r'$E_e$ [GeV]')
    ax2.set_ylim(-.2,.2)
    ax2.set_yticks(numpy.arange(-.2,.21,.1), labels=[None,None,"0",None,None])
    ax2.set_yticks(numpy.arange(-.2,.21,0.02), \
        [None,None,r"$-.16$",None,None,None,r"$-0.08$",None,None,None,"0", \
            None,None,None,"0.08",None,None,None,".16",None,None], \
            minor=True)
    ax1.text(0.05,0.9,"$LUXE$ CNN", \
        transform=ax1.transAxes, verticalalignment='top')
    ax1.text(0.05,0.8,"e-laser IPstrong ECAL", \
        transform=ax1.transAxes, verticalalignment='top')
    ax1.text(0.05,0.6,f"180 BXs {layers} first layers", \
        transform=ax1.transAxes, verticalalignment='top')

    pyplot.savefig(location)

def loss(xx, yy, fname):
    fig,ax = pyplot.subplots()
    ax.scatter(xx,yy, color='k', label="Loss function")
    ax.legend(loc=(0.625,0.8))  # defined by left-bottom of legend box; in the ratio of figure size
    # ax.set_xlim(-3,3)
    ax.set_ylim(0,600)
    ax.set_xlabel(r'Epochs')
    ax.set_ylabel(r'MSELoss')
    ax.text(0.05,0.9,"$LUXE$ CNN\ne-laser IPstrong ECAL", \
        transform=ax.transAxes, verticalalignment='top')
    ax.text(0.05,0.7,f"180 BXs {layers} first layers", \
        transform=ax.transAxes, verticalalignment='top')
    pyplot.savefig(fname)

def projection(xx, yy, fname):
    fig,ax = pyplot.subplots()
    ax.stairs(yy,xx, fill=True, color='b', label="Projection")
    ax.stairs(yy,xx, fill=False, color='k') # redraw the outline in black
    ax.legend(loc=(0.625,0.8))  # defined by left-bottom of legend box; in the ratio of figure size
    ax.set_xlim(-.5,.5)
    ax.set_ylim(0,45)
    ax.set_xlabel(r'$(N_{rec} - N_{gen})/N_{gen}$')
    ax.set_ylabel(r'Occurrences')
    ax.text(0.05,0.9,"$LUXE$ CNN\ne-laser IPstrong ECAL", \
        transform=ax.transAxes, verticalalignment='top')
    ax.text(0.05,0.7,f"180 BXs {layers} first layers", \
        transform=ax.transAxes, verticalalignment='top')
    pyplot.savefig(fname)

def projection_sandbox(xx, yy, fname, data, bins):
    fig,ax = pyplot.subplots()

    mean,std=norm.fit(data)
    plt.hist(data, bins=bins, density=True, label='Projection')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    y = norm.pdf(x, mean, std)
    y /= y.sum()
    plt.plot(x, y, 'r--', linewidth=2, label='Gaussian')
    fname1 = fname[:-4] + '_myplot.png'
    plt.ylabel('Density')
    plt.xlabel(r'$E_{rec}[GeV] / E^{tot}_{dep}[MeV]$')
    plt.text(0.05,0.9,f"mean={round(mean,2)}, std={round(std,2)}", \
        transform=ax.transAxes, verticalalignment='top')
    plt.savefig(fname1)
    plt.clf()

    ax.stairs(yy,xx, fill=True, color='b', label="Projection")
    ax.stairs(yy,xx, fill=False, color='k') # redraw the outline in black
    ax.legend(loc=(0.625,0.8))  # defined by left-bottom of legend box; in the ratio of figure size
    # ax.set_xlim(-.5,.5)
    # ax.set_ylim(0,45)
    ax.set_xlabel(r'$E_{rec}[GeV] / E^{tot}_{dep}[MeV]$')
    ax.set_ylabel(r'Occurrences')
    ax.text(0.05,0.9,"$LUXE$ CNN\ne-laser IPstrong ECAL", \
        transform=ax.transAxes, verticalalignment='top')
    ax.text(0.05,0.7,f"180 BXs {layers} first layers", \
        transform=ax.transAxes, verticalalignment='top')
    pyplot.savefig(fname)

def rel_error(xx, yy, fname):
    fig,ax = pyplot.subplots()
    ax.scatter(xx,yy, color='k', label="Relative error")
    ax.legend(loc=(0.625,0.8))  # defined by left-bottom of legend box; in the ratio of figure size
    # ax.set_xlim(-3,3)
    ax.set_xlabel(r'Epochs')
    ax.set_ylabel(r'$(N_{rec} - N_{gen})/N_{gen}$')
    ax.set_ylim(-10,10)
    ax.text(0.05,0.9,"$LUXE$ CNN\ne-laser IPstrong ECAL", \
        transform=ax.transAxes, verticalalignment='top')
    ax.text(0.05,0.7,f"180 BXs {layers} first layers", \
        transform=ax.transAxes, verticalalignment='top')
    pyplot.savefig(fname)

def tot(xx, yy, fname):
    avg = yy[xx > 200].mean()
    rms = np.sqrt(np.mean(yy[xx > 200]**2))
    fig,ax = pyplot.subplots()
    ax.scatter(xx,yy, color='k')
    ax.legend(loc=(0.625,0.8))  # defined by left-bottom of legend box; in the ratio of figure size
    ax.set_xlim(0,3500)
    ax.set_ylim(-1,1)
    ax.set_xlabel(r'Multiplicity')
    ax.set_ylabel(r'$(N_{rec} - N_{gen})/N_{gen}$')
    ax.text(0.45,0.9,"$LUXE$ CNN\ne-laser IPstrong ECAL", \
        transform=ax.transAxes, verticalalignment='top', horizontalalignment='left')
    ax.text(0.45,0.7,f"180 BXs {layers} first layers", \
        transform=ax.transAxes, verticalalignment='top', horizontalalignment='left')
    ax.text(0.45,0.3,f"Average={round(avg,2)}, RMS={round(rms,2)}", \
        transform=ax.transAxes, verticalalignment='top', horizontalalignment='left')
    pyplot.savefig(fname)

def ratio(xx, yy, fname):
    fig,ax = pyplot.subplots()
    # ax.scatter(xx,yy, color='k', label="E[GeV](output) / PixelSum")
    ax.scatter(xx,yy, color='k')
    ax.legend(loc=(0.625,0.8))  # defined by left-bottom of legend box; in the ratio of figure size
    # ax.set_xlim(-3,3)
    ax.set_ylim(0,400)
    ax.set_xlabel(r'$E_{gen}[GeV]$')
    ax.set_ylabel(r'$E_{rec}[GeV] / E^{tot}_{dep}[MeV]$')
    ax.text(0.45,0.9,"$LUXE$ CNN\ne-laser IPstrong ECAL", \
        transform=ax.transAxes, verticalalignment='top')
    ax.text(0.45,0.7,f"180 BXs {layers} first layers", \
        transform=ax.transAxes, verticalalignment='top')
    pyplot.savefig(fname)


def image_hist(location, yy1, num_events):
    pyplot.style.use("./shan_scripts/luxe.mplstyle")
    fig = pyplot.figure(num=123, figsize=(14.025,14.025))
    ax1 = pyplot.subplot(7,1,(1,5))
    ax = ax1 # shared x axis
    # ax.set_ylim(0,50000)
    data1 = numpy.random.normal(0,1, 10000)
    _, xx = numpy.histogram(data1, bins=48, range=(1,13))
    xx_errorbar = np.linspace(1.125, 12.875, 48) # + 12 / 48 / 2 = 0.125
    yy1 /= ((13-1) / yy1.shape[0])
    ax1.stairs(yy1,xx, fill=False, color='b', linestyle='-', label=r"$N_\mathregular{true}$")

    ax.set_xlim(xx[0],xx[-1])
    # ax2.set_ylim(-1,1)
    for label in ax.get_xticklabels(): label.set_visible(False)
    ax1.legend(loc=(0.7,0.7))  # defined by left-bottom of legend box; in the ratio of figure size
    ax1.set_ylabel(r'd$N$/d$E_e$ [1/GeV]')
    ax1.set_xlabel(r'$E_e$ [GeV]')
    ax1.text(0.05,0.9,"$LUXE$ CNN", \
        transform=ax1.transAxes, verticalalignment='top')
    ax1.text(0.05,0.8,"e-laser IPstrong ECAL", \
        transform=ax1.transAxes, verticalalignment='top')
    ax1.text(0.05,0.6,f"Number of events = {num_events}", \
        transform=ax1.transAxes, verticalalignment='top')
    

    pyplot.savefig(location)

