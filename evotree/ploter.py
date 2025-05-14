import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import logging

tableau_colors = plt.get_cmap("tab10")  # 'tab10' is the Tableau 10 color palette


def get_totalH(Hs):
    CHF = 0
    for i in Hs: CHF = CHF + i
    return CHF

def addvvline(ax,xvalue,color,lstyle,labell,lw):
    if labell == '': ax.axvline(xvalue,color=color, ls=lstyle, lw=lw)
    else: ax.axvline(xvalue,color=color, ls=lstyle, lw=lw, label=labell)

def ordinary_hist(y,bins=50,outfile='Ordinary_hist.pdf',xlabel='Number of persisted polyploid species',ylabel='Number of iterations'):
    fig, ax = plt.subplots(figsize=(5, 5))
    Hs, Bins, patches = ax.hist(y, bins=bins, color='gray', linewidth=0.5, edgecolor="white")
    ax.set_xlabel(xlabel,fontsize=15)
    ax.set_ylabel(ylabel,fontsize=15)
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close()

def bar_hist(xs,ys,outfile='Persisted_Polyploids_Over_Time.pdf',xlabel='Time (million years)', ylabel='Number of survived polyploid species',legends=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(xs, ys, color=tableau_colors(0))
    #ax.set_xticks(xs, labels=[str(i) for i in xs])
    ax.set_xlabel(xlabel,fontsize=15)
    ax.set_ylabel(ylabel,fontsize=15)
    if len(legends) !=0:
        for lable_,value in legends.items():
            ax.plot([],[],color='k',alpha=0,label='{}: {}'.format(lable_,value))
    ax.legend(loc=0,fontsize=15,frameon=False)
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close()

def agedistributiondrawer(ages,plotkde=False,fitexpon=False,outfile='',legends={},ax=None,letter=None,wgdage=None): # Assuming age bounded within [0,5] (resembling Ks bounds)
    y = np.array(ages)
    if len(y)==0:
        logging.info("No survival genes")
        return
    #bounds = [np.floor(min(ages)),np.ceil(max(ages))]
    bounds = [0,5]
    nbins = 50
    bins = np.linspace(0, bounds[1], num=nbins+1)
    if ax is None: fig, ax = plt.subplots(figsize=(5, 5))
    #Hs, Bins, patches = ax.hist(y, bins=bins, color='gray', rwidth=0.8)
    color_ = tableau_colors(0)
    #Hs, Bins, patches = ax.hist(y, bins=bins, color=color_,edgecolor="white")
    Hs, Bins, patches = ax.hist(y, bins=bins, histtype='step',color='gray',lw=3)
    #ax.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.7)
    kdesity = 100
    kde_x = np.linspace(bounds[0],bounds[1],num=nbins*kdesity)
    min_bound = max([0,y.min()])
    kde_x = kde_x[kde_x > min_bound]
    kde = stats.gaussian_kde(y,bw_method=0.1)
    kde_y = kde(kde_x)
    CHF = get_totalH(Hs)
    scaling = CHF*0.1
    if plotkde: ax.plot(kde_x, kde_y*scaling, color='black',alpha=1, ls = '-')
    if fitexpon:
        loc, scale = stats.expon.fit(y)  # 'scale' is 1/λ, and 'loc' is the shift parameter
        expon_pdf = stats.expon.pdf(kde_x, loc=loc, scale=scale)
        ax.plot(kde_x, expon_pdf*scaling, color=color_,alpha=1, ls = '-', lw=3, label='Exponential fit')
    ax.set_xlabel('Age',fontsize=15)
    ax.set_ylabel('Number of retained genes',fontsize=15)
    ax.set_xticks([0,1,2,3,4,5])
    ax.set_title('Gene age distribution',fontsize=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if len(legends) !=0:
        for lable_,value in legends.items():
            ax.plot([],[],color='k',alpha=0,label='{}: {}'.format(lable_,value))
    if wgdage is not None:
        color_wgd = tableau_colors(3)
        addvvline(ax,wgdage,color_wgd,'-',"WGD age: {}".format(wgdage),3)
    ax.legend(loc=0,fontsize=15,frameon=False)
    if letter is not None: ax.text(-0.115, 1.05, "{})".format(letter), transform=ax.transAxes, fontsize=15, weight='bold')
    if ax is None:
        fig.tight_layout()
        fig.savefig(outfile)
        plt.close()
