import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import logging
from evotree.simulatepbmm import addstats


tableau_colors = plt.get_cmap("tab10")  # 'tab10' is the Tableau 10 color palette



class Tracer:
    def __init__(self,data=None,usedata=[],n_row=1,n_col=1,n_chains=1,fs=(5,5)):
        """
        :data is expected to be a *.tsv file with column names of parameters and iterations as the index
        """
        self.data = pd.read_csv(data,header=0,index_col=0,sep='\t')
        self.cols = [col for col in usedata]
        self.n_row = n_row; self.n_col = n_col; self.n_chains = n_chains; self.figsize = fs
        assert len(usedata) <= n_row*n_col
        for col in usedata: assert col in self.data.columns

    def basic_draw(self,bin_lims=[],n_bins=[],barcolors=[],alphas=[],bw_method='silverman',kdecolors=[],kdealphas=[],lss=[],lws=[],titles=[],fontsizes=[],decimal=2):
        fig, axes = plt.subplots(self.n_row, self.n_col, figsize = self.figsize)
        self.fig = fig
        if self.n_row*self.n_col <= 1: axes = [axes]
        self.axes = axes
        y = lambda x:format(float(x), f".{decimal}f")
        for i, col, ax in zip(range(len(axes)),self.cols,axes):
            samples = np.array(self.data.loc[:,col])
            dic_stats = {"Mean":[],"Median":[],"Equal-tail 5th CI":[], "Equal-tail 95th CI":[],"HPD 5th CI":[], "HPD 95th CI":[],"ESS":[]}
            if self.n_chains > 1: dic_stats.update({"R_hat":[]})            
            dic_stats = addstats(dic_stats,samples,self.n_chains)
            if len(bin_lims) >= i+1 and len(bin_lims[i])>1: maxi_,mini_ = bin_lims[i]
            else: maxi_,mini_ = np.max(samples)*1.1,np.min(samples)*0.9
            if len(n_bins) >= i+1 and len(n_bins[i])>0: bins_ = n_bins[i]
            else: bins_ = 50
            if len(barcolors) >= i+1 and len(barcolors[i])>0: color_ = len(barcolors[i])
            else: color_ = "gray"
            if len(alphas) >= i+1 and len(alphas[i])>0: alpha_ = alphas[i]
            else: alpha_ = 0.8
            Hs, Bins, patches = ax.hist(samples, bins = np.linspace(mini_,maxi_,num=bins_),color=color_, alpha=alpha_, rwidth=0.8,label='Posterior samples')
            CHF = get_totalH(Hs)
            scaling = CHF*(maxi_-mini_)/bins_
            kde_x = np.linspace(mini_,maxi_,num=bins_*10)
            kde_y=stats.gaussian_kde(samples,bw_method=bw_method).pdf(kde_x)
            if len(kdecolors) >= i+1 and len(kdecolors[i])>0: kdecolor_ = len(kdecolors[i])
            else: kdecolor_ = "black"
            if len(kdealphas) >= i+1 and len(kdealphas[i])>0: kdealpha_ = kdealphas[i]
            else: kdealpha_ = 0.8
            if len(lss) >= i+1 and len(lss[i])>0: ls_ = lss[i]
            else: ls_ = '-'
            if len(lws) >= i+1 and len(lws[i])>0: lw_ = lws[i]
            else: lw_ = 1
            ax.plot(kde_x, kde_y*scaling, color=kdecolor_,alpha=kdealpha_, ls=ls_, lw=lw_,label='KDE curve')
            if len(fontsizes) >= i+1 and len(fontsizes[i])>0: fontsize_ = fontsizes[i]
            else: fontsize_ = 10
            if len(titles) >= i+1 and len(titles[i])>0:
                title_ = titles[i]
                ax.set_title(title_,fontsize=fontsize_)
            else: plt.suptitle("Posterior distribution", fontsize=fontsize_)
            ax.axvline(x=dic_stats["Mean"][0], ymin=0, ymax=1, color="k", alpha=0.8, ls=':', lw=1, label="Mean: {}".format(y(dic_stats["Mean"][0])))
            ax.axvline(x=dic_stats["Median"][0], ymin=0, ymax=1, color="k", alpha=0.8, ls='--', lw=1, label="Median: {}".format(y(dic_stats["Median"][0])))
            mode, maxim = kde_mode(kde_x, kde_y)
            ax.axvline(x=mode, ymin=0, ymax=1, color="k", alpha=0.8, ls='-', lw=1, label="Mode: {}".format(y(mode)))
            ax.axvline(x=dic_stats["Equal-tail 5th CI"][0], ymin=0, ymax=1, color="k", alpha=0.8, ls='-.', lw=1, label="Equal-tail 5th CI: {}".format(y(dic_stats["Equal-tail 5th CI"][0])))
            ax.axvline(x=dic_stats["Equal-tail 95th CI"][0], ymin=0, ymax=1, color="k", alpha=0.8, ls='-.', lw=1, label="Equal-tail 95th CI: {}".format(y(dic_stats["Equal-tail 95th CI"][0])))
            ax.axvline(x=dic_stats["HPD 5th CI"][0], ymin=0, ymax=1, color="k", alpha=0.8, ls=(0,(3,1,1,1)), lw=1, label="HPD 5th CI: {}".format(y(dic_stats["HPD 5th CI"][0])))
            ax.axvline(x=dic_stats["HPD 95th CI"][0], ymin=0, ymax=1, color="k", alpha=0.8, ls=(0,(3,1,1,1)), lw=1, label="HPD 95th CI: {}".format(y(dic_stats["HPD 95th CI"][0])))
            ax.plot([],[],color='k',label='ESS:{}'.format(y(dic_stats["ESS"][0])),lw=1)
            if self.n_chains > 1: ax.plot([],[],color='k',label='R_hat:{}'.format(y(dic_stats["R_hat"][0])),lw=1)
            ax.legend(loc=0,fontsize=10,frameon=False)
            ax.set_xlabel(col);ax.set_ylabel("Number of samples")
            #ax.spines['top'].set_visible(False)
            #ax.spines['right'].set_visible(False)
    
    def saveplot(self,output="Posterior_Samples.pdf",**kwargs):
        self.fig.tight_layout()
        self.fig.savefig(output,**kwargs)

def kde_mode(kde_x, kde_y):
    maxy_iloc = np.argmax(kde_y)
    mode = kde_x[maxy_iloc]
    return mode, max(kde_y)


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
