import logging
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve
from numpy.linalg import slogdet
from Bio import Phylo
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import to_rgb
from scipy import stats
from scipy.optimize import minimize,minimize_scalar
from scipy.stats import multivariate_normal
import statsmodels.api as sm
from scipy.stats import t
import colorsys
import copy
from evotree.basicdraw import plottree
import arviz as az
import pymc as pm
import pytensor.tensor as pt
from matplotlib.colors import Normalize
Test_tree = "((((((((A:10,(B:1,C:1):9):1,D:11):4,E:15):3,F:18):12,G:30):11,H:41):2,I:43):3,J:46);"

def stringttophylo(string):
    handle = StringIO(string)
    tree = Phylo.read(handle, "newick")
    return tree

# Compute simple covariance matrix
def get_covariance_matrix(tree,taxa=None):
    if taxa is None: species = [tip.name for tip in tree.get_terminals()]
    else: species = taxa
    n = len(species)
    covariance_matrix = np.zeros((n, n))
    for i, sp1 in enumerate(species):
        for j, sp2 in enumerate(species):
            # Compute shared path length between sp1 and sp2
            mrca = tree.common_ancestor(sp1, sp2)
            covariance_matrix[i, j] = tree.distance(mrca)
    return covariance_matrix,species

def generaterandomtrait(tree):
    species = [tip.name for tip in tree.get_terminals()]
    traits_df = pd.DataFrame({'species': species, 'trait': np.random.randint(100, size=len(species))})
    traits = traits_df['trait'].values
    #logging.info("\nSimulated traits:")
    #logging.info(traits)
    return traits

# Log-Likelihood Function
def log_likelihood_BM(cov_matrix, traits, ancestralmean=None):
    """
    Calculate the log-likelihood for the Brownian Motion model.
    """
    #if traits is None: traits = generaterandomtrait(tree)
    #else: traits = gettraitsfromfile(traits)
    n = len(traits)
    mean_trait = np.mean(traits) if ancestralmean is None else ancestralmean
    diff = traits - mean_trait
    # Cholesky decomposition for numerical stability
    L = cholesky(cov_matrix, lower=True)
    # Solve for C^-1 * diff
    C_inv_diff = solve(L.T, solve(L, diff))
    # MLE for sigma^2
    sigma2_mle = np.dot(diff, C_inv_diff) / n
    # Log determinant of C
    log_det_C = 2 * np.sum(np.log(np.diag(L)))
    # Log-likelihood
    log_likelihood = -0.5 * (n * np.log(2 * np.pi) + log_det_C + n * np.log(sigma2_mle))
    return log_likelihood, sigma2_mle, traits

def log_likelihood_BM_cov_matrix(cov_matrix, traits):
    epsilon = 1e-6
    cov_matrix += np.eye(cov_matrix.shape[0]) * epsilon
    L = cholesky(cov_matrix, lower=True)
    C_inv_diff = np.linalg.solve(L.T, np.linalg.solve(L, traits))
    sign, log_det_C = slogdet(cov_matrix)
    if sign <= 0:
        return np.inf
    n = len(traits)
    log_likelihood_value = -0.5 * (n * np.log(2 * np.pi) + log_det_C + np.dot(traits.T, C_inv_diff))
    return -log_likelihood_value

def log_likelihood_BM_givenmean(ancestralmean, cov_matrix, traits):
    n = len(traits)
    diff = traits - ancestralmean
    # Cholesky decomposition for numerical stability
    L = cholesky(cov_matrix, lower=True)
    # Solve for C^-1 * diff
    C_inv_diff = solve(L.T, solve(L, diff))
    # MLE for sigma^2
    sigma2_mle = np.dot(diff, C_inv_diff) / n
    # Log determinant of C
    log_det_C = 2 * np.sum(np.log(np.diag(L)))
    # Log-likelihood
    log_likelihood = -0.5 * (n * np.log(2 * np.pi) + log_det_C + n * np.log(sigma2_mle))
    return -log_likelihood

def pdfancestralmean(cov_matrix, traits, output = 'PDF_ancestral_mean.pdf'):
    traits_ = [abs(i) for i in traits]
    limit = max(traits_)
    #xcoordinates = np.linspace(min(traits), max(traits),num=1000)
    xcoordinates = np.linspace(-limit,limit,num=1000)
    ycoordinates = np.array([-log_likelihood_BM_givenmean(x, cov_matrix, traits) for x in xcoordinates])
    mle_xcoordinate = xcoordinates[np.argmax(ycoordinates)]
    fig,ax = plt.subplots(1,1,figsize=(6,6))
    ax.plot(xcoordinates,ycoordinates,color='black',lw=3)
    ax.vlines(mle_xcoordinate,0,1,transform=ax.get_xaxis_transform(),lw=2,ls='--',color='black')
    y_min = ax.get_ylim()[0]
    ax.fill_between(xcoordinates,np.full(len(xcoordinates),y_min),ycoordinates,color='gray',alpha=0.4)
    ax.set_xlabel("Ancestral mean",fontsize=15)
    ax.set_ylabel("Log-likelihood",fontsize=15)
    ax.set_title("PDF of ancestral mean",fontsize=15)
    fig.tight_layout()
    fig.savefig(output)
    plt.close()

def mleancestralmean(cov_matrix, traits):
    traits_ = [abs(i) for i in traits]
    limit = max(traits_)
    result = minimize_scalar(
        log_likelihood_BM_givenmean,
        bounds=(-limit,limit),
        args=(cov_matrix, traits),
        method='bounded',options={"xatol": 0})
    mle_ancestralmean = result.x
    return mle_ancestralmean

def mlevariablerates(tree, traits, total_taxa):
    guess_rates = np.full(len(total_taxa),1)
    bounds = [(0, None) for _ in range(len(guess_rates))]
    result = minimize(
            givenratescalculpbbll,
            guess_rates,
            args=(tree, traits, total_taxa),
            method='L-BFGS-B',
            bounds=bounds)
    mle_rates = result.x
    #print("MLE Evolutionary Rates:", mle_rates)
    ratesdic = {clade:rate for clade,rate in zip(total_taxa,mle_rates)}
    cov_clades = compute_total_cov_matrix(tree,ratesdic)
    cov_matrix = fromcovcaldes2covmatrix(cov_clades,tree)
    total_cov_matrix = fromcovcaldes2totalcovmatrix(cov_clades,tree)
    negll = log_likelihood_BM_cov_matrix(cov_matrix, traits)
    return mle_rates,cov_matrix,total_cov_matrix,-negll

def givenratescalculpbbll(rates, tree, traits, total_taxa):
    ratesdic = {clade:rate for clade,rate in zip(total_taxa,rates)}
    cov_clades = compute_total_cov_matrix(tree,ratesdic)
    cov_matrix = fromcovcaldes2covmatrix(cov_clades,tree)
    negll = log_likelihood_BM_cov_matrix(cov_matrix, traits)
    return negll

def plottreedis(tree):
    TB,_ = plottree(treeobject=tree,fs=(10,10))
    TB.basicdraw(log="Plotting covariance tree")
    TB.drawscale(plotfulllengthscale=True,fullscaletickheight=0.1,fullscaleticklabeloffset=0.1)
    TB.saveplot('Cov_Tree.pdf')
    TB2,_ = plottree(treeobject=tree,fs=(10,10))
    TB2.polardraw(polar=355)
    TB2.saveplot('Cov_Tree_Circular.pdf')
    #TB.saveplot('Basic_tree.pdf')

def definetaxa(tree):
    Tree_dis = copy.deepcopy(tree)
    tips = Tree_dis.get_terminals()
    #nodes = [i for i in tree.get_nonterminals() if i!= tree.root]
    nodes = [i for i in Tree_dis.get_nonterminals()]
    for i,node in enumerate(Tree_dis.get_nonterminals()): node.name = "Node_{}".format(i)
    taxa = [i.name for i in tips]
    hypothetical_intermediate_ancestors = [i.name for i in nodes]
    total_taxa = taxa + hypothetical_intermediate_ancestors
    return np.array(taxa),np.array(total_taxa),len(taxa),len(hypothetical_intermediate_ancestors),Tree_dis

def Bayesianvariablerate(cov_matrix,tree,tiptraits,nodetraits,total_taxa,ancestralmean):
    # cov_matrix must include internal nodes too
    #tips = tree.get_terminals()
    #nodes = [i for i in tree.get_nonterminals() if i!= tree.root]
    #taxa = [i.name for i in tips]
    #hypothetical_intermediate_ancestors = [i.name for i in nodes]
    #total_taxa = taxa + hypothetical_intermediate_ancestors
    #tree_dis = copy.deepcopy(tree)
    #for i,node in enumerate(tree_dis.get_nonterminals()): node.name = "Node_{}".format(i)
    traits = np.concatenate((tiptraits,nodetraits),axis=None)
    with pm.Model() as model:
        mean_vector = np.full(len(total_taxa),ancestralmean)
        prior_rates = pm.HalfNormal("prior_rates", sigma=1,shape=len(total_taxa))
        rates_clades = {taxon:rate for taxon,rate in zip(total_taxa,prior_rates)}
        tree_dis = copy.deepcopy(tree)
        Phylo.draw_ascii(tree_dis)
        #for tip in tree_dis.get_terminals(): tip.branch_length = tip.branch_length * rates_clades[tip.name]
        #for node in tree_dis.get_nonterminals():
        #    if node.branch_length is None: node.branch_length = 0
        #    else: node.branch_length = node.branch_length * rates_clades[node.name]
        Phylo.draw_ascii(tree_dis)
        cov_matrix,_= compute_cov_matrix_variable(total_taxa, tree_dis)
        mvn = pm.MvNormal("mvn", mu=mean_vector, cov=cov_matrix, shape=len(mean_vector),observed=traits)
        trace = pm.sample(200, return_inferencedata=True,tune=200, chains=4)
        summary = az.summary(trace)
        print(summary)

def generaterandomrate(tree):
    tips = tree.get_terminals()
    for i,node in enumerate(tree.get_nonterminals()):
        if node.name is None: node.name = "Node_{}".format(i)
    nodes = [i for i in tree.get_nonterminals() if i!= tree.root]
    taxa = [i.name for i in tips]
    hypothetical_intermediate_ancestors = [i.name for i in nodes]
    total_taxa = taxa + hypothetical_intermediate_ancestors
    rates = np.abs(np.random.normal(loc=0, scale=1 ,size=len(total_taxa))) # Half normal
    rates_clades = {taxa:rate for taxa,rate in zip(total_taxa,rates)}
    tree_dis = copy.deepcopy(tree) # distance to the root is the variance
    for tip in tree_dis.get_terminals():
        tip.branch_length = tip.branch_length * rates_clades[tip.name]
    for node in tree_dis.get_nonterminals():
        if node.branch_length is None: node.branch_length = 0
        else: node.branch_length = node.branch_length * rates_clades[node.name]
    plottreedis(tree_dis)
    return rates,rates_clades,total_taxa,tree_dis

def compute_total_cov_matrix(tree,ratesdic):
    tips = tree.get_terminals()
    for i,node in enumerate(tree.get_nonterminals()):
        if node.name is None: node.name = "Node_{}".format(i)
    nodes = [i for i in tree.get_nonterminals() if i!= tree.root]
    taxa = [i.name for i in tips]
    hypothetical_intermediate_ancestors = [i.name for i in nodes]
    total_taxa = taxa + hypothetical_intermediate_ancestors
    tree_dis = copy.deepcopy(tree) # distance to the root is the variance
    cov_clades = {}
    for tip in tree_dis.get_terminals():
        tip.branch_length = tip.branch_length * ratesdic[tip.name]
    for node in tree_dis.get_nonterminals():
        if node.branch_length is None: node.branch_length = 0
        else: node.branch_length = node.branch_length * ratesdic[node.name]
    for tip in tree_dis.get_terminals():
        cov_clades[tip.name] = tree_dis.distance(tip)
    for node in tree_dis.get_nonterminals():
        cov_clades[node.name] = tree_dis.distance(node)
    return cov_clades

def fromcovcaldes2covmatrix(cov_clades,tree):
    tips = tree.get_terminals()
    taxa = [i.name for i in tips]
    n = len(taxa)
    cov_matrix = np.zeros((n, n))
    for i, taxon1 in enumerate(taxa):
        for j, taxon2 in enumerate(taxa):
            mrca = tree.common_ancestor(taxon1, taxon2)
            cov_matrix[i, j] = cov_clades[mrca.name]
    return cov_matrix

def fromcovcaldes2totalcovmatrix(cov_clades,tree):
    tips = tree.get_terminals()
    nodes = tree.get_nonterminals()
    taxa = [i.name for i in tips] + [j.name for j in nodes]
    n = len(taxa)
    cov_matrix = np.zeros((n, n))
    for i, taxon1 in enumerate(taxa):
        for j, taxon2 in enumerate(taxa):
            mrca = tree.common_ancestor(taxon1, taxon2)
            cov_matrix[i, j] = cov_clades[mrca.name]
    return cov_matrix

def get_cov_matrix_givensiga(tree, taxa, sigmasquare=1.0):
    n = len(taxa)
    cov_matrix = np.zeros((n, n))
    for i, taxon1 in enumerate(taxa):
        for j, taxon2 in enumerate(taxa):
            mrca = tree.common_ancestor(taxon1, taxon2)
            shared_time = tree.distance(mrca)
            cov_matrix[i, j] = sigmasquare * shared_time
    return cov_matrix

# Compute the phylogenetic variance-covariance matrix
def compute_cov_matrix(tree, sigmasquare=1.0,special=False):
    # sigmasquare : The evolutionary rate
    tips = tree.get_terminals()
    taxa = [i.name for i in tips]
    if special: taxa += ['Polypodiales']
    logging.info("Checking duplicated tip IDs...")
    assert len(taxa) == len(set(taxa))
    logging.info("No duplicated tip IDs detected\n")
    n = len(taxa)
    cov_matrix = np.zeros((n, n))
    for i, taxon1 in enumerate(taxa):
        for j, taxon2 in enumerate(taxa):
            # Compute the shared branch length (the distance bewteen mrca and root)
            mrca = tree.common_ancestor(taxon1, taxon2)
            shared_time = tree.distance(mrca)
            cov_matrix[i, j] = sigmasquare * shared_time
    return cov_matrix, taxa

# Compute the phylogenetic variance-covariance matrix under variable-rate PBM model incorporating ancestral nodes
def compute_cov_matrix_variable(total_taxa, tree_dis):
    n = len(total_taxa)
    cov_matrix = np.zeros((n, n))
    for i, taxon1 in enumerate(total_taxa):
        for j, taxon2 in enumerate(total_taxa):
            # Compute the distance (variance) bewteen mrca and root of tree_dis
            mrca = tree_dis.common_ancestor(taxon1, taxon2)
            cov_matrix[i, j] = tree_dis.distance(mrca)
    return cov_matrix,total_taxa

# Simulate traits under PBM model
def simulate_traits(cov_matrix, taxa, mean=0, iteration=100, mu=0, variantmean=False):
    # Mean vector for the traits
    if variantmean:
        Ancestral_mean_vector = mean
    else:
        Ancestral_mean_vector = np.full(len(taxa), mean)
    logging.info("Ancestral means are {}".format(Ancestral_mean_vector))
    # Add drift term (mu * t) Note: mu = 0 is equal to no drift
    drift_vector = mu * np.ones(len(taxa))
    mean_vector = drift_vector + Ancestral_mean_vector
    # Simulate trait values from a multivariate normal distribution
    traits = np.random.multivariate_normal(mean_vector, cov_matrix, size=iteration)
    traits_dic = {taxa[i]:traits[:,i] for i in range(len(taxa))}
    Simulated_means = traits.mean(axis=0)
    logging.info("Theoretical means are {}".format(mean_vector))
    logging.info("Simulated means are {}".format(Simulated_means))
    Simulated_cov_matrix = np.cov(traits.T)
    #logging.info("Theoretical covariances are {}".format(cov_matrix))
    #logging.info("Simulated covariances are {}\n".format(Simulated_cov_matrix))
    return Ancestral_mean_vector,traits_dic

def plotstochasticprocess(traits,mean_vector,taxa,iteration,output=None):
    if output is None: output = 'Trace_simulation.pdf'
    scaler_width = np.ceil(iteration/100)
    if scaler_width > 2: scaler_width = 2
    fig, ax = plt.subplots(1,1,figsize=(12*scaler_width, 6))
    xcoordinates, ycoordinates = [],[]
    colors = cm.viridis(np.linspace(0, 0.75, len(mean_vector)))
    colors = [adjust_saturation(i,0.8) for i in colors]
    for sp,ini_mean,co in zip(taxa,mean_vector,colors):
        trait_values = traits[sp]
        ycoordinates = np.hstack((np.array([ini_mean]),np.array(trait_values)))
        xcoordinates = np.arange(len(ycoordinates))
        ax.plot(xcoordinates,ycoordinates,lw=2,label=sp,alpha=0.8,color=co)
        ax.plot(xcoordinates[-1],ycoordinates[-1],marker='o',markersize=5,color=co)
        ax.hlines(ini_mean,xcoordinates.min(),xcoordinates.max(),color='k')
    ax.plot([],[],color='k',label='Ancestral trait')
    #ax.legend(loc=0,fontsize=15,frameon=False)
    ax.legend(loc=0,fontsize=15,bbox_to_anchor=(1, 1))
    ax.set_xlabel("Time",fontsize=15)
    ax.set_ylabel("Trait value",fontsize=15)
    fig.tight_layout()
    fig.savefig(output)
    plt.close()

def adjust_saturation(color_name, saturation_factor):
    rgb = to_rgb(color_name)
    r, g, b = rgb
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s = max(0, min(1, s * saturation_factor))
    return colorsys.hls_to_rgb(h, l, s)

def gettrait(trait,taxa,traitcolname=None):
    df = pd.read_csv(trait,header=0,index_col=0,sep='\t')
    Trait = {}
    for sp,tr in zip(df.index,df[traitcolname]):
        Trait[sp] = tr
    logging.info("Trait data:")
    logging.info(Trait)
    return np.array([Trait[taxon] for taxon in taxa])

def kde_mode(kde_x, kde_y):
    maxy_iloc = np.argmax(kde_y)
    mode = kde_x[maxy_iloc]
    return mode, max(kde_y)

def p_value_symbols(p):
    if p >= 0.05: return "ns"
    elif p>=0.01: return "*"
    elif p>=0.001: return "**"
    else: return "***"

def plotsimulationagainstreal(Traits,traits,taxa):
    n = len(taxa)
    fig, axes = plt.subplots(n,1,figsize=(9, 3*n))
    colors = cm.viridis(np.linspace(0, 0.75, len(taxa)))
    colors = [adjust_saturation(i,0.8) for i in colors]
    for sp,realtrait,co,ax in zip(taxa,Traits,colors,axes):
        trait_values = traits[sp]
        kde_x = np.linspace(min(trait_values),max(trait_values),num=1000)
        kde_y = stats.gaussian_kde(trait_values,bw_method='scott').pdf(kde_x)
        ax.plot(kde_x,kde_y,lw=2,color=co,label='Simulation')
        ax.fill_between(kde_x, np.zeros(len(kde_y)), kde_y, alpha=0.3, color=co)
        stat, p_value_w = stats.wilcoxon(trait_values-realtrait)
        p_marker = p_value_symbols(p_value_w)
        mode, maxim = kde_mode(kde_x, kde_y)
        ax.axvline(x = mode, color = 'gray', alpha = 1, ls = '-.', lw = 2,label='Simulated mode {:.2f}'.format(mode))
        ax.axvline(x = realtrait, color = 'k', alpha = 1, ls = '--', lw = 2,label='Observed trait {:.2f}'.format(realtrait)+''.join([r"$^{}$".format(i) for i in p_marker]))
        ax.set_title(sp,fontsize=15)
        ax.legend(loc=1,fontsize=15,frameon=False,bbox_to_anchor=(1, 1))
        ax.set_xlabel("Trait value",fontsize=15)
        ax.set_ylabel("Density",fontsize=15)
    fig.tight_layout()
    output = 'Stats_Sim_vs_Obs_Trait.pdf'
    fig.savefig(output)
    plt.close()

def compute_mle_internal_means(Tree,total_taxa,ntips,nnodes,traits,ancestralmean,total_cov_matrix):
    tip_indices = np.arange(ntips)
    internal_indices = np.arange(nnodes) + len(tip_indices)
    # Extract submatrices and subvectors
    C_tt = total_cov_matrix[np.ix_(tip_indices, tip_indices)]  # Tip-to-tip covariance
    C_it = total_cov_matrix[np.ix_(internal_indices, tip_indices)]  # Internal-to-tip covariance
    # Compute MLE for internal node means
    residual = traits - ancestralmean  # Center tip traits around root mean
    mle_means = ancestralmean + C_it @ solve(C_tt, residual)
    nodenames = total_taxa[internal_indices]
    return mle_means,nodenames

class PBMMBuilder:
    def __init__(self,tree=None,treeobject=None,trait=None,traitcolname=None,output=None,traitobject=None,traitname='Trait'):
        if tree is None:
            self.Tree = treeobject
        else:
            self.Tree = Phylo.read(tree,format='newick')
        self.trait=trait;self.traitcolname=traitcolname;self.output=output;self.traitobject=traitobject;self.traitname=traitname
        self.definetotaltaxa()
        self.gettraitob()
    def infervariablepbmm(self):
        self.getvariableratecov()
        self.variable_mle_ancestralmean = self.getmleancestralmean(self.variable_cov_matrix,output='Variable_pdfancestralmean.pdf')
        self.variable_mle_node_means,self.variable_node_names = self.getmlenodesmean(self.variable_mle_ancestralmean,self.variable_total_cov_matrix)
        self.drawratetree()
    def inferconstantpbmm(self):
        self.getconstantratecovtaxa()
        self.constant_mle_ancestralmean = self.getmleancestralmean(self.constant_cov_matrix,output='Constant_pdfancestralmean.pdf')
        self.getmlesigmacovll()
        self.constant_total_cov_matrix = gettotal_cov(self.Tree,self.total_taxa,sigma2=self.constant_sigma2_mle)
        self.constant_mle_node_means,self.constant_node_names = self.getmlenodesmean(self.constant_mle_ancestralmean,self.constant_total_cov_matrix)
        self.drawalltrait(output='Tree_MLE_Trait.pdf')
    def getmlesigmacovll(self):
        self.constant_ll, self.constant_sigma2_mle,_ = log_likelihood_BM(self.constant_cov_matrix, self.Trait, ancestralmean=self.constant_mle_ancestralmean)
        self.constant_mle_cov_matrix = get_cov_matrix_givensiga(self.Tree, self.taxa, sigmasquare=self.constant_sigma2_mle)
        logging.info(f"Constant rate PBMM LL: {self.constant_ll}")
        logging.info(f"PBMM cov matrix: {self.constant_mle_cov_matrix}")
    def getconstantratecovtaxa(self):
        self.constant_cov_matrix,_ = get_covariance_matrix(self.Tree)
    def gettraitob(self):
        if self.trait is None:
            if traitobject is None:
                self.Trait = generaterandomtrait(self.Tree)
            else:
                self.Trait = traitobject
        else: self.Trait = gettrait(self.trait,self.taxa,traitcolname=self.traitcolname)
    def definetotaltaxa(self):
        self.taxa,self.total_taxa,self.ntips,self.nnodes,self.Tree_nodenamed = definetaxa(self.Tree)
    def drawratetree(self,output='Tree_Rate_Annotation.pdf',traitname='Trait'):
        TB,_ = plottree(treeobject=self.Tree_nodenamed,fs=(10,10))
        TB.topologylw = 3
        colors, norm, colormap = TB.transformcm(self.mle_rates)
        ubrobject = TB.getubrobject(colors,self.total_taxa)
        TB.ubrobject = ubrobject
        TB.basicdraw(log='Plotting raw tree')
        TB.addcolorbar(norm,colormap,TB.ax,TB.fig,fraction=0.05, pad=0.04,text="Evolutionary rate",fontsize=15)
        TB.drawscale(plotfulllengthscale=True,fullscaletickheight=0.1,fullscaleticklabeloffset=0.1)
        tiptrait_dic = {ta:tr for ta,tr in zip(self.taxa,self.Trait)}
        nodetrait_dic = {ta:"{:.4f}".format(tr) for ta,tr in zip(self.variable_node_names,self.variable_mle_node_means)}
        TB.drawtrait(traitobject=[tiptrait_dic],traitobjectname=[traitname],xoffset=0.28,yoffset=0.2,labeloffset=0.2,traitcolor='gray')
        TB.addtext2node(nodetrait_dic,textxoffset=0.01,textsize=10,textalpha=1,textstyle='normal',textcolor='k')
        TB.saveplot(output)
    def getvariableratecov(self,output='MLE_variable_clade_rate.pdf'):
        self.mle_rates,self.variable_cov_matrix,self.variable_total_cov_matrix,self.variable_ll = mlevariablerates(self.Tree_nodenamed, self.Trait, self.total_taxa)
        plotmlecladerates(self.mle_rates,self.total_taxa,output)
        logging.info(f"Variable rate PBMM LL: {self.variable_ll}")
        logging.info(f"PBMM cov matrix: {self.variable_cov_matrix}")
    def getmleancestralmean(self,cov_matrix,output='pdfancestralmean.pdf'):
        mle_ancestralmean = mleancestralmean(cov_matrix,self.Trait)
        pdfancestralmean(cov_matrix, self.Trait, output=output)
        return mle_ancestralmean
    def getmlenodesmean(self,mle_ancestralmean,total_cov_matrix):
        mle_node_means,node_names = compute_mle_internal_means(self.Tree_nodenamed,self.total_taxa,self.ntips,self.nnodes,self.Trait,mle_ancestralmean,total_cov_matrix)
        logging.info(f"MLE for PBMM ancestral mean: {mle_ancestralmean}")
        logging.info("Observed mean: {}".format(self.Trait.mean()))
        return mle_node_means,node_names
    def drawalltrait(self,traitname='Trait',output='Tree_MLE_Trait.pdf'):
        TB,_ = plottree(treeobject=self.Tree_nodenamed,fs=(10,10))
        TB.topologylw = 3
        TB.basicdraw(log='Plotting raw tree')
        TB.drawscale(plotfulllengthscale=True,fullscaletickheight=0.1,fullscaleticklabeloffset=0.1)
        tiptrait_dic = {ta:tr for ta,tr in zip(self.taxa,self.Trait)}
        nodetrait_dic = {ta:"{:.4f}".format(tr) for ta,tr in zip(self.constant_node_names,self.constant_mle_node_means)}
        TB.drawtrait(traitobject=[tiptrait_dic],traitobjectname=[traitname],xoffset=0.28,yoffset=0.2,labeloffset=0.2,traitcolor='gray')
        TB.addtext2node(nodetrait_dic,textxoffset=0.01,textsize=10,textalpha=1,textstyle='normal',textcolor='k')
        TB.saveplot(output)

def plotmlecladerates(mle_rates,total_taxa,output):
    fig,ax = plt.subplots(1,1,figsize=(6,6))
    xcoordinates = np.arange(len(mle_rates))
    ax.bar(xcoordinates,mle_rates,width=0.8,align='center')
    ax.set_xticks(xcoordinates,labels=total_taxa,rotation=90)
    ax.set_xlabel("Clade",fontsize=15)
    ax.set_ylabel("Trait evolutionary rate",fontsize=15)
    fig.tight_layout()
    fig.savefig(output)
    plt.close()

def gettotal_cov(tree,total_taxa,sigma2=1):
    rates = np.full(len(total_taxa),sigma2)
    ratesdic = {clade:rate for clade,rate in zip(total_taxa,rates)}
    cov_clades = compute_total_cov_matrix(tree,ratesdic)
    total_cov_matrix = fromcovcaldes2totalcovmatrix(cov_clades,tree)
    return total_cov_matrix

def pbmmodeling(**kwargs):
    PB = PBMMBuilder(**kwargs)
    logging.info("\nConstant rate model\n...")
    PB.inferconstantpbmm()
    logging.info("\nVariable rate model\n...")
    PB.infervariablepbmm()

def standalone_pbmmodeling(tree=None,trait=None,traitcolname=None,output='PBMModeling.pdf'):
    logging.info("Start Phylogenetic Brownian Motion Modeling (PBMM) analysis\n...\n")
    if tree is None:
        Tree = stringttophylo(Test_tree)
    else:
        Tree = Phylo.read(tree,format='newick')
    cov_matrix,taxa = get_covariance_matrix(Tree)
    if trait is None: trait = generaterandomtrait(Tree)
    else: trait = gettrait(trait,taxa,traitcolname=traitcolname)
    total_taxa,ntips,nnodes,Tree_nodenamed = definetaxa(Tree)
    mle_rates,cov_matrix,total_cov_matrix,ll = mlevariablerates(Tree_nodenamed, trait, total_taxa)
    logging.info(f"Variable rate PBMM LL: {ll}")
    logging.info(f"PBMM cov matrix: {cov_matrix}")
    mle_ancestralmean = mleancestralmean(cov_matrix, trait)
    pdfancestralmean(cov_matrix, trait)
    mle_node_means,node_names = compute_mle_internal_means(Tree_nodenamed,total_taxa,ntips,nnodes,trait,mle_ancestralmean,total_cov_matrix)
    TB,_ = plottree(treeobject=Tree_nodenamed,fs=(10,10))
    TB.topologylw = 3
    TB.basicdraw(log='Plotting raw tree')
    TB.drawscale(plotfulllengthscale=True,fullscaletickheight=0.1,fullscaleticklabeloffset=0.1)
    tiptrait_dic = {ta:tr for ta,tr in zip(taxa,trait)}
    nodetrait_dic = {ta:str(tr) for ta,tr in zip(node_names,mle_node_means)}
    TB.drawtrait(traitobject=[tiptrait_dic],traitobjectname=['Species richness'],xoffset=0.25,yoffset=0.2,labeloffset=0.2)
    TB.addtext2node(nodetrait_dic,textxoffset=0.01,textsize=10,textalpha=1,textstyle='normal',textcolor='k')
    TB.saveplot('Raw_Tree_WithTrait.pdf')
    logging.info(f"MLE for PBMM ancestral mean: {mle_ancestralmean}")
    logging.info("Observed mean: {}".format(trait.mean()))

def test_pbmmodeling(variablerate=False,tree=None,trait=None,traitcolname=None,output='PBMModeling.pdf'):
    logging.info("Start Phylogenetic Brownian Motion Modeling (PBMM) analysis\n...\n")
    if tree is None:
        Tree = stringttophylo(Test_tree)
    else:
        Tree = Phylo.read(tree,format='newick')
    TB2,_ = plottree(treeobject=Tree,fs=(10,10))
    TB2.polardraw(polar=355)
    TB2.saveplot('Raw_Tree_Circular.pdf')
    cov_matrix,taxa = get_covariance_matrix(Tree)
    #logging.info("Covariance Matrix:")
    #logging.info(cov_matrix)
    if trait is None: trait = generaterandomtrait(Tree)
    else: trait = gettrait(trait,taxa,traitcolname=traitcolname)
    mle_ancestralmean = mleancestralmean(cov_matrix, trait)
    pdfancestralmean(cov_matrix, trait)
    total_taxa,ntips,nnodes,Tree_nodenamed = definetaxa(Tree)
    mle_node_means,node_names,total_cov_matrix = compute_mle_internal_means(Tree_nodenamed,total_taxa,ntips,nnodes,trait,mle_ancestralmean)
    mle_rates = mlevariablerates(Tree_nodenamed, trait, total_taxa)
    mle_rates_dic = {clade:rate for clade,rate in zip(total_taxa,mle_rates)}
    TB,_ = plottree(treeobject=Tree_nodenamed,fs=(10,10))
    TB.topologylw = 3
    TB.basicdraw(log='Plotting raw tree')
    TB.drawscale(plotfulllengthscale=True,fullscaletickheight=0.1,fullscaleticklabeloffset=0.1)
    #total_trait_dic = {**{ta:tr for ta,tr in zip(taxa,trait)},**{ta:tr for ta,tr in zip(node_names,mle_node_means)}}
    tiptrait_dic = {ta:tr for ta,tr in zip(taxa,trait)}
    nodetrait_dic = {ta:str(tr) for ta,tr in zip(node_names,mle_node_means)}
    TB.drawtrait(traitobject=[tiptrait_dic],traitobjectname=['Species richness'],xoffset=0.25,yoffset=0.2,labeloffset=0.2)
    TB.addtext2node(nodetrait_dic,textxoffset=0.01,textsize=10,textalpha=1,textstyle='normal',textcolor='k')
    TB.saveplot('Raw_Tree_WithTrait.pdf')
    #Bayesianvariablerate(total_cov_matrix,Tree_nodenamed,trait,mle_node_means,total_taxa,mle_ancestralmean)
    log_likelihood, sigma2_mle, trait = log_likelihood_BM(cov_matrix, trait, ancestralmean=mle_ancestralmean)
    #logging.info("\n")
    logging.info(f"PBMM Log-Likelihood: {log_likelihood}")
    logging.info(f"MLE for PBMM sigma^2: {sigma2_mle}")
    logging.info(f"MLE for PBMM ancestral mean: {mle_ancestralmean}")
    logging.info("Obseved mean: {}".format(trait.mean()))
    if variablerate: rates,rates_clades,total_taxa,tree_dis = generaterandomrate(Tree)
    iteration = 1000
    #ini_mean = 0 if trait is None else trait.mean()
    drift = 0
    if not variablerate:
        cov_matrix, taxa = compute_cov_matrix(Tree, sigmasquare=sigma2_mle)
    else:
        cov_matrix, taxa = compute_cov_matrix_variable(total_taxa, tree_dis)
    #logging.info("Phylogenetic Variance-Covariance Matrix:")
    #logging.info("{}\n".format(cov_matrix))
    mean_vector, sm_traits = simulate_traits(cov_matrix, taxa, mean=mle_ancestralmean, iteration=iteration, mu=drift)
    logging.info("In total {} iterations".format(iteration))
    #plotstochasticprocess(sm_traits,mean_vector,taxa,iteration,output=output)
    plotsimulationagainstreal(trait,sm_traits,taxa)
    logging.info("Done")
