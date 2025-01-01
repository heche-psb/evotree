import logging
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve
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
    return covariance_matrix

def generaterandomtrait(tree):
    species = [tip.name for tip in tree.get_terminals()]
    traits_df = pd.DataFrame({'species': species, 'trait': np.random.randint(100, size=len(species))})
    traits = traits_df['trait'].values
    logging.info("\nSimulated traits:")
    logging.info(traits)
    return traits

# Log-Likelihood Function
def log_likelihood_BM(cov_matrix, tree, traits=None):
    """
    Calculate the log-likelihood for the Brownian Motion model.
    """
    if traits is None: traits = generaterandomtrait(tree)
    #else: traits = gettraitsfromfile(traits)
    n = len(traits)
    mean_trait = np.mean(traits)
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

def plottreedis(tree):
    TB,_ = plottree(treeobject=tree,fs=(10,10))
    TB.basicdraw()
    TB.drawscale(plotfulllengthscale=True,fullscaletickheight=0.1,fullscaleticklabeloffset=0.1)
    TB.saveplot('Cov_Tree.pdf')
    #TB.saveplot('Basic_tree.pdf')

def generaterandomrate(tree):
    tips = tree.get_terminals()
    for i,node in enumerate(tree.get_nonterminals()):
        node.name = "Node_{}".format(i)
    nodes = [i for i in tree.get_nonterminals() if i!= tree.root]
    taxa = [i.name for i in tips]
    hypothetical_intermediate_ancestors = [i.name for i in nodes]
    total_taxa = taxa + hypothetical_intermediate_ancestors
    logging.info("Checking duplicated tip IDs...")
    assert len(total_taxa) == len(set(total_taxa))
    logging.info("No duplicated tip or internal IDs detected\n")
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
    logging.info("Simulated means are {}\n".format(Simulated_means))
    Simulated_cov_matrix = np.cov(traits.T)
    logging.info("Theoretical covariances are {}".format(cov_matrix))
    logging.info("Simulated covariances are {}\n".format(Simulated_cov_matrix))
    return Ancestral_mean_vector,traits_dic

def plotstochasticprocess(traits,mean_vector,taxa,iteration):
    output = 'Trace_simulation.pdf'
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

def pbmmodeling(constantrate=False,tree=None,trait=None,output='PBMModeling.pdf'):
    logging.info("Start Phylogenetic Brownian Motion Modeling (PBMM) analysis\n...\n")
    if tree is None:
        Tree = stringttophylo(Test_tree)
    TB,_ = plottree(treeobject=Tree,fs=(10,10))
    TB.basicdraw()
    TB.drawscale(plotfulllengthscale=True,fullscaletickheight=0.1,fullscaleticklabeloffset=0.1)
    TB.saveplot('Raw_Tree.pdf')
    cov_matrix = get_covariance_matrix(Tree)
    logging.info("Covariance Matrix:")
    logging.info(cov_matrix)
    log_likelihood, sigma2_mle, trait = log_likelihood_BM(cov_matrix, Tree, traits=trait)
    logging.info("\n")
    logging.info(f"Log-Likelihood: {log_likelihood}")
    logging.info(f"MLE for sigma^2: {sigma2_mle}")
    if not constantrate: rates,rates_clades,total_taxa,tree_dis = generaterandomrate(Tree)
    iteration = 200
    ini_mean = 10
    evolutionary_rate = 1
    drift = 0
    if constantrate:
        cov_matrix, taxa = compute_cov_matrix(Tree, sigmasquare=evolutionary_rate)
    else:
        cov_matrix, taxa = compute_cov_matrix_variable(total_taxa, tree_dis)
    logging.info("Phylogenetic Variance-Covariance Matrix:")
    logging.info("{}\n".format(cov_matrix))
    mean_vector, traits = simulate_traits(cov_matrix, taxa, mean=ini_mean, iteration=iteration, mu=drift)
    logging.info("In total {} iterations".format(iteration))
    plotstochasticprocess(traits,mean_vector,taxa,iteration)
    logging.info("Done")
