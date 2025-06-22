import logging
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve
from scipy.linalg import cho_factor, cho_solve
from numpy.linalg import slogdet
from scipy.integrate import quad
from scipy.special import kv, gamma
from Bio import Phylo
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import to_rgb
from scipy import stats
from scipy.optimize import minimize,minimize_scalar
from scipy.stats import multivariate_normal,chi2
import statsmodels.api as sm
from scipy.stats import t
import colorsys
import copy
import itertools
from evotree.basicdraw import plottree
from statsmodels.tsa.stattools import acf
import arviz as az
#import pymc as pm
#import pytensor.tensor as pt
from matplotlib.colors import Normalize

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.scipy.linalg import cho_factor as jcho_factor
from jax.scipy.linalg import cho_solve as jcho_solve
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS




Test_tree = "((((((((A:10,(B:1,C:1):9):1,D:11):4,E:15):3,F:18):12,G:30):11,H:41):2,I:43):3,J:46);"

def variance_gamma_pdf(d, alpha, beta):
    term = (beta**(2*alpha) * np.abs(d)**(alpha - 0.5) * kv(alpha - 0.5, beta * np.abs(d)))
    denom = (np.sqrt(np.pi) * 2**(alpha - 1) * gamma(alpha))
    return term / denom

def getBFdenominator(single_pair,samples_df):
    p1, p2 = single_pair
    deltas = np.array(samples_df[p2] - samples_df[p1])
    kde = stats.gaussian_kde(deltas)
    posterior_density_at_0 = kde.evaluate(0.0)[0]
    return posterior_density_at_0

def lognormal_params_from_mean_var(mean, variance):
    """
    Compute loc (μ_log) and scale (σ_log) for a log-normal distribution.
    """
    sigma_log_sq = np.log(1 + variance / mean**2)
    sigma_log = np.sqrt(sigma_log_sq)
    mu_log = np.log(mean) - 0.5 * sigma_log_sq
    return {"loc": mu_log, "scale": sigma_log}

def calculateHPD(train_in,per):
    sorted_in = np.sort(train_in)
    upper_bound = np.percentile(train_in, per)
    lower_bound = np.percentile(train_in, 100-per)
    upper_bound_indice,lower_bound_indice = 0,0
    cutoff,candidates = int(np.ceil(per*len(sorted_in)/100)),[]
    for i,v in enumerate(sorted_in):
        if v >= upper_bound:
            upper_bound_indice = i
            break
    for i,v in enumerate(sorted_in):
        if v >= lower_bound:
            lower_bound_indice = i
            break
    for (x,y) in itertools.product(np.arange(0,lower_bound_indice,1,dtype=int), np.arange(upper_bound_indice,len(sorted_in),1,dtype=int)):
        if (y-x+1) >= cutoff: candidates.append((sorted_in[y] - sorted_in[x],(x,y)))
    lower,upper = sorted(candidates, key=lambda y: y[0])[0][1][0],sorted(candidates, key=lambda y: y[0])[0][1][1]
    return sorted_in[upper],sorted_in[lower]

def compute_ess(samples, max_lag=100):
    """
    Compute Effective Sample Size (ESS) from autocorrelation
    """
    n = len(samples)
    autocorr = acf(samples, nlags=max_lag, fft=False)
    # Exclude lag 0
    acf_lags = autocorr[1:]
    # Truncate at first non-positive ACF
    for i, r in enumerate(acf_lags):
        if r <= 0:
            acf_lags = acf_lags[:i]
            break
    tau = 1 + 2 * np.sum(acf_lags)
    ess = n / tau
    return ess

def compute_rhat(flat_samples, n_chains):
    """
    Compute R-hat from flattened posterior samples.
    """
    samples = flat_samples.reshape(int(n_chains), int(len(flat_samples)/n_chains))
    num_chains, num_samples = samples.shape
    #B = num_samples * np.var(chain_means, ddof=1)
    # Between-chain variance (B)
    chain_means = np.mean(samples, axis=1)
    grand_mean = np.mean(chain_means)
    B = (num_samples / (num_chains - 1)) * np.sum((chain_means - grand_mean) ** 2)
    # Within-chain variance (W)
    W = np.mean(np.var(samples, axis=1, ddof=1))
    var_hat = ((num_samples - 1) / num_samples) * W + (1 / num_samples) * B
    R_hat = np.sqrt(var_hat / W)
    return R_hat

def compute_rhat_az(samples,num_chains):
    samples_rs = samples.reshape(int(num_chains), int(len(samples)/num_chains))
    posterior_dict = {"para": samples_rs}
    idata = az.from_dict(posterior=posterior_dict)
    rhat_value = az.rhat(idata)["para"].item()
    return rhat_value

def addstats(dic_stats,samples,num_chains):
    if num_chains > 1: dic_stats["R_hat"] += [compute_rhat_az(samples,num_chains)]
    dic_stats["Mean"] += [samples.mean()]; dic_stats["Median"] += [np.median(samples)]
    dic_stats["Equal-tail 5th CI"] += [np.percentile(samples, 5)]; dic_stats["Equal-tail 95th CI"] += [np.percentile(samples, 95)]
    HPD_95th, HPD_5th = calculateHPD(samples, 90)
    dic_stats["HPD 5th CI"] += [HPD_5th]; dic_stats["HPD 95th CI"] += [HPD_95th]
    #dic_stats["ess"] += [compute_ess(samples)]
    dic_stats["ESS"] += [az.ess(np.expand_dims(samples, axis=0))] # (shape: [n_chains, n_samples])
    return dic_stats

def stringttophylo(string):
    handle = StringIO(string)
    tree = Phylo.read(handle, "newick")
    return tree

# Construct simple covariance matrix
def get_covariance_matrix(tree,taxa=None,sigma2=1):
    if taxa is None: species = [tip.name for tip in tree.get_terminals()]
    else: species = taxa
    n = len(species)
    covariance_matrix = np.zeros((n, n))
    for i, sp1 in enumerate(species):
        for j, sp2 in enumerate(species):
            # Compute shared path length between sp1 and sp2
            mrca = tree.common_ancestor(sp1, sp2)
            covariance_matrix[i, j] = tree.distance(mrca) * sigma2
    return covariance_matrix,species

def generaterandomtrait(tree):
    species = [tip.name for tip in tree.get_terminals()]
    traits_df = pd.DataFrame({'species': species, 'trait': np.random.normal(loc=0.0, scale=1.0, size=len(species))})
    traits = np.array(traits_df['trait'].values)
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

def marginalpdfancestralmean_sigma(cov_matrix, traits, mle_estimates, output = 'Marginal_PDF_RootValue_Sigma.pdf'):
    """
    This func is only for developing purpose
    """
    traits_ = [abs(i) for i in traits]
    limit = max(traits_)
    mle_trait,mle_sigma = mle_estimates
    trait_coordinates = np.linspace(-limit,limit,num=1000)
    sigma_coordinates = np.linspace(1e-8,1e10,num=1000)
    ll_trait_coordinates = np.array([-neg_log_likelihood_correct_pbmm((t,mle_sigma), traits, cov_matrix) for t in trait_coordinates])
    ll_sigma_coordinates = np.array([-neg_log_likelihood_correct_pbmm((mle_trait,np.log(s)), traits, cov_matrix) for s in sigma_coordinates])
    fig,axes = plt.subplots(1,2,figsize=(10,6))
    ax1,ax2 = axes.flatten()
    ax1.plot(trait_coordinates,ll_trait_coordinates,color='black',lw=3)
    ax1.vlines(mle_trait,0,1,transform=ax1.get_xaxis_transform(),lw=2,ls='--',color='black')
    y_min1 = ax1.get_ylim()[0]
    ax1.fill_between(trait_coordinates,np.full(len(trait_coordinates),y_min1),ll_trait_coordinates,color='gray',alpha=0.4)
    ax1.set_xlabel("Root trait",fontsize=15)
    ax1.set_ylabel("Log-likelihood",fontsize=15)
    ax1.set_title("LL of root trait",fontsize=15)
    ax2.plot(sigma_coordinates,ll_sigma_coordinates,color='black',lw=3)
    ax2.vlines(mle_sigma,0,1,transform=ax2.get_xaxis_transform(),lw=2,ls='--',color='black')
    y_min2 = ax2.get_ylim()[0]
    ax2.fill_between(sigma_coordinates,np.full(len(sigma_coordinates),y_min2),ll_sigma_coordinates,color='gray',alpha=0.4)
    ax2.set_xlabel("Sigma",fontsize=15)
    ax2.set_ylabel("Log-likelihood",fontsize=15)
    ax2.set_title("LL of sigma",fontsize=15)
    mle_sigma_coordinate = sigma_coordinates[np.argmax(ll_sigma_coordinates)]
    print("MLE sigma: {}".format(mle_sigma_coordinate))
    print(ll_sigma_coordinates.max(),ll_sigma_coordinates.min(),ll_sigma_coordinates.mean())
    fig.tight_layout()
    fig.savefig(output)
    plt.close()


def jointpdfancestralmean_sigma(cov_matrix, traits, output = 'Joint_PDF_RootValue_Sigma.pdf'):
    """
    This func is only for developing purpose
    """
    traits_ = [abs(i) for i in traits]
    limit = max(traits_)
    trait_coordinates = np.linspace(-limit,limit,num=1000)
    sigma_coordinates = np.linspace(np.log(1e-8),np.log(1e3),num=1000)
    neg_lls = np.zeros((1000,1000))
    for i,t in enumerate(trait_coordinates):
        for j,s in enumerate(sigma_coordinates):
            neg_lls[i,j] = neg_log_likelihood_correct_pbmm((t,s), traits, cov_matrix)
    P1, P2 = np.meshgrid(trait_coordinates,sigma_coordinates)
    fig,ax = plt.subplots(1,1,figsize=(6,6))
    plt.contourf(P1, P2, -neg_lls, levels=30, cmap='viridis')
    plt.colorbar(label='Log-Likelihood')
    ax.set_xlabel('Root trait value')
    ax.set_ylabel('Sigma')
    ax.set_title('Joint Log-Likelihood Distribution')
    fig.tight_layout()
    fig.savefig(output)
    plt.close()

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
    #Tree_dis = copy.deepcopy(tree)
    Tree_dis = tree
    tips = Tree_dis.get_terminals()
    #nodes = [i for i in tree.get_nonterminals() if i!= tree.root]
    for i,node in enumerate(Tree_dis.get_nonterminals()): node.name = "Node_{}".format(i)
    taxa = [i.name for i in tips]
    nodes = [i for i in Tree_dis.get_nonterminals()]
    nodes_noroot = [i for i in tree.get_nonterminals() if i!= tree.root]
    hypothetical_intermediate_ancestors = [i.name for i in nodes] # Including Root, which is the first node
    h_i_a_noroot = [i.name for i in nodes_noroot]
    total_taxa = taxa + hypothetical_intermediate_ancestors
    t_t_noroot = taxa+h_i_a_noroot
    return np.array(taxa),np.array(total_taxa),np.array(hypothetical_intermediate_ancestors),len(taxa),len(hypothetical_intermediate_ancestors),Tree_dis,np.array(t_t_noroot)

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
    # cov_clades have all tips+nodes (including root) as keys
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

def fromcovcaldes2totalcovmatrix(tree,cov_clades,total_taxa):
    n = len(total_taxa)
    cov_matrix = jnp.zeros((n, n))
    for i, taxon1 in enumerate(total_taxa):
        for j, taxon2 in enumerate(total_taxa):
            mrca = tree.common_ancestor(taxon1, taxon2)
            #cov_matrix[i, j] = cov_clades[mrca.name]
            cov_matrix = cov_matrix.at[i, j].set(cov_clades[mrca.name])
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
    Trait = {sp:tr for sp,tr in zip(df.index,df[traitcolname])}
    #logging.info("Trait data:")
    #logging.info(Trait)
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
    cho_tt = cho_factor(C_tt, lower=True)
    mle_means = ancestralmean + C_it @ cho_solve(cho_tt, residual) # the conditional MLE of internal nodes is a linear function of the observed tip values # the conditional distribution of a multivariate normal vector
    nodenames = total_taxa[internal_indices]
    return mle_means,nodenames

def compute_mle_internal_means_problematic(Tree,total_taxa,ntips,nnodes,traits,ancestralmean,total_cov_matrix):
    tip_indices = np.arange(ntips)
    internal_indices = np.arange(nnodes) + len(tip_indices)
    # Extract submatrices and subvectors
    C_tt = total_cov_matrix[np.ix_(tip_indices, tip_indices)]  # Tip-to-tip covariance
    C_it = total_cov_matrix[np.ix_(internal_indices, tip_indices)]  # Internal-to-tip covariance
    # Compute MLE for internal node means
    chol = cho_factor(C_tt)
    mle_means = C_it @ cho_solve(chol, traits)
    nodenames = total_taxa[internal_indices]
    return mle_means,nodenames

# Log-likelihood function under Brownian Motion
def neg_log_likelihood_correct_pbmm(params, y, C):
    """
    mu: the ancestral (root) value
    sigma2: the evolutionary rate
    """
    mu, log_sigma2 = params
    sigma2 = np.exp(log_sigma2)
    V = sigma2 * C
    try:
        # Use Cholesky decomposition for numerical stability
        cho = cho_factor(V) # c, low = cho_factor(A)
        diff = y - mu
        sol = cho_solve(cho, diff)
        log_det = 2.0 * np.sum(np.log(np.diag(cho[0])))
        n = len(diff)
        log_lik = -0.5 * (n * np.log(2 * np.pi) + log_det + diff.T @ sol)
        return -log_lik
    except np.linalg.LinAlgError:
        return np.inf

def log_likelihood_correct_givnC_pbmm(mu, y, C_tt):
    """
    mu: the ancestral (root) value
    C_tt: the covariance matrix of tips
    """
    try:
        # Use Cholesky decomposition for numerical stability
        cho = cho_factor(C_tt) # c, low = cho_factor(A)
        diff = y - mu
        sol = cho_solve(cho, diff)
        log_det = 2.0 * np.sum(np.log(np.diag(cho[0])))
        n = len(diff)
        log_lik = -0.5 * (n * np.log(2 * np.pi) + log_det + diff.T @ sol)
        return log_lik
    except np.linalg.LinAlgError:
        return np.inf

def pbmm_mle_eq(X, C):
    n = len(X)
    ones = np.ones(n)
    cho_C = cho_factor(C, lower=True)
    Cinv_X = cho_solve(cho_C, X)
    Cinv_1 = cho_solve(cho_C, ones)
    # MLE of mu
    numerator = ones @ Cinv_X
    denominator = ones @ Cinv_1
    mu_hat = numerator / denominator
    # Residual
    residual = X - mu_hat * ones
    Cinv_residual = cho_solve(cho_C, residual)
    # MLE of sigma^2
    sigma2_hat = (residual @ Cinv_residual) / n
    neg_ll = neg_log_likelihood_correct_pbmm((mu_hat,np.log(sigma2_hat)),X, C)
    # variance of mu_hat
    denom = ones.T @ Cinv_1
    var_mu_hat = sigma2_hat / denom
    return mu_hat, sigma2_hat, -neg_ll, var_mu_hat

def pdf_diff(d, mu, sigma):
    if d == 0:  # Special case (peak)
        integrand = lambda r: stats.lognorm.pdf(r, s=sigma, scale=np.exp(mu)) ** 2
        return quad(integrand, 0, np.inf)[0]
    elif d > 0:
        integrand = lambda r: stats.lognorm.pdf(d + r, s=sigma, scale=np.exp(mu)) * stats.lognorm.pdf(r, s=sigma, scale=np.exp(mu))
        return quad(integrand, 0, np.inf)[0]
    else:  # Use symmetry
        return pdf_diff(-d, mu, sigma)

def Monte_Carlo_LogNormal_Prior_PDF(d, mu, sigma, N=50000):
    dist_lognorm = stats.lognorm(s=sigma, scale=np.exp(mu))
    r1_samples = dist_lognorm.rvs(size=N)
    r2_samples = dist_lognorm.rvs(size=N)
    delta_samples = r1_samples - r2_samples
    kde = stats.gaussian_kde(delta_samples)
    prior_density_at_0 = kde.evaluate(d)[0]
    return prior_density_at_0

def Monte_Carlo_Gamma_Prior_PDF(d, alpha, beta, N=50000):
    scale = 1 / beta
    r1_samples = stats.gamma.rvs(a=alpha, scale=scale, size=N)
    r2_samples = stats.gamma.rvs(a=alpha, scale=scale, size=N)
    delta_samples = r1_samples - r2_samples
    kde_prior = stats.gaussian_kde(delta_samples)
    prior_density_at_0 = kde_prior.evaluate(d)[0]
    return prior_density_at_0

def pdf_diff_lognormal_zero(mu_log, sigma_log):
    """
    Calculate the PDF at zero of the difference of two i.i.d. LogNormal(mu_log, sigma_log) RVs.

    Parameters:
    - mu_log: mean of the underlying normal distribution (log scale)
    - sigma_log: std dev of the underlying normal distribution (log scale)

    Returns:
    - pdf value at zero of the difference r1 - r2
    """
    numerator = 4 * mu_log**2 - (sigma_log**4 - 4 * sigma_log**2 * mu_log + 4 * mu_log**2)
    exponent = -numerator / (4 * sigma_log**2)
    coef = 1 / (2 * np.sqrt(np.pi) * sigma_log)
    return coef * np.exp(exponent)

def getBFdenominator_onedirection(pair,samples_df):
    p1, p2 = pair
    posterior_prob = np.mean(samples_df[p1] > samples_df[p2])
    return posterior_prob

def savage_dickey_density_ratio(pb,sample=None,compared_parameters=[],lognormal=True,onedirection=False):
    if sample is not None: samples_df = pd.read_csv(sample,header=0,index_col=0,sep='\t')
    else: exit(0)
    all_pairs = list(itertools.combinations(compared_parameters, 2))
    if onedirection:
        posterior_densities_at_0 = np.array([getBFdenominator_onedirection(pair,samples_df) for pair in all_pairs])
    else:
        posterior_densities_at_0 = np.array([getBFdenominator(pair,samples_df) for pair in all_pairs])
    N = 50000
    if lognormal:
        params = lognormal_params_from_mean_var(mean=pb.constant_sigma2_mle, variance=pb.var_sigma2_hat)
        sigma_log,mu_log = params['scale'],params['loc']
        prior_density_at_0 = Monte_Carlo_LogNormal_Prior_PDF(0, mu_log, sigma_log, N=N)
        #prior_density_at_0 = pdf_diff_lognormal_zero(mu_log, sigma_log)
        #prior_density_at_0 = pdf_diff(0,mu_log, sigma_log)
    else:
        alpha = np.square(pb.constant_sigma2_mle)/pb.var_sigma2_hat
        beta = pb.constant_sigma2_mle/pb.var_sigma2_hat
        prior_density_at_0 = Monte_Carlo_Gamma_Prior_PDF(0, alpha, beta, N=N)
    if onedirection: prior_density_at_0 = 0.5
    #print("Prior: ",prior_density_at_0)
    #print("Posterior: ",posterior_densities_at_0)
    BF = prior_density_at_0 / posterior_densities_at_0
    BF_Pair = [(bf,pair) for bf,pair in zip(BF,all_pairs)]
    for i in BF_Pair: print(i)
    return BF_Pair

def writebf(BF_Pair,output="BF.tsv"):
    y = lambda x:"--".join(x)
    dic = {"Pair":[y(i[1]) for i in BF_Pair],"BF":[i[0] for i in BF_Pair]}
    df = pd.DataFrame(data=dic)
    df.to_csv(output,header=True,index=False,sep='\t')

def pglsnormalizating(trait):
    x_raw = trait
    x_mean = np.mean(x_raw)
    x_std = np.std(x_raw)
    x_normalized = (x_raw - x_mean) / x_std
    return x_normalized

def normalizating(trait):
    x_raw = trait
    x_mean = np.mean(x_raw)
    x_std = np.std(x_raw)
    x_normalized = (x_raw - x_mean) / x_std
    return x_std,x_mean,x_normalized

def getparasmregression(results):
    GLS_intercept,GLS_slope = results.params
    GLS_p_value = results.pvalues[1] # p-values for intercept and slope
    GLS_r_squared = results.rsquared
    GLS_log_likelihood = results.llf
    return GLS_intercept,GLS_slope,GLS_p_value,GLS_r_squared,GLS_log_likelihood

def GLSlikelihood(lambdaval,cov_matrix,x,y):
    diag = np.diag(np.diag(cov_matrix))
    new_cov_matrix = lambdaval * cov_matrix + (1 - lambdaval) * diag
    #new_cov_matrix = lambdaval * cov_matrix + (1 - lambdaval) * np.eye(len(cov_matrix))
    X = sm.add_constant(x)
    model = sm.GLS(y, X, sigma=new_cov_matrix)
    results = model.fit()
    #_,_,_,_,GLS_log_likelihood = getparasmregression(results)
    return -results.llf

def MLElambda_residuals(cov_matrix,x,y):
    result = minimize_scalar(
        GLSlikelihood,
        bounds=(0, 1), # Lambda is constrained between 0 and 1
        args=(cov_matrix, x, y),
        method='bounded')
    mle_lambda = result.x
    lambdavals = np.linspace(0,1,20)
    for debug_lam in lambdavals:
        debug_ll = GLSlikelihood(debug_lam,cov_matrix,x,y)
        print("lambda {} -- ll {}".format(debug_lam,-debug_ll))
    return mle_lambda

def givelamda(cov_matrix,lamda=1):
    diag = np.diag(np.diag(cov_matrix))
    test_cov_matrix = lamda * cov_matrix + (1 - lamda) * diag
    return test_cov_matrix

def GLSconfidenceband(results,x,y,X):
    y_pred = results.predict(X)
    X = np.column_stack((np.ones_like(x), x))
    cov_matrix = results.cov_params()
    std_err = np.sqrt(np.sum(np.dot(X, cov_matrix) * X, axis=1))
    alpha = 0.05
    df = len(x) - len(results.params)
    t_value = t.ppf(1 - alpha / 2, df)
    margin_error = t_value * std_err
    pred_low = y_pred - margin_error
    pred_high = y_pred + margin_error
    return pred_low,pred_high

def estimate_mu_hat(n,cov_matrix,trait):
    ones = np.ones(n)
    cho_C = cho_factor(cov_matrix, lower=True)
    Cinv_X = cho_solve(cho_C, trait)
    Cinv_1 = cho_solve(cho_C, ones)
    # MLE of mu
    numerator = ones @ Cinv_X
    denominator = ones @ Cinv_1
    mu_hat = numerator / denominator
    return mu_hat

class pglsregression:
    def __init__(self,tree=None,treeobject=None,data=None,compared_parameters=[],n_row=1,n_col=1,fs=(5,5),output="PGLS.pdf"):
        """
        : To plot all revelant stats of PGLS
        """
        if tree is None: self.Tree = treeobject
        else: self.Tree = Phylo.read(tree,format='newick')
        self.data = data;self.compared_parameters = compared_parameters;self.output = output
        self.n_row = n_row; self.n_col = n_col;self.figsize = fs
        fig, axes = plt.subplots(self.n_row, self.n_col, figsize = self.figsize)
        if self.n_row*self.n_col <= 1: axes = [axes]
        self.fig = fig;self.axes = axes
        self.constant_cov_matrix,_ = get_covariance_matrix(self.Tree)
        self.taxa,self.total_taxa,self.nodes,self.ntips,self.nnodes,self.Tree_nodenamed,self.t_t_noroot = definetaxa(self.Tree)

    def pgls(self,logtransform=False,normalization=False,logtransformx=False,logtransformy=False):
        """
        PGLS regression
        """
        all_pairs = list(itertools.combinations(self.compared_parameters, 2))
        df = pd.read_csv(self.data,header=0,index_col=0,sep='\t')
        for (p1,p2),ax in zip(all_pairs,self.axes):
            p1_array,p2_array = np.array(df[p1]),np.array(df[p2])
            if logtransform: p1_array,p2_array = np.log(p1_array),np.log(p2_array)
            elif logtransformx: p1_array = pglsnormalizating(p1_array)
            elif logtransformy: p2_array = pglsnormalizating(p2_array)
            if normalization: p1_array,p2_array = pglsnormalizating(p1_array),pglsnormalizating(p2_array)
            self.drawpgls(p1_array,p2_array,ax,p1,p2)

    def simulation2benchmark(self,lambdaval=1):
        """
        Simulating predictor and response variables under given lambda
        """
        # Simulate predictor X as BM
        x = multivariate_normal(mean=np.zeros(self.ntips), cov=self.constant_cov_matrix).rvs()
        # Simulate response Y with known beta and lambda
        true_lambda = lambdaval
        intercept = 0.2
        slope = 1.5
        sigma2 = 1.0
        self.true_cov_matrix = givelamda(self.constant_cov_matrix,lamda=true_lambda)
        residual = multivariate_normal(mean=np.zeros(self.ntips), cov=sigma2 * self.true_cov_matrix).rvs()
        y = intercept + slope * x + residual
        print("True Pagel's lambda: {}".format(true_lambda))
        x_y = [(i,j) for i,j in zip(x,y)]
        x_y_sorted = sorted(x_y, key=lambda x:x[0])
        x,y = np.array([i[0] for i in x_y_sorted]),np.array([i[1] for i in x_y_sorted])
        mle_lambda = MLElambda_residuals(self.constant_cov_matrix,x,y)
        print("Estimated Pagel's lambda: {}".format(mle_lambda))

    def drawpgls(self,x,y,ax,predict_trait_name,reponse_trait_name):
        x_y = [(i,j) for i,j in zip(x,y)]
        x_y_sorted = sorted(x_y, key=lambda x:x[0])
        x,y = np.array([i[0] for i in x_y_sorted]),np.array([i[1] for i in x_y_sorted])
        X = sm.add_constant(x)
        model = sm.GLS(y, X, sigma=self.constant_cov_matrix)
        results = model.fit()
        GLS_intercept,GLS_slope,GLS_p_value,GLS_r_squared,GLS_log_likelihood = getparasmregression(results)
        mle_lambda = MLElambda_residuals(self.constant_cov_matrix,x,y)
        mle_cov_matrix = givelamda(self.constant_cov_matrix,lamda=mle_lambda)
        model_mle = sm.GLS(y, X, sigma=mle_cov_matrix)
        results_mle = model_mle.fit()
        mle_GLS_intercept,mle_GLS_slope,mle_GLS_p_value,mle_GLS_r_squared,mle_GLS_log_likelihood = getparasmregression(results_mle)
        X_OLS = sm.add_constant(x)
        OLS_model = sm.OLS(y, X_OLS)
        results_OLS = OLS_model.fit()
        OLS_intercept,OLS_slope,OLS_p_value,OLS_r_squared,OLS_log_likelihood = getparasmregression(results_OLS)
        ax.plot(x, y, 'ok', alpha = 0.5, label='Trait value',markersize=5)
        ax.plot(x, OLS_intercept + OLS_slope*x, 'b', label='OLS regression')
        OLS_pred_low,OLS_pred_high = GLSconfidenceband(results_OLS,x,y,X_OLS)
        ax.fill_between(x, OLS_pred_low, OLS_pred_high, color='b', alpha=0.2, label='OLS 95% confidence band')
        ax.plot(x, GLS_intercept+x*GLS_slope, 'r', label='PGLS regression')
        GLS_pred_low,GLS_pred_high = GLSconfidenceband(results,x,y,X)
        ax.fill_between(x, GLS_pred_low, GLS_pred_high, color='r', alpha=0.2, label= 'PGLS 95% confidence band')
        ax.plot(x, mle_GLS_intercept+x*mle_GLS_slope, 'y', label=r"$\mathrm{{PGLS}}_{\lambda}$"+' regression')
        mle_GLS_pred_low, mle_GLS_pred_high = GLSconfidenceband(results_mle,x,y,X)
        ax.fill_between(x, mle_GLS_pred_low, mle_GLS_pred_high, color='y', alpha=0.2, label=r"$\mathrm{{PGLS}}_{\lambda}$"+' 95% confidence band')
        ax.set_ylabel('{}'.format(reponse_trait_name),fontsize = 15)
        ax.set_xlabel('{}'.format(predict_trait_name),fontsize = 15)
        ax.set_title("Linear regression",fontsize = 15)
        ax.plot([], [], label="OLS"+" R-squared: {:.4f}".format(OLS_r_squared),color='b')
        ax.plot([], [], label="OLS"+" P-value: {:.4f}".format(OLS_p_value),color='b')
        ax.plot([], [], label="PGLS R-squared: {:.4f}".format(GLS_r_squared),color='r')
        ax.plot([], [], label="PGLS P-value: {:.4f}".format(GLS_p_value),color='r')
        ax.plot([], [], label=r"MLE $\lambda$" + ": {:.4f}".format(mle_lambda),color='y')
        ax.plot([], [], label=r"$\mathrm{{PGLS}}_{\lambda}$"+" R-squared: {:.4f}".format(mle_GLS_r_squared),color='y')
        ax.plot([], [], label=r"$\mathrm{{PGLS}}_{\lambda}$"+" P-value: {:.4f}".format(mle_GLS_p_value),color='y')
        ax.legend(fontsize=15,bbox_to_anchor=(1, 1))
        self.fig.tight_layout()
        self.fig.savefig(self.output)

    def pglsll_givenlambda(self,x,y,lambda_val):
        diag = np.diag(np.diag(self.constant_cov_matrix))
        lambda_cov_matrix = lambda_val * self.constant_cov_matrix + (1 - lambda_val) * diag
        mean_vector = np.full(len(self.taxa), 0)
        log_like = multivariate_normal.logpdf(self.Trait, mean=mean_vector, cov=lambda_cov_matrix)
        #neg_ll = neg_log_likelihood_correct_pbmm((self.constant_mu_mle, np.log(self.constant_sigma2_mle)), self.Trait, lambda_cov_matrix)
        return -log_like

class PBMMBuilder:
    def __init__(self,tree=None,treeobject=None,trait=None,traitcolname=None,output=None,traitobject=None,traitname='Trait',normalization=False):
        if tree is None:
            self.Tree = treeobject
        else:
            self.Tree = Phylo.read(tree,format='newick')
        self.trait=trait;self.traitcolname=traitcolname;self.output=output;self.traitobject=traitobject;self.traitname=traitname
        self.definetotaltaxa()
        self.gettraitob()
        self.normalization = normalization

    def definetotaltaxa(self):
        self.taxa,self.total_taxa,self.nodes,self.ntips,self.nnodes,self.Tree_nodenamed,self.t_t_noroot = definetaxa(self.Tree)

    def infervariablepbmm(self):
        self.getvariableratecov()
        self.variable_mle_ancestralmean = self.getmleancestralmean(self.variable_cov_matrix,output='Variable_pdfancestralmean.pdf')
        self.variable_mle_node_means,self.variable_node_names = self.getmlenodesmean(self.variable_mle_ancestralmean,self.variable_total_cov_matrix)
        self.drawratetree()

    def inferconstantpbmm(self):
        self.getconstantratecovtaxa()
        self.getmlesigmacovll()
        self.constant_total_cov_matrix = gettotal_cov(self.Tree,self.total_taxa,sigma2=self.constant_sigma2_mle)
        self.constant_mle_node_means,self.constant_node_names = self.getmlenodesmean(self.constant_mle_ancestralmean,self.constant_total_cov_matrix)
        self.drawalltrait(output='Tree_MLE_Trait.pdf')

    def calculate_ini_parameters(self,logtransform=False):
        # Initial guess from constant model
        self.getconstantratecovtaxa()
        if logtransform: self.Trait = np.log(self.Trait)
        if self.normalization:
            self.Normalizator_std,self.Normalizator_mean,self.Normalized_Trait = normalizating(self.Trait)
            self.Trait = self.Normalized_Trait
        mu_init = np.mean(self.Trait) # ini guess for root value
        log_sigma2_init = np.log(np.var(self.Trait))
        mu_mle_eq, sigma2_mle_eq, ll_eq, var_mu_hat = pbmm_mle_eq(self.Trait, self.constant_cov_matrix)
        self.constant_mu_mle, self.constant_sigma2_mle, self.var_mu_hat = mu_mle_eq, sigma2_mle_eq, var_mu_hat
        self.var_sigma2_hat = 2 * np.square(var_mu_hat) * (self.ntips-1) / np.square(self.ntips)

    def pagel_lambda(self):
        """
        Calculate Pagel's λ
        """
        self.calculate_ini_parameters()
        result = minimize_scalar(self.ll_givenlambda,
                bounds=(0, 1),
                method='bounded')
        mle_lambda = result.x
        neg_ll = result.fun
        null_neg_ll = self.ll_givenlambda(0)
        LR_stat = 2 * (null_neg_ll - neg_ll)
        p_value = chi2.sf(LR_stat, df=1)
        print("Pagel's λ {} for {} (P-value: {:.5f})".format(mle_lambda,self.traitname,p_value))

    def pagel_lambda_nut(self,num_warmup=200,num_samples=200,num_chains=2):
        """
        Bayesian Inference of Pagel's λ
        """
        self.calculate_ini_parameters()
        rng_key = jrandom.PRNGKey(1)
        kernel = NUTS(self.pagel_ll_modeler)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
        mcmc.run(rng_key)
        # Retrieve samples
        mcmc.print_summary()
        posterior_samples = mcmc.get_samples()
        print(posterior_samples)

    def pagel_ll_modeler(self):
        """
        TODO: Deterministic mu given lamda_cov_matrix
        """
        mu = numpyro.sample("mu", dist.Normal(self.constant_mu_mle, np.sqrt(self.var_mu_hat)))
        self.ntips_and_nnodes = self.ntips + self.nnodes
        #lambdaval = numpyro.sample("lambda", dist.Uniform(low=0.0, high=1.0))
        lambdaval = numpyro.sample("lambda", dist.HalfNormal(scale=1.0))
        C_tt = givelamda(self.constant_cov_matrix,lamda=lambdaval)
        n = self.ntips
        ones = jnp.ones(n)
        cho_C = jcho_factor(C_tt, lower=True)
        Cinv_X = jcho_solve(cho_C, self.Trait)
        Cinv_1 = jcho_solve(cho_C, ones)
        # MLE of mu
        numerator = ones @ Cinv_X
        denominator = ones @ Cinv_1
        mu = numerator / denominator
        #numpyro.deterministic("mu", mu)
        # Observed traits likelihood
        numpyro.sample("obs", dist.MultivariateNormal(mu * jnp.ones(self.ntips), covariance_matrix=C_tt), obs=self.Trait)

    def ll_givenlambda(self,lambda_val):
        diag = np.diag(np.diag(self.constant_cov_matrix))
        lambda_cov_matrix = lambda_val * self.constant_cov_matrix + (1 - lambda_val) * diag
        #mean_vector = np.full(len(self.taxa), self.constant_mu_mle)
        n = self.ntips
        ones = np.ones(n)
        cho_C = cho_factor(lambda_cov_matrix, lower=True)
        Cinv_X = cho_solve(cho_C, self.Trait)
        Cinv_1 = cho_solve(cho_C, ones)
        # MLE of mu
        numerator = ones @ Cinv_X
        denominator = ones @ Cinv_1
        mu_hat = numerator / denominator
        mean_vector = np.full(n,mu_hat)
        log_like = multivariate_normal.logpdf(self.Trait, mean=mean_vector, cov=lambda_cov_matrix)
        #neg_ll = neg_log_likelihood_correct_pbmm((self.constant_mu_mle, np.log(self.constant_sigma2_mle)), self.Trait, lambda_cov_matrix)
        return -log_like

    def simulation2benchmark(self,lambdaval=1):
        """
        Simulate a single trait variable for benchmarking
        """
        self.getconstantratecovtaxa()
        self.true_cov_matrix = givelamda(self.constant_cov_matrix,lamda=lambdaval)
        x = multivariate_normal(mean=np.zeros(self.ntips), cov=self.true_cov_matrix).rvs()
        self.Trait = x
        #self.calculate_ini_parameters()
        self.getconstantratecovtaxa()
        result = minimize_scalar(self.ll_givenlambda,                                                                                                           bounds=(0, 1),
                method='bounded')
        mle_lambda = result.x
        print("True Pagel's λ {}".format(lambdaval))
        print("Estimated Pagel's λ {}".format(mle_lambda))
        print("True Mean {}".format(x.mean()))
        mu_hat = estimate_mu_hat(self.ntips,givelamda(self.constant_cov_matrix,lamda=mle_lambda),self.Trait)
        print("Estimated Mean {}".format(mu_hat))

    def ancestry_infer_variablepbmm(self,logtransform=False,lognormalrate=True,posteriorsamplesoutput=None,bayesstatsoutput=None,num_warmup=200,num_samples=200,num_chains=2):
        """
        TODO1: Find time windows with significant rise of evolutionary rates
        TODO2: Test Rate shift before and after specific time points or phylogenetic nodes
        TODO3: Regression of sigma2 on time
        TODO4: BF calculation across models with different hypothesis (Rate-Equality) (Run two models and process on Posterior samles)
        TODO5: Transplant DendroPy
        TODO6: Simulation-based benchmarking
        TODO7: Exponential prior on rate parameters (e.g. mean, shape, variance)
        """
        self.calculate_ini_parameters(logtransform=logtransform)
        # Make use of the estiamted mu and sigma2 to define the parameters in variable-rate model
        self.lognormalrate = lognormalrate
        self.variable_PSamples, self.variable_PMeans = self.getvariableratecov_NUT(num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains) # Posterior Mean {'sigma2':[..],'mu':..}
        nodes_traits, self.variable_nodes_traits_means, self.samples_ll, self.variable_node_names = self.calculnodetraitsfromsamples()
        self.variable_rates = self.variable_PMeans["sigma2"]
        self.variable_PMeans.update({"Internal_Trait":self.variable_nodes_traits_means})
        self.variable_PSamples.update({"Internal_Trait":nodes_traits})
        if posteriorsamplesoutput is not None and bayesstatsoutput is not None: self.export_Psamples(output=posteriorsamplesoutput,output_stats=bayesstatsoutput,num_chains=num_chains)


    def ancestry_infer_constantpbmm(self):
        self.getconstantratecovtaxa()
        mu_init = np.mean(self.Trait) # ini guess for root value
        log_sigma2_init = np.log(np.var(self.Trait))
        mu_mle_eq, sigma2_mle_eq, ll_eq, var_mu_hat = pbmm_mle_eq(self.Trait, self.constant_cov_matrix)
        self.constant_mu_mle, self.constant_sigma2_mle = mu_mle_eq, sigma2_mle_eq
        self.constant_ll = ll_eq
        logging.info("Constant-Rate PBMM Maximum Loglikelihood: {}".format(ll_eq))
        logging.info("Constant-Rate PBMM MLE Root {}, Sigma2 {}".format(mu_mle_eq, sigma2_mle_eq))
        logging.info("Raw Mean {}, Variance is {}".format(mu_init,np.exp(log_sigma2_init)))
        self.constant_total_cov_matrix = gettotal_cov(self.Tree,self.total_taxa,sigma2=self.constant_sigma2_mle)
        self.constant_mle_node_means,self.constant_node_names = self.getmlenodesmean(self.constant_mu_mle,self.constant_total_cov_matrix)

    def export_Psamples(self, output="Posterior_Samples.tsv", output_stats = "Posterior_Samples_Stats.tsv",num_chains=1):
        sigma2_samples = self.variable_PSamples['sigma2']
        node_trait_samples = self.variable_PSamples['Internal_Trait']
        dic,dic_stats = {},{"Mean":[],"Median":[],"Equal-tail 5th CI":[], "Equal-tail 95th CI":[],"HPD 5th CI":[], "HPD 95th CI":[],"ESS":[]}
        if num_chains > 1: dic_stats.update({"R_hat":[]})
        indexes = []
        for n_para in range(sigma2_samples.shape[1]):
            samples_sigma2 = sigma2_samples[:,n_para]
            dic[self.total_taxa[n_para]+"_sigma2"] = samples_sigma2
            dic_stats = addstats(dic_stats,samples_sigma2,num_chains); indexes+=[self.total_taxa[n_para]+"_sigma2"]
        for n_para in range(node_trait_samples.shape[1]):
            samples_trait = node_trait_samples[:,n_para]
            dic[self.nodes[n_para]+"_trait"] = samples_trait
            dic_stats = addstats(dic_stats,samples_trait,num_chains); indexes+=[self.nodes[n_para]+"_trait"]
        df = pd.DataFrame.from_dict(dic).rename_axis("Iteration")
        df["ll"] = self.samples_ll
        df.index = df.index + 1 # Shift the numbering to start from 1
        df.to_csv(output, header=True, index=True, sep='\t')
        df_stats = pd.DataFrame.from_dict(dic_stats)
        df_stats.index = indexes
        df_stats.to_csv(output_stats, header=True, index=True, sep='\t') 

    def getmlesigmacovll(self):
        self.constant_ll, self.constant_sigma2_mle,_ = log_likelihood_BM(self.constant_cov_matrix, self.Trait, ancestralmean=self.constant_mle_ancestralmean)
        self.constant_mle_cov_matrix = get_cov_matrix_givensiga(self.Tree, self.taxa, sigmasquare=self.constant_sigma2_mle)
        logging.info(f"Constant rate PBMM LL: {self.constant_ll}")
        logging.info(f"PBMM cov matrix: {self.constant_mle_cov_matrix}")

    def getconstantratecovtaxa(self):
        self.constant_cov_matrix,_ = get_covariance_matrix(self.Tree)

    def gettraitob(self):
        if self.trait is None:
            if self.traitobject is None:
                self.Trait = generaterandomtrait(self.Tree)
            else:
                self.Trait = np.array([traitobject[tip] for tip in self.taxa]) # Assumed a dict
        else: self.Trait = gettrait(self.trait,self.taxa,traitcolname=self.traitcolname)

    def exportnodetips(self,output="Tree_info.tsv"):
        dic = {"Node_ID":[],"Descendant":[]}
        for i in self.Tree.get_nonterminals():
            dic["Node_ID"] += [i.name]
            dic["Descendant"] += [", ".join([j.name for j in i.get_terminals()])]
        for tipname in self.taxa:
            dic["Node_ID"] += [tipname]
            dic["Descendant"] += [tipname]
        df = pd.DataFrame.from_dict(dic)
        df.to_csv(output, header=True, index=False, sep='\t')

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

    def getvariableratecov_modeler(self):
        """
        TODO: Deterministic internal mu and ll
        """
        mu = numpyro.sample("mu", dist.Normal(self.constant_mu_mle, np.sqrt(self.var_mu_hat)))
        self.ntips_and_nnodes = self.ntips + self.nnodes

        # One option is lognormal
        if self.lognormalrate:
            params = lognormal_params_from_mean_var(mean=self.constant_sigma2_mle, variance=self.var_sigma2_hat)
            log_sigma2 = numpyro.sample("log_sigma2", dist.Normal(loc=params['loc'],scale=params['scale']).expand([self.ntips_and_nnodes]))
            sigma2_ = jnp.exp(log_sigma2)
        #sigma2 = numpyro.sample("sigma2", dist.HalfNormal(scale=np.sqrt(self.var_sigma2_hat)).expand([self.ntips_and_nnodes]))
        #sigma2 += self.constant_sigma2_mle

        # One option is the Gamma distributed rates
        # guessed variance = self.var_sigma2_hat, guessed mean = self.constant_sigma2_mle
        else:
            alpha = np.square(self.constant_sigma2_mle)/self.var_sigma2_hat
            beta = self.constant_sigma2_mle/self.var_sigma2_hat
            sigma2_ = numpyro.sample("sigma2", dist.Gamma(concentration=alpha, rate=beta).expand([self.ntips_and_nnodes]))

        # Build the C
        ratesdic = {clade:rate for clade,rate in zip(self.total_taxa,sigma2_)}
        cov_clades = compute_total_cov_matrix(self.Tree,ratesdic)
        total_cov_matrix = fromcovcaldes2totalcovmatrix(self.Tree,cov_clades,self.total_taxa)
        tip_indices = np.arange(self.ntips)
        
        # Optional sampling of internal nodes
        #internal_indices = np.arange(self.nnodes) + len(tip_indices)
        #C_ii = total_cov_matrix[np.ix_(internal_indices[1:], internal_indices[1:])]
        #numpyro.sample("internal_traits", dist.MultivariateNormal(mu * jnp.ones(self.nnodes-1), covariance_matrix=C_ii))

        # Extract submatrices and subvectors
        C_tt = total_cov_matrix[np.ix_(tip_indices, tip_indices)]  # Tip-to-tip covariance
        # Observed traits likelihood
        numpyro.sample("obs", dist.MultivariateNormal(mu * jnp.ones(self.ntips), covariance_matrix=C_tt), obs=self.Trait)

        #C_it = total_cov_matrix[np.ix_(internal_indices, tip_indices)]  # Internal-to-tip covariance
        #C_ii = total_cov_matrix[np.ix_(internal_indices, internal_indices)] # Internal-to-Internal covariance
        #jitter = 1e-6
        #C_ii += jitter * jnp.eye(C_ii.shape[0])
        # Internal node traits prior 
        #numpyro.sample("internal_traits", dist.MultivariateNormal(mu * jnp.ones(self.nnodes), covariance_matrix=C_ii))
        #C_ii = total_cov_matrix[np.ix_(internal_indices[1:], internal_indices[1:])]
        #numpyro.sample("internal_traits", dist.MultivariateNormal(mu * jnp.ones(self.nnodes-1), covariance_matrix=C_ii))

    def getvariableratecov_NUT(self, num_warmup=200, num_samples=200, num_chains=2):
        rng_key = jrandom.PRNGKey(1)
        kernel = NUTS(self.getvariableratecov_modeler)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
        mcmc.run(rng_key)
        # Retrieve samples
        mcmc.print_summary()
        posterior_samples = mcmc.get_samples()
        if self.lognormalrate: posterior_samples["sigma2"] = np.exp(posterior_samples["log_sigma2"])
        posterior_means = {k: jnp.mean(v, axis=0).tolist() for k, v in posterior_samples.items()}
        return posterior_samples,posterior_means

    def calculnodetraitsfromsamples(self):
        mu_samples = self.variable_PSamples['mu']
        sigma2_samples = self.variable_PSamples['sigma2']
        nodes_traits = []
        samples_ll = []
        for iteration in range(sigma2_samples.shape[0]): # each row is one iteration, each column is a sigma2 parameter
            sigma2_ = sigma2_samples[iteration,:]
            ratesdic = {clade:rate for clade,rate in zip(self.total_taxa,sigma2_)}
            cov_clades = compute_total_cov_matrix(self.Tree,ratesdic)
            total_cov_matrix = fromcovcaldes2totalcovmatrix(self.Tree,cov_clades,self.total_taxa)
            mu_sample = mu_samples[iteration]
            mle_means,node_names = compute_mle_internal_means(self.Tree,self.total_taxa,self.ntips,self.nnodes,self.Trait,mu_sample,total_cov_matrix)
            nodes_traits.append(mle_means)
            tip_indices = np.arange(self.ntips)
            internal_indices = np.arange(self.nnodes) + len(tip_indices)
            C_tt = total_cov_matrix[np.ix_(tip_indices, tip_indices)]
            samples_ll.append(log_likelihood_correct_givnC_pbmm(mu_sample, self.Trait, C_tt))
        nodes_traits = np.array(nodes_traits)
        nodes_traits_means = np.mean(nodes_traits,axis=0).tolist()
        return nodes_traits, nodes_traits_means, samples_ll, node_names

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
        mle_node_means,node_names = compute_mle_internal_means(self.Tree,self.total_taxa,self.ntips,self.nnodes,self.Trait,mle_ancestralmean,total_cov_matrix)
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

    def drawalltrait_variable(self,traitname=None,output='Tree_MLE_Trait.pdf',nodetextdecimal=None,traitdecimal=1,**kargs):
        if traitname is None: traitname = self.traitname
        TB,_ = plottree(treeobject=self.Tree_nodenamed,**kargs)
        colors, norm, colormap = TB.transformcm(self.variable_rates)
        ubrobject = TB.getubrobject(colors,self.total_taxa)
        TB.ubrobject = ubrobject
        TB.basicdraw()
        tiptrait_dic = {ta:tr for ta,tr in zip(self.taxa,self.Trait)}
        nodetrait_dic = {ta:"{}".format(tr) for ta,tr in zip(self.variable_node_names,self.variable_nodes_traits_means)}
        TB.drawtrait(traitobject=[tiptrait_dic],traitobjectname=[traitname],xoffset=0.28,yoffset=0.2,labeloffset=0.2,traitcolor='gray',decimal=traitdecimal)
        TB.addtext2node(nodetrait_dic,textxoffset=0.01,textsize=10,textalpha=1,textstyle='normal',textcolor='k',decimal=nodetextdecimal)
        TB.addcolorbar(norm,colormap,TB.ax,TB.fig,fraction=0.05, pad=0.04,text="Evolutionary rate",fontsize=15)
        return TB

    def drawalltrait_constant(self,traitname=None,output='Tree_MLE_Trait.pdf',nodetextdecimal=None,traitdecimal=1,**kargs):
        if traitname is None: traitname = self.traitname
        TB,_ = plottree(treeobject=self.Tree_nodenamed,**kargs)
        TB.basicdraw()
        tiptrait_dic = {ta:tr for ta,tr in zip(self.taxa,self.Trait)}
        nodetrait_dic = {ta:"{}".format(tr) for ta,tr in zip(self.constant_node_names,self.constant_mle_node_means)}
        TB.drawtrait(traitobject=[tiptrait_dic],traitobjectname=[traitname],xoffset=0.28,yoffset=0.2,labeloffset=0.2,traitcolor='gray',decimal=traitdecimal)
        TB.addtext2node(nodetrait_dic,textxoffset=0.01,textsize=10,textalpha=1,textstyle='normal',textcolor='k',decimal=nodetextdecimal)
        return TB

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
    total_cov_matrix = fromcovcaldes2totalcovmatrix(tree,cov_clades,total_taxa)
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
    mle_node_means,node_names = compute_mle_internal_means(Tree,total_taxa,ntips,nnodes,trait,mle_ancestralmean,total_cov_matrix)
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
    mle_node_means,node_names,total_cov_matrix = compute_mle_internal_means(Tree,total_taxa,ntips,nnodes,trait,mle_ancestralmean)
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

