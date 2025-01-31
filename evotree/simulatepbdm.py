import numpy as np
from io import StringIO
from Bio import Phylo
import logging
from evotree.basicdraw import plottree
import copy
from tqdm import trange
from scipy.optimize import minimize
from scipy.special import expm1
from evotree.ploter import agedistributiondrawer
import matplotlib.pyplot as plt

def assembletree(lineages,lineages_birth_time,lineages_death_time,lineages_duration_time):
    if len(lineages_birth_time) == 1:
        return None,None
    distance_to_root = {li:(len(li.split('_'))-1) for li in lineages_birth_time.keys()}
    nodes_category = categorynodes(list(lineages_birth_time.keys()))
    nodes_string = gettipnodestring(nodes_category,lineages_duration_time)
    tip_lineage,tip_distance = sorted(distance_to_root.items(),key=lambda x:x[1])[-1]
    nodes_string = processdtr(nodes_category,nodes_string,tip_lineage,distance_to_root,lineages_duration_time,self=False)
    root_lineage = sorted(distance_to_root.items(),key=lambda x:x[1])[0][0]
    nodes_string[root_lineage] = "({}:{},{}:{}):{}".format(nodes_string["S1_1"],lineages_duration_time["S1_1"],nodes_string["S1_2"],lineages_duration_time["S1_2"],lineages_duration_time[root_lineage])
    Tree = stringttophylo(nodes_string[root_lineage])
    return Tree,nodes_string[root_lineage]

def stringttophylo(string):
    handle = StringIO(string)
    tree = Phylo.read(handle, "newick")
    return tree

def gettipnodestring(nodes_category,lineages_duration_time):
    nodes_string = {}
    for li,cate in nodes_category.items():
        if cate == "Tip_Node":
            nodes_string[li] = li
    return nodes_string

def categorynodes(lineages):
    nodes_category = {}
    for li in lineages:
        if not li+"_1" in lineages:
            nodes_category[li] = "Tip_Node"
    for li in lineages:
        if li not in nodes_category:
            nodes_category[li] = "Internal_Node"
    return nodes_category

def processdtr(nodes_category,nodes_string,lineage,distance_to_root,lineages_duration_time,self=False):
    if distance_to_root[lineage] == 0:
        self = True
    if self:
        lineage = lineage+"_1"
        sister_lineage = lineage+"_2"
    branch_length = lineages_duration_time[lineage]
    sister_lineage = getsisterclade(lineage)
    sister_lineage_branch_length = lineages_duration_time[sister_lineage]
    parent_lineage = lineage[:-2]
    if lineage in nodes_string and sister_lineage in nodes_string:
        parent_string = "({}:{},{}:{})".format(nodes_string[lineage],branch_length,nodes_string[sister_lineage],sister_lineage_branch_length)
        if parent_lineage not in nodes_string:
            nodes_string[parent_lineage] = parent_string
        #if distance_to_root[parent_lineage] != 0:
        if len(nodes_string) < len(distance_to_root):
            if self:
                return processdtr(nodes_category,nodes_string,parent_lineage[:-2],distance_to_root,lineages_duration_time,self=False)
            else:
                return processdtr(nodes_category,nodes_string,parent_lineage,distance_to_root,lineages_duration_time,self=False)
        else:
            return nodes_string
    elif lineage in nodes_string and not sister_lineage in nodes_string:
        return processdtr(nodes_category,nodes_string,sister_lineage,distance_to_root,lineages_duration_time,self=True)
    elif not lineage in nodes_string and sister_lineage in nodes_string:
        return processdtr(nodes_category,nodes_string,lineage,distance_to_root,lineages_duration_time,self=True)
    else:
        return processdtr(nodes_category,nodes_string,lineage,distance_to_root,lineages_duration_time,self=True)

def getsisterclade(clade):
    if clade.endswith("_1"): return clade[:-2] + "_2"
    if clade.endswith("_2"): return clade[:-2] + "_1"

# The probability that a lineage survives up to time T, S(t) is given by:
def alineagesurviprob(lambda_rate, mu_rate, T, rho = 1):
    S_t = np.exp(-T*(lambda_rate+mu_rate)) * rho
    return S_t

# Under the Birth-Death Model, the probability that a single lineage survives to the present (time T) is given by:
def probsurvival(lambda_rate, mu_rate, T):
    p = 1 - np.exp(-T*(lambda_rate-mu_rate))
    return np.log(p)

def pbd_log_likelihood(params, branching_durations, T, rho):
    """
    Calculate the log-likelihood of a phylogenetic tree under the PBDM conditioning on the tree is not extinct.
    Parameters:
        params: [lambda, mu]
            lambda: Speciation rate
            mu: Extinction rate
            T: Tree height (time from root to present)
            ## rho: Sampling probability, fixed as 1
        branching_durations: List of tuples (branching time, duration)
    Returns:
        log_likelihood: Log-likelihood value
    """
    lambda_rate, mu_rate = params
    n_tips = len(branching_durations) + 1  # Number of tips in the tree
    total_rate = lambda_rate + mu_rate
    n_lineages = 1
    log_likelihood = 0
    for t,dt in sorted(branching_durations, key=lambda x:x[0]):
        log_likelihood += np.log(lambda_rate * n_lineages)  # Speciation event
        log_likelihood += -total_rate * n_lineages * dt  # Survival contribution
        n_lineages += 1
    log_likelihood += n_lineages * np.log(rho) # Sampling contribution (probability that observed lineages survive to the present)
    return -log_likelihood

def mle_constant_rates(branching_durations, T, survival_ratio):
    initial_guess = np.array([0.3,0.1])
    bounds = [(1e-6, None), (1e-6, None)]
    result = minimize(pbd_log_likelihood,
            initial_guess,
            args=(branching_durations, T, survival_ratio),
            method='L-BFGS-B',
            bounds=bounds)
    mle_lambda_rate,mle_mu_rate = result.x
    return mle_lambda_rate,mle_mu_rate

def simulate_pbdm_withwgd(lambda_rate, mu_rate, initial_lineages, max_time, wgd_time, retention_rate=0.1, silence=False):
    """
    Simulates a Phylogenetic Birth–Death Model (PBDM).
    Parameters:
        lambda_rate (float): Speciation rate per lineage.
        mu_rate (float): Extinction rate per lineage.
        initial_lineages (int): Initial number of lineages.
        max_time (float): Maximum simulation time.
    """
    # Initialize state variables
    t = 0;log_likelihood = 0.0
    lineages = ["S{}".format(i+1) for i in range(initial_lineages)]
    lineages_birth_time = {i:t for i in lineages}
    lineages_death_time = {}
    lineages_duration_time = {}
    wgdflag = False
    compensateflag = False
    wgd_lineages = []
    wgdlineages_birth_time = {};wgdlineages_death_time = {};wgdlineages_duration_time = {}
    while t <= max_time and len(lineages) > 0:
        # Total event rate
        total_rate = len(lineages) * (lambda_rate + mu_rate)
        if total_rate == 0:
            break
        # Time to next event
        dt = np.random.exponential(1 / total_rate)
        t += dt
        if t >= wgd_time and wgdflag == False:
            wgdflag = True
            #postwgd_lineages = []
            for li in lineages:
                diploidization_event = np.random.choice(["retention", "loss"], p=[retention_rate/1, (1-retention_rate)/1])
                new_id1,new_id2 = "{}_wgd_1".format(li),"{}_wgd_2".format(li)
                #lineages_birth_time[new_id1] = wgd_time
                lineages_birth_time[new_id2] = wgd_time
                #lineages_death_time[li] = wgd_time
                #wgdlineages_birth_time[new_id1] = wgd_time
                #wgdlineages_birth_time[new_id2] = wgd_time
                #wgdlineages_death_time[li] = wgd_time
                if diploidization_event == "retention":
                    #wgd_lineages += [new_id1,new_id2]
                    wgd_lineages += [new_id2]
                    lineages += [new_id2]
                    wgdlineages_birth_time[new_id2] = wgd_time
                else:
                    #wgd_lineages += [new_id1]
                    #postwgd_lineages += [new_id1]
                    #wgdlineages_death_time[new_id2] = wgd_time
                    lineages_death_time[new_id2] = wgd_time
        if t >= wgd_time and compensateflag == False:
            wgd_evo_t = copy.deepcopy(wgd_time)
            while wgd_evo_t <= t and len(wgd_lineages) > 0:
                # Total event rate
                total_rate = len(wgd_lineages) * (lambda_rate + mu_rate)
                if total_rate == 0:
                    break
                # Time to next event
                dt = np.random.exponential(1 / total_rate)
                wgd_evo_t += dt
                if wgd_evo_t > t:
                    break
                lineage = np.random.choice(wgd_lineages)
                # Determine event type
                event = np.random.choice(["speciation", "extinction"], p=[lambda_rate / (lambda_rate + mu_rate), mu_rate / (lambda_rate + mu_rate)])
                if event == "speciation":
                    log_likelihood += np.log(lambda_rate*len(wgd_lineages)) # Contribution of speciation to the log-likelihood
                    # Add two new lineages
                    new_id1,new_id2 = "{}_1".format(lineage),"{}_2".format(lineage)
                    wgdlineages_birth_time[new_id1] = wgd_evo_t
                    lineages_birth_time[new_id1] = wgd_evo_t
                    wgdlineages_birth_time[new_id2] = wgd_evo_t
                    lineages_birth_time[new_id2] = wgd_evo_t
                    wgd_lineages+= [new_id1,new_id2]
                    lineages+= [new_id1,new_id2]
                    # Remove the ancestral lineage
                    lineages_death_time[lineage] = wgd_evo_t
                    wgdlineages_death_time[lineage] = wgd_evo_t
                    wgd_lineages.remove(lineage)
                    if lineage not in lineages: lineages.remove(lineage)
                elif event == "extinction":
                    log_likelihood += np.log(mu_rate*len(wgd_lineages)) # Contribution of extinction to the log-likelihood
                    # Remove the lineage
                    lineages_death_time[lineage] = wgd_evo_t
                    wgdlineages_death_time[lineage] = wgd_evo_t
                    wgd_lineages.remove(lineage)
                    if lineage not in lineages: lineages.remove(lineage)
                log_likelihood += -total_rate * dt # Contribution of survival to the log-likelihood
        if t > max_time:
            t -= dt
            dt = max_time - t
            log_likelihood += - total_rate*dt
            for li in lineages: lineages_death_time[li] = max_time
            break
        # Select a lineage randomly for the event
        if t >= wgd_time and compensateflag == False:
            priorwgd_lineages = [i for i in lineages if i not in wgd_lineages]
            lineage = np.random.choice(priorwgd_lineages)
            compensateflag = True
        else:
            lineage = np.random.choice(lineages)
        # Determine event type
        event = np.random.choice(["speciation", "extinction"], p=[lambda_rate / (lambda_rate + mu_rate), mu_rate / (lambda_rate + mu_rate)])
        if event == "speciation":
            log_likelihood += np.log(lambda_rate*len(lineages)) # Contribution of speciation to the log-likelihood
            # Add two new lineages
            new_id1,new_id2 = "{}_1".format(lineage),"{}_2".format(lineage)
            lineages_birth_time[new_id1] = t
            lineages_birth_time[new_id2] = t
            lineages+= [new_id1,new_id2]
            # Remove the ancestral lineage
            lineages_death_time[lineage] = t
            lineages.remove(lineage)
        elif event == "extinction":
            log_likelihood += np.log(mu_rate*len(lineages)) # Contribution of extinction to the log-likelihood
            # Remove the lineage
            lineages_death_time[lineage] = t
            lineages.remove(lineage)
        log_likelihood += -total_rate * dt # Contribution of survival to the log-likelihood

    for key,value in lineages_birth_time.items():
        #lineages_duration_time[key] = lineages_death_time.get(key,max_time) - value
        lineages_duration_time[key] = lineages_death_time[key] - value
    N_speciation = (len(lineages_birth_time)-1)/2
    if not silence:
        if len(lineages) == 0:
            logging.info("All lineages go extinct at {}".format(t))
            logging.info("In total {} speciation events occurred".format(int(round(N_speciation))))
        elif len(lineages_birth_time) == 1:
            logging.info("No speciation or extinction occurring in the ancestry lineage")
        else:
            logging.info("{} lineages survived".format(len(lineages)))
            logging.info("In total {} speciation events occurred".format(int(round(N_speciation))))
    
    wgdt = wgd_time
    while wgdt <= max_time and len(wgd_lineages) > 0:
        # Total event rate
        total_rate = len(wgd_lineages) * (lambda_rate + mu_rate)
        if total_rate == 0:
            break
        # Time to next event
        dt = np.random.exponential(1 / total_rate)
        wgdt += dt
        if wgdt > max_time:
            wgdt -= dt
            dt = max_time - wgdt
            log_likelihood += - total_rate*dt
            for li in wgd_lineages: wgdlineages_death_time[li] = max_time
            break
        # Select a lineage randomly for the event
        lineage = np.random.choice(list(wgd_lineages))
        # Determine event type
        event = np.random.choice(["speciation", "extinction"], p=[lambda_rate / (lambda_rate + mu_rate), mu_rate / (lambda_rate + mu_rate)])
        if event == "speciation":
            log_likelihood += np.log(lambda_rate*len(wgd_lineages)) # Contribution of speciation to the log-likelihood
            # Add two new lineages
            new_id1,new_id2 = "{}_1".format(lineage),"{}_2".format(lineage)
            wgdlineages_birth_time[new_id1] = t
            wgdlineages_birth_time[new_id2] = t
            wgd_lineages+= [new_id1,new_id2]
            # Remove the ancestral lineage
            wgdlineages_death_time[lineage] = t
            wgd_lineages.remove(lineage)
        elif event == "extinction":
            log_likelihood += np.log(mu_rate*len(wgd_lineages)) # Contribution of extinction to the log-likelihood
            # Remove the lineage
            wgdlineages_death_time[lineage] = t
            wgd_lineages.remove(lineage)
        log_likelihood += -total_rate * dt # Contribution of survival to the log-likelihood
    for key,value in wgdlineages_birth_time.items():
        wgdlineages_duration_time[key] = wgdlineages_death_time[key] - value
    return lineages,lineages_birth_time,lineages_death_time,lineages_duration_time,log_likelihood,wgd_lineages,wgdlineages_birth_time,wgdlineages_death_time,wgdlineages_duration_time

def simulate_pbdm(lambda_rate, mu_rate, initial_lineages, max_time, silence=False):
    """
    Simulates a Phylogenetic Birth–Death Model (PBDM). 
    Parameters:
        lambda_rate (float): Speciation rate per lineage.
        mu_rate (float): Extinction rate per lineage.
        initial_lineages (int): Initial number of lineages.
        max_time (float): Maximum simulation time.
    """
    # Initialize state variables
    t = 0;log_likelihood = 0.0
    lineages = ["S{}".format(i+1) for i in range(initial_lineages)]
    lineages_birth_time = {i:t for i in lineages}
    lineages_death_time = {}
    lineages_duration_time = {}
    while t <= max_time and len(lineages) > 0:
        # Total event rate
        total_rate = len(lineages) * (lambda_rate + mu_rate)
        if total_rate == 0:
            break
        # Time to next event
        dt = np.random.exponential(1 / total_rate)
        t += dt
        if t > max_time:
            t -= dt
            dt = max_time - t
            log_likelihood += - total_rate*dt
            for li in lineages: lineages_death_time[li] = max_time
            break
        # Select a lineage randomly for the event
        lineage = np.random.choice(list(lineages))
        # Determine event type
        event = np.random.choice(["speciation", "extinction"], p=[lambda_rate / (lambda_rate + mu_rate), mu_rate / (lambda_rate + mu_rate)])
        if event == "speciation":
            log_likelihood += np.log(lambda_rate*len(lineages)) # Contribution of speciation to the log-likelihood
            # Add two new lineages
            new_id1,new_id2 = "{}_1".format(lineage),"{}_2".format(lineage)
            lineages_birth_time[new_id1] = t
            lineages_birth_time[new_id2] = t
            lineages+= [new_id1,new_id2]
            # Remove the ancestral lineage
            lineages_death_time[lineage] = t
            lineages.remove(lineage)
        elif event == "extinction":
            log_likelihood += np.log(mu_rate*len(lineages)) # Contribution of extinction to the log-likelihood
            # Remove the lineage
            lineages_death_time[lineage] = t
            lineages.remove(lineage)
        log_likelihood += -total_rate * dt # Contribution of survival to the log-likelihood
    for key,value in lineages_birth_time.items():
        #lineages_duration_time[key] = lineages_death_time.get(key,max_time) - value
        lineages_duration_time[key] = lineages_death_time[key] - value
    N_speciation = (len(lineages_birth_time)-1)/2
    if not silence:
        if len(lineages) == 0:
            logging.info("All lineages go extinct at {}".format(t))
            logging.info("In total {} speciation events occurred".format(int(round(N_speciation))))
        elif len(lineages_birth_time) == 1:
            logging.info("No speciation or extinction occurring in the ancestry lineage")
        else:
            logging.info("{} lineages survived".format(len(lineages)))
            logging.info("In total {} speciation events occurred".format(int(round(N_speciation))))
    return lineages,lineages_birth_time,lineages_death_time,lineages_duration_time,log_likelihood

def plotagedistribution(lineages_duration_times,lineagess,plotkde=False,fitexpon=False,legends={},outfile='',ax=None,letter=None,wgdage=None):
    Ages = []
    for iter_id,li in lineagess.items(): # Retrieve ages of only extant members
        # li is the list of extant lineages
        age_dict_all = lineages_duration_times[iter_id]
        ages = [age_dict_all[extant] for extant in li]
        Ages += ages
    Ages = np.array(Ages)
    agedistributiondrawer(Ages,plotkde=plotkde,fitexpon=fitexpon,outfile=outfile,legends=legends,ax=ax,letter=letter,wgdage=wgdage)

class PBDMbuilder:
    def __init__(self,lambda_rate=0.5,mu_rate=0.5,initial_lineages=1,max_time=10):
        self.lambda_rate = lambda_rate;self.mu_rate = mu_rate
        self.initial_lineages = initial_lineages;self.max_time = max_time
    def basicsimutree(self):
        self.lineages,self.lineages_birth_time,self.lineages_death_time,self.lineages_duration_time,self.loglikelihood = simulate_pbdm(self.lambda_rate,self.mu_rate,self.initial_lineages,self.max_time)
        self.extinct_lineages = [li for li,dt in self.lineages_death_time.items() if dt < self.max_time]
    def buildagedistributionwithwgd(self,n_iteration=10,outfile='',ax=None,letter=None,wgd_time=1,retention_rate=0.5):
        outfile = "Gene_Age_Distribution_WGDtime_{}_dup{}_los{}_iter{}.pdf".format(wgd_time,self.lambda_rate,self.mu_rate,n_iteration)
        lineagess,lineages_birth_times,lineages_death_times,lineages_duration_times,loglikelihoods = [{} for i in range(5)]
        wgd_lineagess,wgdlineages_birth_times,wgdlineages_death_times,wgdlineages_duration_times = [{} for i in range(4)]
        for i in trange(n_iteration):
            lineages,lineages_birth_time,lineages_death_time,lineages_duration_time,loglikelihood,wgd_lineages,wgdlineages_birth_time,wgdlineages_death_time,wgdlineages_duration_time = simulate_pbdm_withwgd(self.lambda_rate,self.mu_rate,self.initial_lineages,self.max_time,wgd_time, retention_rate=retention_rate,silence=True)
            lineagess[i] = lineages;lineages_birth_times[i] = lineages_birth_time;lineages_death_times[i] = lineages_death_time;lineages_duration_times[i] = lineages_duration_time;loglikelihoods[i] = loglikelihood
            wgd_lineagess[i] = wgd_lineages;wgdlineages_birth_times[i]=wgdlineages_birth_time;wgdlineages_death_times[i]=wgdlineages_death_time;wgdlineages_duration_times[i]=wgdlineages_duration_time
        N_genes = []
        for li in lineagess.values(): N_genes+=li
        logging.info("{} iterations with {} survivors".format(n_iteration,len(N_genes)))
        legends = {"Duplication rate":self.lambda_rate,"Loss rate":self.mu_rate,"Gene family":n_iteration,"Retention rate":retention_rate}
        plotagedistribution(lineages_duration_times,lineagess,plotkde=False,fitexpon=False,legends=legends,outfile=outfile,ax=ax,letter=letter)
        #plotagedistribution(wgdlineages_duration_times,wgd_lineagess,plotkde=False,fitexpon=True,legends=legends,outfile=outfile,ax=ax,letter=letter)
    def buildagedistribution(self,n_iteration=10,outfile='',ax=None,letter=None):
        outfile = "Gene_Age_Distribution_dup{}_los{}_iter{}.pdf".format(self.lambda_rate,self.mu_rate,n_iteration)
        lineagess,lineages_birth_times,lineages_death_times,lineages_duration_times,loglikelihoods = [{} for i in range(5)]
        for i in trange(n_iteration):
            lineages,lineages_birth_time,lineages_death_time,lineages_duration_time,loglikelihood = simulate_pbdm(self.lambda_rate,self.mu_rate,self.initial_lineages,self.max_time,silence=True)
            lineagess[i] = lineages;lineages_birth_times[i] = lineages_birth_time;lineages_death_times[i] = lineages_death_time;lineages_duration_times[i] = lineages_duration_time;loglikelihoods[i] = loglikelihood
        N_genes = []
        for li in lineagess.values(): N_genes+=li
        logging.info("{} iterations with {} survivors".format(n_iteration,len(N_genes)))
        legends = {"Duplication rate":self.lambda_rate,"Loss rate":self.mu_rate,"Gene family":n_iteration}
        plotagedistribution(lineages_duration_times,lineagess,plotkde=False,fitexpon=True,legends=legends,outfile=outfile,ax=ax,letter=letter)
    def constructnewicktree(self):
        self.Tree,self.treetext = assembletree(self.lineages,self.lineages_birth_time,self.lineages_death_time,self.lineages_duration_time)
    def getextantree(self):
        if self.Tree is None:
            self.ExtantTree = None
            return
        if len(self.extinct_tiplineages) >= len(self.Tree.get_terminals())-1:
            self.ExtantTree = None
            return
        self.ExtantTree = copy.deepcopy(self.Tree)
        for extinct_lineage in self.extinct_tiplineages:
            if self.ExtantTree.get_path(extinct_lineage, terminal=True):
                self.ExtantTree.prune(extinct_lineage)
    def drawextantree(self):
        if self.ExtantTree is None: return
        TB,_ = plottree(treeobject=self.ExtantTree,fs=(10,10))
        TB.basicdraw(log="Plotting simulated extant tree")
        TB.drawscale(plotfulllengthscale=True,fullscaletickheight=0.1,fullscaleticklabeloffset=0.1)
        TB.saveplot('Simulated_Extant_Tree.pdf')
        self.TB = TB
    def drawtree(self):
        if self.Tree is None: return
        TB,_ = plottree(treeobject=self.Tree,fs=(10,10))
        tipnames = [tip.name for tip in self.Tree.get_terminals()]
        TB.extinct_lineages = [i for i in self.extinct_lineages if i in tipnames]
        self.extinct_tiplineages = TB.extinct_lineages
        self.survival_ratio = 1 - len(self.extinct_tiplineages)/len(self.Tree.get_terminals())
        TB.basicdraw(log="Plotting simulated full tree")
        logging.info("Survival ratio is {}".format(self.survival_ratio))
        TB.drawscale(plotfulllengthscale=True,fullscaletickheight=0.1,fullscaleticklabeloffset=0.1)
        TB.addextranodes(TB.extinct_lineages,marker='x',markersize=5,markercolor='k',markeralpha=1,labels=None,labelxoffset=0.1,labelyoffset=0.1,fontsize=15,labelalpha=1,fontstyle='normal',labelcolor='k')
        TB.saveplot('Simulated_Full_Tree.pdf')
    def getbranching_durations(self):
        if self.ExtantTree is None: return
        self.root = self.ExtantTree.root
        self.nodes = [node for node in self.ExtantTree.get_nonterminals()]
        self.branching_durations = [(self.root.distance(node),node.branch_length) for node in self.nodes if node != self.root]
    def infermleconstantrates(self):
        if self.ExtantTree is None: return
        self.getbranching_durations()
        self.mle_constant_lambda_rate,self.mle_constant_mu_rate = mle_constant_rates(self.branching_durations, self.TB.Total_length, self.survival_ratio)
        logging.info("MLE of lambda is {} under constant-rate model".format(self.mle_constant_lambda_rate))
        logging.info("MLE of mu is {} under constant-rate model".format(self.mle_constant_mu_rate))

def pbdmmodeling(lambda_rate=2,mu_rate=2,initial_lineages=1,max_time=5):
    PBDM = PBDMbuilder()
    PBDM.basicsimutree()
    PBDM.constructnewicktree()
    PBDM.drawtree()
    PBDM.getextantree()
    PBDM.drawextantree()
    PBDM.infermleconstantrates()

def pbdmmodelingage():
    PBDM = PBDMbuilder(lambda_rate=1,mu_rate=1,initial_lineages=1,max_time=5)
    dup_los = [(1.5,1),(1,1),(1,1.5)] # 3 ratios
    letters = ['a','b','c']
    fig, axs = plt.subplots(1,3,figsize=(18, 6))
    outfile = "Gene_Age_Distribution_3_ratios.pdf"
    for dup_los,ax,letter in zip(dup_los,axs.flatten(),letters):
        dup,los = dup_los
        PBDM = PBDMbuilder(lambda_rate=dup,mu_rate=los,initial_lineages=1,max_time=5)
        PBDM.buildagedistribution(n_iteration=10000,ax=ax,letter=letter)
        fig.tight_layout()
        fig.savefig(outfile)
        plt.close()

def pbdmmodelingagewithwgd():
    PBDM = PBDMbuilder(lambda_rate=1,mu_rate=1,initial_lineages=1,max_time=5)
    dup_los = [(1,1),(1,1),(1,1)] # same ratio
    qs = [0.8,0.4,0.2]
    letters = ['a','b','c']
    fig, axs = plt.subplots(1,3,figsize=(18, 6))
    outfile = "Gene_Age_Distribution_3_retentionrates_withWGD.pdf"
    for dup_los,ax,letter,q in zip(dup_los,axs.flatten(),letters,qs):
        dup,los = dup_los
        PBDM = PBDMbuilder(lambda_rate=dup,mu_rate=los,initial_lineages=1,max_time=5)
        PBDM.buildagedistributionwithwgd(n_iteration=10000,ax=ax,letter=letter,wgd_time=4,retention_rate=q)
        fig.tight_layout()
        fig.savefig(outfile)
        plt.close()
