import numpy as np
from io import StringIO
from Bio import Phylo
import logging
from evotree.basicdraw import plottree

def assembletree(lineages,lineages_birth_time,lineages_death_time,lineages_duration_time):
    if len(lineages_birth_time) == 1:
        logging.info("No speciation event occurs at all!")
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

def simulate_pbdm(lambda_rate, mu_rate, initial_lineages, max_time):
    """
    Simulates a Phylogenetic Birth–Death Model (PBDM). 
    Parameters:
        lambda_rate (float): Speciation rate per lineage.
        mu_rate (float): Extinction rate per lineage.
        initial_lineages (int): Initial number of lineages.
        max_time (float): Maximum simulation time.
    """
    # Initialize state variables
    t = 0
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
            break
        # Select a lineage randomly for the event
        lineage = np.random.choice(list(lineages))
        # Determine event type
        event = np.random.choice(["speciation", "extinction"], p=[lambda_rate / (lambda_rate + mu_rate), mu_rate / (lambda_rate + mu_rate)])
        if event == "speciation":
            # Add two new lineages
            #new_id1 = "{}_{}".format(str(uuid.uuid4()),t)
            #new_id2 = "{}_{}".format(str(uuid.uuid4()),t)
            new_id1,new_id2 = "{}_1".format(lineage),"{}_2".format(lineage)
            lineages_birth_time[new_id1] = t
            lineages_birth_time[new_id2] = t
            lineages+= [new_id1,new_id2]
            # Remove the ancestral lineage
            lineages_death_time[lineage] = t
            lineages.remove(lineage)
        elif event == "extinction":
            # Remove the lineage
            lineages_death_time[lineage] = t
            lineages.remove(lineage)
    for key,value in lineages_birth_time.items():
        lineages_duration_time[key] = lineages_death_time.get(key,max_time) - value
    return lineages,lineages_birth_time,lineages_death_time,lineages_duration_time

class PBDMbuilder:
    def __init__(self,lambda_rate=0.5,mu_rate=0.3,initial_lineages=1,max_time=10):
        self.lambda_rate = lambda_rate;self.mu_rate = mu_rate
        self.initial_lineages = initial_lineages;self.max_time = max_time
    def basicsimutree(self):
        self.lineages,self.lineages_birth_time,self.lineages_death_time,self.lineages_duration_time = simulate_pbdm(self.lambda_rate,self.mu_rate,self.initial_lineages,self.max_time)
        self.extinct_lineages = [li for li,dt in self.lineages_death_time.items() if dt < self.max_time]
    def constructnewicktree(self):
        self.Tree,self.treetext = assembletree(self.lineages,self.lineages_birth_time,self.lineages_death_time,self.lineages_duration_time)
    def drawtree(self):
        if self.Tree is None:
            return
        TB,_ = plottree(treeobject=self.Tree,fs=(10,10))
        tipnames = [tip.name for tip in self.Tree.get_terminals()]
        TB.extinct_lineages = [i for i in self.extinct_lineages if i in tipnames]
        TB.basicdraw(log="Plotting simulated tree")
        TB.drawscale(plotfulllengthscale=True,fullscaletickheight=0.1,fullscaleticklabeloffset=0.1)
        TB.addextranodes(TB.extinct_lineages,marker='x',markersize=5,markercolor='k',markeralpha=1,labels=None,labelxoffset=0.1,labelyoffset=0.1,fontsize=15,labelalpha=1,fontstyle='normal',labelcolor='k')
        TB.saveplot('Simulated_Tree.pdf')

def pbdmmodeling(lambda_rate=0.5,mu_rate=0.3,initial_lineages=1,max_time=10):
    PBDM = PBDMbuilder()
    PBDM.basicsimutree()
    PBDM.constructnewicktree()
    PBDM.drawtree()

