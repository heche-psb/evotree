import logging
from Bio import Phylo
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.patches import Rectangle
from matplotlib import colors
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_rgba
import copy
import numpy as np
import pandas as pd
import re

Test_nonultrametric = "((((((((A:20,(B:1,C:1):9):1,D:11):4,E:15):3,F:18):12,G:30):11,H:41):2,I:43):3,J:46);"
Test_tree = "((((((((Avvvvvvvvvvvvvvvvvvvvvvvvv:10,(B:1,C:1):9):1,D:11):4,E:15):3,F:18):12,G:30):11,H:41):2,(I:41,J:41):2):3,K:46);"

def gettotallength(tree):
    diss = max([tree.distance(tip) for tip in tree.get_terminals()])
    return diss

def gettotalspecies(tree):
    return len(tree.get_terminals())+len(tree.get_nonterminals())

def getdepths_sizes(root):
    Depths = root.depths(unit_branch_lengths=True)
    root_branch_length = min(Depths.values())
    Depths = {clade:int(round(depth-root_branch_length)) for clade,depth in Depths.items()}
    depths_sizes_dic = {}
    maxi_depth = 0
    depths_sizeordered = {}
    clades_size = {}
    clades_alltips = {}
    identifier = 0
    for clade,depth in Depths.items():
        if depth>=maxi_depth: maxi_depth = depth
        if clade.name is None:
            clade.name = "Depth_{}_ID_{}".format(depth,identifier)
            identifier+=1
        clade_ = copy.deepcopy(clade)
        clade_.collapse_all()
        size = len(clade_.clades)
        clades_size[clade.name] = size
        depths_sizes_dic[clade.name] = (depth,size)
        clades_alltips[clade.name] = [tip.name for tip in clade_.clades]
        if depth not in depths_sizeordered:
            depths_sizeordered[depth] = [(clade.name,size)]
        else:
            depths_sizeordered[depth] += [(clade.name,size)]
            depths_sizeordered[depth] = sorted(depths_sizeordered[depth],key=lambda x:x[1])
    return depths_sizes_dic,maxi_depth,depths_sizeordered,clades_size,clades_alltips

def findcladebyname(tree,name):
    return next(tree.find_clades(name))

class TreeBuilder:
    def __init__(self, tree, topologylw=1,userfig=None,userax=None,fs=(10,10),tiplabelroffset=0.02,tiplabelthetaoffset=0,starttheta=0,tiplabelxoffset=0.02,tiplabelyoffset=0,showtiplabel=True,plottipnode=False,shownodelabel=False,plotnnode=True,nodelabelroffset=0.01,nodelabelthetaoffset=0,plotnodeuncertainty=False,nulw=1,nucalpha=0.4,nuccolor='blue',userbranchcolor=None,tiplabelalign='left',nodelabelalign='left',tiplabelsize=10,tiplabelalpha=1,tiplabelcolor='k',tipnodesize=6,tipnodecolor='k',tipnodealpha=1,tiplabelstyle='normal',tipnodemarker='o',nodelabelsize=10,nodelabelalpha=1,nodelabelcolor='k',nnodesize=3,nnodecolor='k',nnodealpha=1,nodelabelstyle='normal',nnodemarker='o',nodelabelxoffset=0.02,nodelabelyoffset=0,ubrobject=None,extinct_lineages=[]):
        self.tree = tree
        self.root = tree.root
        self.root_depth_size_dic,self.maxi_depth,self.depths_sizeordered,self.clades_size,self.clades_alltips = getdepths_sizes(self.root)
        self.nodes = [node for node in tree.get_nonterminals()]
        self.tips = [tip for tip in tree.get_terminals()]
        self.checkdupids()
        self.Total_length = gettotallength(self.tree)
        self.Total_species = gettotalspecies(self.tree)
        self.topologylw = topologylw;self.userfig = userfig;self.userax = userax
        self.fs=fs;self.tiplabelroffset = tiplabelroffset;self.tiplabelthetaoffset = tiplabelthetaoffset;self.starttheta = starttheta
        self.tiplabelxoffset = tiplabelxoffset;self.tiplabelyoffset = tiplabelyoffset;self.showtiplabel = showtiplabel
        self.plottipnode = plottipnode;self.shownodelabel = shownodelabel;self.plotnnode = plotnnode;self.nodelabelroffset = nodelabelroffset
        self.nodelabelthetaoffset = nodelabelthetaoffset
        self.plotnodeuncertainty = plotnodeuncertainty;self.nucalpha = nucalpha;self.nulw = nulw;self.nuccolor = nuccolor
        self.userbranchcolor = userbranchcolor;self.tiplabelalign = tiplabelalign;self.nodelabelalign = nodelabelalign
        self.tiplabelsize = tiplabelsize;self.tiplabelalpha = tiplabelalpha;self.tiplabelcolor = tiplabelcolor;self.tipnodesize = tipnodesize
        self.tipnodecolor = tipnodecolor;self.tipnodealpha = tipnodealpha;self.tiplabelstyle = tiplabelstyle;self.tipnodemarker = tipnodemarker
        self.nodelabelsize = nodelabelsize;self.nodelabelalpha = nodelabelalpha;self.nodelabelcolor = nodelabelcolor
        self.nnodesize = nnodesize; self.nnodecolor = nnodecolor;self.nnodealpha = nnodealpha;self.nodelabelstyle = nodelabelstyle
        self.nnodemarker = nnodemarker;self.nodelabelxoffset = nodelabelxoffset;self.nodelabelyoffset = nodelabelyoffset
        self.ubrobject = ubrobject
        self.extinct_lineages = extinct_lineages
    def checkdupids(self):
        node_ids = [node.name for node in self.nodes if node.name is not None]
        tip_ids = [tip.name for tip in self.tips if tip.name is not None]
        all_ids = node_ids + tip_ids
        assert len(all_ids) == len(set(all_ids))
    def polardraw(self,polar=355,log="Plotting circular tree"):
        logging.info(log)
        self.endtheta = polar
        if self.userfig is None and self.userax is None:
            fig, ax = plt.subplots(1,1,figsize=self.fs,subplot_kw={'projection': 'polar'})
            self.fig,self.ax = fig,ax
        else:
            self.fig,self.ax = self.userfig,self.userax
        self.drawtipspolar()
        self.drawnodespolar()
        self.drawlinespolar()
        self.scaler_theta = abs(self.endtheta-self.starttheta)/100
        thetamin,thetamax = self.starttheta-self.scaler_theta,min([360,self.endtheta+self.scaler_theta])
        if thetamax-thetamin >=360: thetamax = 360 + thetamin
        #self.ax.set_thetamin(thetamin);self.ax.set_thetamax(thetamax)
        self.ax.spines['polar'].set_visible(False)
        self.ax.grid(False)
        self.ax.set_rticks([])
        self.ax.set_xticks([])
        self.ax.axis('off')
    def highlightnodepolar(self,nodes=[],colors=[],nodesizes=[],nodealphas=[],nodemarkers=[]):
        if len(nodes) == 0:
            return
        if colors == []:
            colors = ['k' for i in range(len(nodes))]
        if nodesizes == []:
            nodesizes = np.full(len(nodes),self.nodesize)
        if nodealphas == []:
            nodealphas = np.full(len(nodes),1)
        if nodemarkers == []:
            nodemarkers = ['o' for i in range(len(nodes))]
        for node,cr,ns,al,marker in zip(nodes,colors,nodesizes,nodealphas,nodemarkers):
            if type(node) is not str:
                node = self.tree.common_ancestor(*node).name
            if node not in self.allnodes_thetacoordinates or node not in self.allnodes_rcoordinates:
                logging.error("Cannot find {} in the tree!".format(node))
                exit(0)
            thetacoor,rcoor = self.allnodes_thetacoordinates[node],self.allnodes_rcoordinates[node]
            self.ax.plot((thetacoor,thetacoor),(rcoor,rcoor),marker=marker,alpha=al,markersize=ns,color=cr)
    def highlightcladepolar(self,clades=[],facecolors=[],alphas=[],lws=[],gradual=True,leftoffset=None,rightoffset=None,bottomoffset=None,topoffset=None):
        if len(clades) == 0:
            return
        if facecolors == []:
            facecolors = ['blue' for i in range(len(clades))]
        if alphas == []:
            alphas = np.full(len(clades),1)
        if lws == []:
            lws = np.full(len(clades),self.topologylw)
        for clade,fcr,al,lw in zip(clades,facecolors,alphas,lws):
            if type(clade) is not str:
                clade = self.tree.common_ancestor(*clade).name
            thetacs = [self.allnodes_thetacoordinates[tip.name] for tip in findcladebyname(self.tree,clade).get_terminals()]
            rcs = [self.allnodes_rcoordinates[tip.name] for tip in findcladebyname(self.tree,clade).get_terminals()]
            rcoor = self.allnodes_rcoordinates[clade]
            thetacoor = min(thetacs)
            width = max(rcs) - rcoor
            height = max(thetacs) - min(thetacs)
            if leftoffset is not None: rcoor += self.Total_length*leftoffset
            if rightoffset is not None: width += self.Total_length*rightoffset
            if bottomoffset is not None: thetacoor += (self.endtheta-self.starttheta)*bottomoffset
            if topoffset is not None: height += (self.endtheta-self.starttheta)*topoffset
            if gradual:
                color_limits_rgba = [to_rgba(fcr, alpha=0.1),to_rgba(fcr, alpha=al)]
                cmap = LinearSegmentedColormap.from_list("alpha_gradient",color_limits_rgba)
                r = np.linspace(rcoor, rcoor + width, 100)
                theta = np.linspace(thetacoor, thetacoor + height, 360)
                R, Theta = np.meshgrid(r, theta)
                self.ax.pcolormesh(Theta, R, R, cmap=cmap, shading='auto')
            else:
                r = np.linspace(rcoor, rcoor + width, 100)
                theta = np.linspace(thetacoor, thetacoor + height, 360)
                R, Theta = np.meshgrid(r, theta)
                self.ax.pcolormesh(Theta, R, np.ones_like(R), color=fcr, shading='auto',alpha=al)
    def basicdraw(self,log="Plotting tree"):
        logging.info(log)
        if self.userfig is None and self.userax is None:
            fig, ax = plt.subplots(1,1,figsize=self.fs)
        else:
            fig, ax = self.userfig, self.userax
        #ax.set_ylim(0,len(self.tips)+1)
        self.fig,self.ax = fig,ax
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.set_xticks([])  # Remove x ticks
        self.ax.set_yticks([])  # Remove y ticks
        self.ax.set_xticklabels([])  # Remove x tick labels
        self.ax.set_yticklabels([])  # Remove y tick labels
        self.drawtips()
        self.drawnodes()
        self.drawlines()
        #self.drawlines(ubr=userbranchcolor,topologylw=topologylw)
    def showlegend(self,**kwargs):
        if "fontsize" not in kwargs: kwargs.update({'fontsize':self.tiplabelsize})
        self.ax.legend(**kwargs)
    def highlightnode(self,nodes=[],colors=[],nodesizes=[],nodealphas=[],nodemarkers=[],addlegend=False,legendlabel=None):
        if len(nodes) == 0:
            return
        if colors == []:
            colors = ['k' for i in range(len(nodes))]
        if nodesizes == []:
            nodesizes = np.full(len(nodes),self.nnodesize)
        if nodealphas == []:
            nodealphas = np.full(len(nodes),1)
        if nodemarkers == []:
            nodemarkers = ['o' for i in range(len(nodes))]
        for node,cr,ns,al,marker,ind in zip(nodes,colors,nodesizes,nodealphas,nodemarkers,range(len(nodes))):
            if type(node) is not str:
                node = self.tree.common_ancestor(*node).name
            if node not in self.allnodes_xcoordinates or node not in self.allnodes_ycoordinates:
                logging.error("Cannot find {} in the tree!".format(node))
                exit(0)
            xcoor,ycoor = self.allnodes_xcoordinates[node],self.allnodes_ycoordinates[node]
            if ind == 0 and addlegend:
                self.ax.plot((xcoor,xcoor),(ycoor,ycoor),marker=marker,alpha=al,markersize=ns,color=cr,label=legendlabel)
            else:
                self.ax.plot((xcoor,xcoor),(ycoor,ycoor),marker=marker,alpha=al,markersize=ns,color=cr)
    def highlightclade(self,clades=[],facecolors=[],alphas=[],lws=[],gradual=True,leftoffset=None,rightoffset=0.01,bottomoffset=-0.01,topoffset=0.02,labels=[],labelsize=None,labelstyle=None,labelcolors=[],labelalphas=[],labelva='center',labelha='center',labelrt=None,labelboxcolors=[],labelboxedgecolors=[],labelboxalphas=[],labelboxpads=[],labelxoffset=None,labelyoffset=None):
        if len(clades) == 0:
            return
        if facecolors == []:
            facecolors = ['gray' for i in range(len(clades))]
        if alphas == []:
            alphas = np.full(len(clades),0.5)
        if lws == []:
            lws = np.full(len(clades),self.topologylw)
        for clade,fcr,al,lw,ind in zip(clades,facecolors,alphas,lws,range(len(clades))):
            if type(clade) is not str:
                clade = self.tree.common_ancestor(*clade).name 
            xcoor = self.allnodes_xcoordinates[clade]
            xcs = [self.allnodes_xcoordinates[tip.name] for tip in findcladebyname(self.tree,clade).get_terminals()]
            ycs = [self.allnodes_ycoordinates[tip.name] for tip in findcladebyname(self.tree,clade).get_terminals()]
            ycoor = min(ycs)
            height = max(ycs) - min(ycs)
            width = max(xcs) - xcoor
            if leftoffset is not None: xcoor += self.Total_length*leftoffset
            if rightoffset is not None: width += self.Total_length*rightoffset
            if bottomoffset is not None: ycoor += len(self.tips)*bottomoffset
            if topoffset is not None: height += len(self.tips)*topoffset
            resolution_x,resolution_y = 2000,2000
            if gradual:
                minimal = 0 if al < 0.1 else 0.1
                color_limits_rgba = [to_rgba(fcr, alpha=0.1),to_rgba(fcr, alpha=al)]
                cmap = LinearSegmentedColormap.from_list("alpha_gradient",color_limits_rgba)
                gradient = np.linspace(0, 1, resolution_x).reshape(1, -1)
                gradient = np.repeat(gradient, resolution_y, axis=0)
                self.ax.imshow(gradient,extent=(xcoor, xcoor+width, ycoor, ycoor+height), origin="lower", aspect="auto", cmap=cmap, interpolation="antialiased")
            else:
                color_rgb = np.array(colors.to_rgb(fcr))
                color_array = np.ones((resolution_x, resolution_y, 4))
                color_array[..., :3] = color_rgb
                color_array[..., 3] = al
                self.ax.imshow(color_array,extent=(xcoor, xcoor+width, ycoor, ycoor+height),origin="lower",aspect="auto")
            if labels != []:
                xcoor_text = xcoor
                if labelxoffset is not None: xcoor_text += labelxoffset*self.Total_length
                ycoor_text = ycoor+height
                if labelyoffset is not None: ycoor_text += labelyoffset*len(self.tips)
                if labelsize is None: labelsize = [self.tiplabelsize for i in range(len(clades))]
                if labelstyle is None: labelstyle = ['normal' for i in range(len(clades))]
                if labelcolors == []: labelcolors = ['black' for i in range(len(clades))]
                if labelalphas == []: labelalphas = [1 for i in range(len(clades))]
                if labelboxcolors == []: labelboxcolors = ['none' for i in range(len(clades))]
                if labelboxalphas == []: labelboxalphas = [1 for i in range(len(clades))]
                if labelboxpads == []: labelboxpads = [1 for i in range(len(clades))]
                if labelboxedgecolors == []: labelboxedgecolors = ['none' for i in range(len(clades))]
                self.ax.text(xcoor_text,ycoor_text,labels[ind],fontsize=labelsize[ind],style=labelstyle[ind],color=labelcolors[ind],alpha=labelalphas[ind],va=labelva,ha=labelha,rotation=labelrt,bbox={'facecolor':labelboxcolors[ind],'alpha': labelboxalphas[ind],'pad':labelboxpads[ind],'edgecolor':labelboxedgecolors[ind]})
    def saveplot(self,outpath,dpi=200):
        self.fig.tight_layout()
        self.fig.savefig(outpath,dpi=dpi)
        plt.close()
    def drawtipspolar(self):
        tips_rcoordinates = {tip.name:self.root.distance(tip) for tip in self.tips}
        self.Total_theta = self.endtheta/180*np.pi
        self.per_sp_theta = (self.endtheta-self.starttheta)/180*np.pi/(len(self.tips)-1)
        thetacoordinate = self.starttheta/180*np.pi - self.per_sp_theta
        tips_thetacoordinates = {}
        clades_sizeordered = self.depths_sizeordered[1]
        for clades_size in clades_sizeordered:
            clade_name,size = clades_size
            clade = findcladebyname(self.tree,clade_name)
            if clade.is_terminal():
                thetacoordinate+=self.per_sp_theta
                tips_thetacoordinates[clade_name] = thetacoordinate
            else:
                thetacoordinate,tips_thetacoordinates = polarrecursionuntilalltip(clade,thetacoordinate,tips_thetacoordinates,self.clades_size,self.per_sp_theta)
        for tip in sorted(self.tips,key=lambda x:tips_thetacoordinates[x.name]):
            if self.plottipnode: self.ax.plot(tips_thetacoordinates[tip.name],tips_rcoordinates[tip.name],marker=self.tipnodemarker,markersize=self.tipnodesize,color=self.tipnodecolor,alpha=self.tipnodealpha)
            if self.showtiplabel:
                angle=thetatovalue(tips_thetacoordinates[tip.name])
                if angle <= 90 or angle > 270:
                    rotation = angle
                    labelalign_ = self.tiplabelalign
                else:
                    rotation = angle + 180
                    if self.tiplabelalign == 'left': labelalign_ = 'right'
                    if self.tiplabelalign == 'right': labelalign_ = 'left'
                text = self.ax.text(tips_thetacoordinates[tip.name]+self.Total_theta*self.tiplabelthetaoffset,tips_rcoordinates[tip.name]+self.Total_length*self.tiplabelroffset,tip.name,rotation=rotation,rotation_mode='anchor',va='center',ha=labelalign_,fontsize=self.tiplabelsize,fontstyle=self.tiplabelstyle,color=self.tiplabelcolor,alpha=self.tiplabelalpha)
        self.tips_thetacoordinates,self.tips_rcoordinates = tips_thetacoordinates,tips_rcoordinates
        self.allnodes_thetacoordinates,self.allnodes_rcoordinates = {**tips_thetacoordinates},{**tips_rcoordinates}
    def drawnodespolar(self):
        self.nodes_rcoordinates = {}
        self.nodes_thetacoordinates = {}
        if self.plotnodeuncertainty:
            logging.info("Adding node uncertainty")
        for node in self.nodes:
            children = findfirstchildren(node)
            self.nodes_thetacoordinates[node.name] = recursiongetycor(children,self.tips_thetacoordinates)
            self.nodes_rcoordinates[node.name] = self.root.distance(node)
            if self.plotnnode: self.ax.plot(self.nodes_thetacoordinates[node.name],self.nodes_rcoordinates[node.name],marker=self.nnodemarker,markersize=self.nnodesize,color=self.nnodecolor,alpha=self.nnodealpha)
            angle=thetatovalue(self.nodes_thetacoordinates[node.name])
            if angle <= 90 or angle > 270:
                rotation = angle
                labelalign_ = self.nodelabelalign
            else:
                rotation = angle + 180
                if self.nodelabelalign == 'left': labelalign_ = 'right'
                if self.nodelabelalign == 'right': labelalign_ = 'left'
            if self.shownodelabel: self.ax.text(self.nodes_thetacoordinates[node.name]+self.Total_theta*self.nodelabelthetaoffset,self.nodes_rcoordinates[node.name]+self.Total_length*self.nodelabelroffset,node.name,ha=labelalign_,va='center',rotation_mode='anchor',rotation=rotation,fontsize=self.nodelabelsize,alpha=self.nodelabelalpha,color=self.nodelabelcolor,fontstyle=self.nodelabelstyle)
            if self.plotnodeuncertainty:
                nodeuncertainty = getnuc(node)
                if None in nodeuncertainty:
                    continue
                nodeuncertainty = -np.array(nodeuncertainty)+self.Total_length
                self.ax.plot((self.nodes_thetacoordinates[node.name],self.nodes_thetacoordinates[node.name]),(nodeuncertainty[1],nodeuncertainty[0]),lw=self.nuclw,color=nucc,alpha=self.nucalpha)
        self.allnodes_thetacoordinates = {**self.allnodes_thetacoordinates,**self.nodes_thetacoordinates}
        self.allnodes_rcoordinates = {**self.allnodes_rcoordinates,**self.nodes_rcoordinates}
    def drawlinespolar(self,rbr=False):
        ## TODO Jagged joint point
        if self.userbranchcolor is None:
            if self.ubrobject is not None: branch_colors = ubrobject
            else:
                branch_colors = {**{tip.name:'black' for tip in self.tips},**{node.name:'black' for node in self.nodes}}
            if rbr:
                for key in branch_colors: branch_colors[key] = random_color_hex()
        else:
            branch_colors = getubr(self.userbranchcolor)
        #thetass,rss,crs = [[] for _ in range(3)]
        for tip in self.tips:
            ls = '--' if tip.name in self.extinct_lineages else '-'
            thetass,rss = [],[]
            firstparent = getfirstparent(tip,self.nodes)
            segment_length = self.root.distance(tip)-self.root.distance(firstparent)
            rmin,rmax = self.root.distance(firstparent),self.root.distance(tip)
            self.ax.plot((self.tips_thetacoordinates[tip.name],self.tips_thetacoordinates[tip.name]),(rmin,rmax),color=branch_colors.get(tip.name,'k'),lw=self.topologylw,solid_joinstyle='round',ls=ls)
            thetass +=[self.tips_thetacoordinates[tip.name],self.tips_thetacoordinates[tip.name]];rss+=[rmin,rmax]
            thetamin, thetamax = sorted([self.tips_thetacoordinates[tip.name],self.nodes_thetacoordinates[firstparent.name]])
            thetas = np.linspace(thetamin, thetamax, 1000)
            self.ax.plot(thetas,np.full(len(thetas),self.nodes_rcoordinates[firstparent.name]),color=branch_colors.get(tip.name,'k'),lw=self.topologylw,solid_joinstyle='round',ls=ls)
            #thetass += [i for i in thetas];rss+=[i for i in np.full(len(thetas),self.nodes_rcoordinates[firstparent.name])]
            #self.ax.plot(thetass,rss,color=branch_colors.get(tip.name,'k'),lw=self.topologylw,solid_joinstyle='round')
            #thetass +=[i for i in thetas];rss+=[i for i in np.full(len(thetas),self.nodes_rcoordinates[firstparent.name])]
            #crs+= [branch_colors.get(tip.name,'black') for _ in range(len(thetas))]
        for node in self.nodes:
            thetass,rss = [],[]
            if self.root.distance(node) == 0:
                if node.branch_length is None:
                    continue
                else:
                    self.ax.plot((self.nodes_thetacoordinates[node.name],self.nodes_thetacoordinates[node.name]),(0,-self.nodes_rcoordinates[node.name]),color=branch_colors.get(node.name,'k'),lw=self.topologylw,solid_joinstyle='round')
                    continue
            firstparent = getfirstparent(node,self.nodes)
            segment_length = self.root.distance(node)-self.root.distance(firstparent)
            rmin,rmax = self.root.distance(firstparent),self.root.distance(node)
            self.ax.plot((self.nodes_thetacoordinates[node.name],self.nodes_thetacoordinates[node.name]),(rmin,rmax),color=branch_colors.get(node.name,'k'),lw=self.topologylw,solid_joinstyle='round')
            thetass += [self.nodes_thetacoordinates[node.name],self.nodes_thetacoordinates[node.name]];rss +=[rmin,rmax]
            thetamin,thetamax = sorted([self.nodes_thetacoordinates[node.name],self.nodes_thetacoordinates[firstparent.name]])
            thetas = np.linspace(thetamin, thetamax, 1000)
            self.ax.plot(thetas,np.full(len(thetas),self.nodes_rcoordinates[firstparent.name]),color=branch_colors.get(node.name,'k'),lw=self.topologylw,solid_joinstyle='round')
            #thetass +=[i for i in thetas];rss +=[i for i in np.full(len(thetas),self.nodes_rcoordinates[firstparent.name])]
            #self.ax.plot(thetass,rss,color=branch_colors.get(node.name,'k'),lw=self.topologylw,solid_joinstyle='round')
            #crs +=[branch_colors.get(node.name,'black') for _ in range(len(thetas))]
        #self.ax.plot(thetass,crs,color='k',lw=self.topologylw)
    def drawscalepolar(self,plotfulllengthscale=False,inipoint=(0,0),endpoint=(0,0),scalecolor='k',scalelw=None,fullscalelw=None,fullscalexticks=None,fullscalecolor='k',fullscalels='-'):
        if plotfulllengthscale:
            rmin,rmax = 0,self.Total_length
            if fullscalelw is None: fullscalelw=self.topologylw
            if fullscalexticks is None: fullscalexticks = np.linspace(rmin,rmax,6)
            for tick in fullscalexticks:
                thetas = np.linspace(self.starttheta/180*np.pi,self.endtheta/180*np.pi, 500)
                rs = np.full(len(thetas),tick)
                self.ax.plot(thetas,rs,lw=fullscalelw,color=fullscalecolor,ls=fullscalels)
        if inipoint!=endpoint:
            print('Yes')
            degree1,r1 = inipoint
            degree2,r2 = endpoint
            theta1,theta2 = degree1/180*np.pi,degree2/180*np.pi
            if scalelw is None: slw = self.topologylw
            self.ax.plot((theta1,theta2),(r1,r2),color=self.scalecolor,lw=slw) 
    def drawscale(self,plotfulllengthscale=False,inipoint=(0,0),endpoint=(0,0),scalecolor='k',scalelw=None,fullscalelw=None,fullscaley=0,fullscalexticks=None,fullscalecolor='k',fullscaleticklw=None,fullscaletickcolor='k',fullscaletickheight=0.1,fullscaleticklabels=None,fullscaleticklabelsize=None,fullscaleticklabelcolor='k',fullscaleticklabeloffset=0.1,scaler_y=0.25,addgeo=False,geoscaling=1,geouppery=0.5,geolowery=0.1):
        Ymin,Ymax = [],[]
        if plotfulllengthscale:
            if fullscalelw is None: fullscalelw = self.topologylw
            if fullscaleticklw is None: fullscaleticklw = self.topologylw
            if fullscaleticklabelsize is None: fullscaleticklabelsize = self.tiplabelsize
            xmin,xmax = 0,self.Total_length*geoscaling
            ycoordi = fullscaley
            if fullscalexticks is None:
                xticks = np.linspace(xmin,xmax,6)
            else:
                xticks = [xmax-i for i in fullscalexticks]
            if fullscaleticklabels is None:
                if fullscalexticks is None:
                    fullscaleticklabels = ["{:.1f}".format(float(tick)) for tick in xticks[::-1]]
                else:
                    fullscaleticklabels = [str(tick) for tick in fullscalexticks]
            self.ax.plot((xticks[-1]/geoscaling,xticks[0]/geoscaling), (ycoordi,ycoordi), color=fullscalecolor, linewidth=fullscalelw)
            y2 = fullscaletickheight
            for tick,ticklabel in zip(xticks,fullscaleticklabels):
                tick = tick/geoscaling
                self.ax.plot((tick,tick), (ycoordi,ycoordi-y2), color=fullscaletickcolor, linewidth=fullscaleticklw)
                self.ax.text(tick,ycoordi-y2-fullscaleticklabeloffset,ticklabel,fontsize=fullscaleticklabelsize,color=fullscaleticklabelcolor,ha='center',va='top')
            self.ax.text(self.Total_length+self.Total_length*self.tiplabelxoffset,ycoordi-y2-fullscaleticklabeloffset,"Mya",ha='left',va='top',fontsize=fullscaleticklabelsize,color=fullscaleticklabelcolor)
            y2+=fullscaleticklabeloffset
        else:
            ycoordi,y2 = 0,0
        if inipoint!=endpoint:
            if scalelw is None: slw = self.topologylw
            x1,y1 = inipoint
            x2,y2 = endpoint
            self.ax.plot((x1,x2),(y1,y2),color=scalecolor,linewidth=slw)
            ymin,ymax = sorted([y1,y2])
            ymin,ymax = min([ymin,0]),max([ymax,len(self.tips)+1])
            Ymin,Ymax = min([ymin,ycoordi-y2]),max([ymax,ycoordi-y2])
        else:
            Ymin,Ymax = min([0,ycoordi-y2]),max([len(self.tips)+1,ycoordi-y2])
        if addgeo:
            time = 0
            epoch_boundaries = [2.58,23.03,66.0,145.0,201.4,251.902,298.9,358.9,419.2,443.8,485.4,538.8,1000,1600,2500,2800,3200,3600,4031]
            epoch_labels = ["","Neo","Paleogene","Cretaceous","Jurassic","Triassic","Permian","Carboniferous","Devonian","Silurian","Ordovician","Cambrian","Neo-proterozoic","Meso-proterozoic","Paleo-proterozoic","Neo-archean","Meso-archean","Paleo-archean","Eo-archean"]
            epoch_colors = ["#fff880ff","#fddd1cff","#f89b5cff","#80cf5cff","#33bde9ff","#8a3ea4ff","#e74d40ff","#69b2b0ff","#ca9547ff","#b3e4c2ff","#00a990ff","#83ad6aff","#fab24aff","#f9b269ff","#f1457eff","#f799c7ff","#f467b2ff","#f140a9ff","#d9058dff"]
            self.scaled_Total_length = self.Total_length*geoscaling
            left_bound = 0
            for boundary,la,cr in zip(epoch_boundaries,epoch_labels,epoch_colors):
                if self.scaled_Total_length >= boundary:
                    left = self.scaled_Total_length - boundary
                    bottom = geolowery
                    width = boundary - left_bound
                    height = geouppery
                    rect = Rectangle((left/geoscaling, bottom), width/geoscaling, height, facecolor=cr, alpha=1)
                    self.ax.add_patch(rect)
                else:
                    left = 0
                    bottom = geolowery
                    width = self.scaled_Total_length - left_bound
                    height = geouppery
                    rect = Rectangle((left/geoscaling, bottom), width/geoscaling, height, facecolor=cr, alpha=1)
                    self.ax.add_patch(rect)
                    break
                left_bound = boundary
        self.ax.set_ylim(Ymin-scaler_y,Ymax+scaler_y)
    def drawwgdpolar(self,wgd=None,cr='r',al=0.6,lw=4):
        if wgd is None:
            return
        logging.info("Adding WGD")
        df = pd.read_csv(wgd,header=0,index_col=None,sep='\t').drop_duplicates(subset=['WGD ID'])
        self.allnodes_thetacoordinates = {**self.nodes_thetacoordinates,**self.tips_thetacoordinates}
        for fullsp,hcr in zip(df["Full_Species"],df["90% HCR"]):
            sps = fullsp.split(", ")
            node = self.tree.common_ancestor(*sps)
            lower,upper = [float(i)/100 for i in hcr.split('-')]
            thetacoordi = self.allnodes_thetacoordinates[node.name]
            self.ax.plot((thetacoordi,thetacoordi),(self.Total_length-upper,self.Total_length-lower),lw=lw,color=cr,alpha=al)
    def drawtips(self):
        tips_ycoordinates = {}
        ycoordinate = 0
        tips_xcoordinates = {tip.name:self.root.distance(tip) for tip in self.tips}
        clades_sizeordered = self.depths_sizeordered[1] # first children from root
        #clades_sizeordered = sorted(self.depths_sizeordered.items(),key=lambda x:x[0])[1]
        for clades_size in clades_sizeordered:
            clade_name,size = clades_size
            clade = findcladebyname(self.tree,clade_name)
            if clade.is_terminal():
                ycoordinate+=1
                tips_ycoordinates[clade_name] = ycoordinate
            else:
                ycoordinate,tips_ycoordinates = recursionuntilalltip(clade,ycoordinate,tips_ycoordinates,self.clades_size)
        for tip in self.tips:
            if self.plottipnode: self.ax.plot(tips_xcoordinates[tip.name],tips_ycoordinates[tip.name],marker=self.tipnodemarker,markersize=self.tipnodesize,color=self.tipnodecolor,alpha=self.tipnodealpha)
            if self.showtiplabel:
                text = self.ax.text(tips_xcoordinates[tip.name]+self.Total_length*self.tiplabelxoffset,tips_ycoordinates[tip.name]+len(self.tips)*self.tiplabelyoffset,tip.name,ha='left',va='center',fontsize=self.tiplabelsize,fontstyle=self.tiplabelstyle,alpha=self.tiplabelalpha,color=self.tiplabelcolor)
        self.tips_ycoordinates,self.tips_xcoordinates = tips_ycoordinates,tips_xcoordinates
        self.allnodes_ycoordinates,self.allnodes_xcoordinates = {**tips_ycoordinates},{**tips_xcoordinates}
    def addtext2node(self,nodedic,textxoffset=0,textyoffset=0,textsize=5,textalpha=1,textstyle='normal',textcolor='k'):
        logging.info("Adding text to node")
        for node,text in nodedic.items():
            self.ax.text(self.allnodes_xcoordinates[node]+self.Total_length*textxoffset,self.allnodes_ycoordinates[node]+len(self.tips)*textyoffset,text,ha='left',va='center',fontsize=textsize,alpha=textalpha,fontstyle=textstyle,color=textcolor)
    
    def addextranodes(self,nodenames,marker='x',markersize=5,markercolor='k',markeralpha=1,labels=None,labelxoffset=0.1,labelyoffset=0.1,fontsize=15,labelalpha=1,fontstyle='normal',labelcolor='k'):
        if labels is not None:
            assert len(nodenames) == len(labels)
            for nodename,label in zip(nodenames,labels):
                self.ax.plot(self.allnodes_xcoordinates[nodename],self.allnodes_ycoordinates[nodename],marker=marker,markersize=markersize,color=markercolor,alpha=markeralpha)
                self.ax.text(self.allnodes_xcoordinates[nodename]+self.Total_length*labelxoffset,self.allnodes_ycoordinates[nodename]+len(self.tips)*labelyoffset,label,ha='left',va='center',fontsize=fontsize,alpha=labelalpha,fontstyle=fontstyle,color=labelcolor)
        else:
            for nodename in nodenames:
                self.ax.plot(self.allnodes_xcoordinates[nodename],self.allnodes_ycoordinates[nodename],marker=marker,markersize=markersize,color=markercolor,alpha=markeralpha)

    def drawnodes(self):
        self.nodes_ycoordinates = {}
        self.nodes_xcoordinates = {}
        if self.plotnodeuncertainty:
            logging.info("Adding node uncertainty")
        for node in self.nodes:
            children = findfirstchildren(node)
            self.nodes_ycoordinates[node.name] = recursiongetycor(children,self.tips_ycoordinates)
            self.nodes_xcoordinates[node.name] = self.root.distance(node)
            if self.plotnnode: self.ax.plot(self.nodes_xcoordinates[node.name],self.nodes_ycoordinates[node.name],marker=self.nnodemarker,markersize=self.nnodesize,color=self.nnodecolor,alpha=self.nnodealpha)
            if self.shownodelabel: self.ax.text(self.nodes_xcoordinates[node.name]+self.Total_length*self.nodelabelxoffset,self.nodes_ycoordinates[node.name]+len(self.tips)*self.nodelabelyoffset,node.name,ha='left',va='center',fontsize=self.nodelabelsize,alpha=self.nodelabelalpha,fontstyle=self.nodelabelstyle,color=self.nodelabelcolor)
            if self.plotnodeuncertainty:
                nodeuncertainty = getnuc(node)
                if None in nodeuncertainty:
                    continue
                nodeuncertainty = -np.array(nodeuncertainty)+self.Total_length
                self.ax.plot((nodeuncertainty[1],nodeuncertainty[0]),(self.nodes_ycoordinates[node.name],self.nodes_ycoordinates[node.name]),lw=self.nulw,color=self.nuccolor,alpha=self.nucalpha)
        self.allnodes_ycoordinates = {**self.allnodes_ycoordinates,**self.nodes_ycoordinates}
        self.allnodes_xcoordinates = {**self.allnodes_xcoordinates,**self.nodes_xcoordinates}
    def drawlines(self,rbr=False):
        drawed_nodes = []
        if self.userbranchcolor is None:
            if self.ubrobject is None:
                branch_colors = {**{tip.name:'k' for tip in self.tips},**{node.name:'k' for node in self.nodes}}
            else:
                branch_colors = self.ubrobject
            if rbr:
                for key in branch_colors: branch_colors[key] = random_color_hex()
        else:
            branch_colors = getubr(self.userbranchcolor)
        for tip in self.tips:
            ls = '--' if tip.name in self.extinct_lineages else '-'
            firstparent = getfirstparent(tip,self.nodes)
            segment_length = self.root.distance(tip)-self.root.distance(firstparent)
            xmin,xmax = self.root.distance(firstparent),self.root.distance(tip)
            self.ax.plot((xmin,xmax),(self.tips_ycoordinates[tip.name],self.tips_ycoordinates[tip.name]),color=branch_colors.get(tip.name,'k'),linewidth=self.topologylw,ls=ls,zorder=0)
            ymin, ymax = sorted([self.tips_ycoordinates[tip.name],self.nodes_ycoordinates[firstparent.name]])
            self.ax.plot((self.nodes_xcoordinates[firstparent.name],self.nodes_xcoordinates[firstparent.name]),(ymin, ymax),color=branch_colors.get(tip.name,'k'),linewidth=self.topologylw,ls=ls,zorder=0)
        for node in self.nodes:
            if self.root.distance(node) == 0:
                if self.root.branch_length is None:
                    continue
                else:
                    self.ax.plot((-self.root.branch_length,0),(self.nodes_ycoordinates[node.name],self.nodes_ycoordinates[node.name]),color=branch_colors.get(node.name,'k'),linewidth=self.topologylw,zorder=0)
                    continue
            firstparent = getfirstparent(node,self.nodes)
            segment_length = self.root.distance(node)-self.root.distance(firstparent)
            xmin,xmax = self.root.distance(firstparent),self.root.distance(node)
            self.ax.plot((xmin,xmax),(self.nodes_ycoordinates[node.name],self.nodes_ycoordinates[node.name]),color=branch_colors.get(node.name,'k'),linewidth=self.topologylw,zorder=0)
            ymin, ymax = sorted([self.nodes_ycoordinates[node.name],self.nodes_ycoordinates[firstparent.name]])
            self.ax.plot((self.nodes_xcoordinates[firstparent.name],self.nodes_xcoordinates[firstparent.name]),(ymin, ymax),color=branch_colors.get(node.name,'k'),linewidth=self.topologylw,zorder=0)
    def drawtrait(self,trait=(),xoffset=0.2,yoffset=0.2,labeloffset=0.2,usedata=(),traitobject=(),traitobjectname=(),traitcolor='k',scalingx=0.1,decimal=1,labelsize=None):
        if trait == () and traitobject == ():
            return
        if labelsize is None: labelsize = self.tiplabelsize
        logging.info("Adding trait")
        updated_offset_bound = 0
        Trait = [*trait,*traitobject]
        #colors = cm.viridis(np.linspace(0, 1, len(Trait)))
        for ind,trait_file in enumerate(Trait):
            xoffset += updated_offset_bound
            if ind < len(trait):
                df = pd.read_csv(trait_file,header=0,index_col=0,sep='\t')
                if usedata!=():
                    trait_dic_orig = {i:j for i,j in zip(df.index,df.loc[:,usedata[ind]])}
                    traitname = usedata[ind]
                else:
                    trait_dic_orig = {i:j for i,j in zip(df.index,df.iloc[:,0])}
                    traitname = df.columns[0]
            else:
                trait_dic_orig,traitname = trait_file,traitobjectname[ind]
            trait_dic = copy.deepcopy(trait_dic_orig)
            scaling = scalingx*self.Total_length
            self.ax.text(self.Total_length*(1+xoffset)+0.5*scaling,len(self.tips)+1,traitname,ha='center',va='center',fontsize=labelsize)
            min_trait = sorted(trait_dic.values())[0]
            if min_trait < 0:
                for key,value in trait_dic.items():
                    trait_dic[key] = value - min_trait # Positivelize all trait values
            max_trait = sorted(trait_dic.values())[-1]
            for key,value in trait_dic.items(): trait_dic[key] = value/max_trait*scaling
            left_coordi = self.Total_length*(xoffset+1)
            widths = []
            ys = []
            real_trait_values = list(trait_dic_orig.values())
            real_Maxi = np.max(real_trait_values)
            scaled_trait_values = list(trait_dic.values())
            scaled_Maxi = np.max(scaled_trait_values)
            scaled_Half = scaled_Maxi/2
            real_Half = real_Maxi/2 if min_trait >= 0 else scaled_Half*max_trait/scaling+min_trait
            for clade,tr in trait_dic.items():
                if clade in self.allnodes_ycoordinates:
                    ys.append(self.allnodes_ycoordinates[clade])
                    widths.append(tr)
                    if tr > updated_offset_bound:
                        updated_offset_bound = tr
            #for tip in self.tips:
            #    widths.append(trait_dic.get(tip.name,0))
                #xmin,xmax = self.tips_xcoordinates[tip.name]+self.Total_length*offset, self.tips_xcoordinates[tip.name]+self.Total_length*offset+trait_dic.get(tip.name,0)
            #    if trait_dic.get(tip.name,0) > updated_offset_bound:
            #        updated_offset_bound = trait_dic.get(tip.name,0)
            #    ys.append(self.tips_ycoordinates[tip.name])
                #self.ax.plot((xmin,xmax),(self.tips_ycoordinates[tip.name],self.tips_ycoordinates[tip.name]),color=colors[ind],lw=2)
            #self.ax.plot((self.Total_length*(1+offset),self.Total_length*(1+offset)),(0,len(self.tips)),color='k',lw=2)
            #for node in self.nodes:
            #    widths.append(trait_dic.get(node.name,0))
            #    if trait_dic.get(node.name,0) > updated_offset_bound:
            #        updated_offset_bound = trait_dic.get(node.name,0)
            #    ys.append(self.nodes_ycoordinates[node.name])
            self.ax.barh(ys,widths,height=0.5,left=left_coordi,align='center',color=traitcolor)
            self.ax.plot((self.Total_length*(1+xoffset),self.Total_length*(1+xoffset)),(0.5,len(self.tips)+0.5),color='k',lw=2)
            self.ax.plot((self.Total_length*(1+xoffset),self.Total_length*(1+xoffset)+scaled_Maxi),(0.5,0.5),color='k',lw=2)
            self.ax.plot((self.Total_length*(1+xoffset)+scaled_Maxi,self.Total_length*(1+xoffset)+scaled_Maxi),(0.5,0.5-yoffset),color='k',lw=2)
            self.ax.plot((self.Total_length*(1+xoffset)+scaled_Maxi/2,self.Total_length*(1+xoffset)+scaled_Maxi/2),(0.5,0.5-yoffset),color='k',lw=2)
            self.ax.plot((self.Total_length*(1+xoffset),self.Total_length*(1+xoffset)),(0.5,0.5-yoffset),color='k',lw=2)
            #self.ax.text(self.Total_length*(1+xoffset)+scaled_Mini,1-yoffset-labeloffset,ha='center',va='top',fontsize=labelsize)
            #self.ax.text(self.Total_length*(1+xoffset)+scaled_Half,0.5-yoffset-labeloffset,'{:.{}f}'.format(real_Half,decimal),ha='center',va='top',fontsize=labelsize)
            self.ax.text(self.Total_length*(1+xoffset)+scaled_Maxi,0.5-yoffset-labeloffset,'{:.{}f}'.format(real_Maxi,decimal),ha='center',va='top',fontsize=labelsize)
    def drawwgd(self,wgd=None,cr='r',al=0.6,lw=4,addlegend=False,legendlabel=None):
        if wgd is None:
            return
        logging.info("Adding WGD")
        df = pd.read_csv(wgd,header=0,index_col=None,sep='\t').drop_duplicates(subset=['WGD ID'])
        self.allnodes_ycoordinates = {**self.nodes_ycoordinates,**self.tips_ycoordinates}
        for fullsp,hcr,ind in zip(df["Full_Species"],df["90% HCR"],range(df.shape[0])):
            sps = fullsp.split(", ")
            node = self.tree.common_ancestor(*sps)
            lower,upper = [float(i)/100 for i in hcr.split('-')]
            ycoordi = self.allnodes_ycoordinates[node.name]
            if ind == 0 and addlegend:
                self.ax.plot((self.Total_length-upper,self.Total_length-lower),(ycoordi,ycoordi),lw=lw,color=cr,alpha=al,label=legendlabel)
            else:
                self.ax.plot((self.Total_length-upper,self.Total_length-lower),(ycoordi,ycoordi),lw=lw,color=cr,alpha=al)
    def transformcm(self,trait):
        norm = Normalize(vmin=np.min(trait), vmax=np.max(trait))
        colormap = plt.cm.viridis
        colors = colormap(norm(trait))
        return colors,norm,colormap
    def getubrobject(self,colors,taxon):
        cladecolors = {clade:cr for clade,cr in zip(taxon,colors)}
        return cladecolors
    def addcolorbar(self,norm,colormap,ax,fig,fraction=0.05, pad=0.04,text='Color bar',fontsize=15):
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax, fraction=fraction, pad=pad)
        cbar.set_label(text,fontsize=fontsize)

def thetatovalue(theta):
    return theta/np.pi*180

def getnuc(node):
    if node.comment is not None:
        return list(map(float, re.findall(r"\{([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\}", node.comment)[0]))
    else:
        return None,None

def getubr(ubr):
    df = pd.read_csv(ubr,header=None,index_col=None,sep='\t')
    branch_colors = {i:j for i,j in zip(df.iloc[:,0],df.iloc[:,1])}
    return branch_colors

def random_color_hex():
    return "#{:06x}".format(np.random.randint(0, high=0xFFFFFF),size=1)

def getfirstparent(target,nodes):
    All_parents = [(node,node.distance(target)) for node in nodes if node.is_parent_of(target) and node != target]
    return sorted(All_parents,key = lambda x:x[1])[0][0]

def recursiongetycor(children,tips_ycor):
    child1,child2 = children
    if child1.is_terminal() and child2.is_terminal():
        return (tips_ycor[child1.name]+tips_ycor[child2.name])/2
    if not child1.is_terminal() and child2.is_terminal():
        children = findfirstchildren(child1)
        child1_ycoor = recursiongetycor(children,tips_ycor)
        return (child1_ycoor+tips_ycor[child2.name])/2
    if child1.is_terminal() and not child2.is_terminal():
        children = findfirstchildren(child2)
        child2_ycoor = recursiongetycor(children,tips_ycor)
        return (child2_ycoor+tips_ycor[child1.name])/2
    if not child1.is_terminal() and not child2.is_terminal():
        children1 = findfirstchildren(child1)
        child1_ycoor = recursiongetycor(children1,tips_ycor)
        children2 = findfirstchildren(child2)
        child2_ycoor = recursiongetycor(children2,tips_ycor)
        return (child2_ycoor+child1_ycoor)/2

def polarrecursionuntilalltip(node,thetacoordinate,tips_thetacoordinates,clade_size,persp_theta):
    child1,child2 = findfirstchildren(node)
    if child1.is_terminal():
        thetacoordinate +=persp_theta
        tips_thetacoordinates[child1.name] = thetacoordinate
    if child2.is_terminal():
        thetacoordinate +=persp_theta
        tips_thetacoordinates[child2.name] = thetacoordinate
    if child1.is_terminal() and child2.is_terminal():
        return thetacoordinate,tips_thetacoordinates
    if child1.is_terminal() and not child2.is_terminal():
        return polarrecursionuntilalltip(child2,thetacoordinate,tips_thetacoordinates,clade_size,persp_theta)
    if not child1.is_terminal() and child2.is_terminal():
        return polarrecursionuntilalltip(child1,thetacoordinate,tips_thetacoordinates,clade_size,persp_theta)
    if not child1.is_terminal() and not child2.is_terminal():
        if clade_size[child1.name] <= clade_size[child2.name]:
            thetacoordinate,tips_thetacoordinates = polarrecursionuntilalltip(child1,thetacoordinate,tips_thetacoordinates,clade_size,persp_theta)
            thetacoordinate,tips_thetacoordinates = polarrecursionuntilalltip(child2,thetacoordinate,tips_thetacoordinates,clade_size,persp_theta)
            return thetacoordinate,tips_thetacoordinates
        else:
            thetacoordinate,tips_thetacoordinates = polarrecursionuntilalltip(child2,thetacoordinate,tips_thetacoordinates,clade_size,persp_theta)
            thetacoordinate,tips_thetacoordinates = polarrecursionuntilalltip(child1,thetacoordinate,tips_thetacoordinates,clade_size,persp_theta)
            return thetacoordinate,tips_thetacoordinates

def recursionuntilalltip(node,ycoordinate,tips_ycoordinates,clade_size):
    child1,child2 = findfirstchildren(node)
    if child1.is_terminal():
        ycoordinate +=1
        tips_ycoordinates[child1.name] = ycoordinate
    if child2.is_terminal():
        ycoordinate +=1
        tips_ycoordinates[child2.name] = ycoordinate
    if child1.is_terminal() and child2.is_terminal():
        return ycoordinate,tips_ycoordinates
    if child1.is_terminal() and not child2.is_terminal():
        return recursionuntilalltip(child2,ycoordinate,tips_ycoordinates,clade_size)
    if not child1.is_terminal() and child2.is_terminal():
        return recursionuntilalltip(child1,ycoordinate,tips_ycoordinates,clade_size)
    if not child1.is_terminal() and not child2.is_terminal():
        if clade_size[child1.name] <= clade_size[child2.name]:
            ycoordinate,tips_ycoordinates = recursionuntilalltip(child1,ycoordinate,tips_ycoordinates,clade_size)
            ycoordinate,tips_ycoordinates = recursionuntilalltip(child2,ycoordinate,tips_ycoordinates,clade_size)
            return ycoordinate,tips_ycoordinates
        else:
            ycoordinate,tips_ycoordinates = recursionuntilalltip(child2,ycoordinate,tips_ycoordinates,clade_size)
            ycoordinate,tips_ycoordinates = recursionuntilalltip(child1,ycoordinate,tips_ycoordinates,clade_size)
            return ycoordinate,tips_ycoordinates 

def findfirstchildren(node):
    Depths = node.depths(unit_branch_lengths=True)
    children = []
    for clade,depth in sorted(Depths.items(), key=lambda x:x[1])[1:3]:
        children += [clade]
    return children

def stringttophylo(string):
    handle = StringIO(string)
    tree = Phylo.read(handle, "newick")
    return tree

def plottree(tree=None,treeobject=None,polar=None,fs=(10,10),trait=(),usedtraitcolumns=(),wgd=None,output=None):
    if treeobject is None:
        if tree is None:
            Tree = stringttophylo(Test_tree)
        else:
            Tree = Phylo.read(tree,format='newick')
    else:
        Tree = treeobject
    TB = TreeBuilder(Tree)
    return TB,Tree
