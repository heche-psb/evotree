import logging
from Bio import Phylo
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.patches import Rectangle
from matplotlib import colors
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
    def __init__(self, tree):
        self.tree = tree
        self.root = tree.root
        self.root_depth_size_dic,self.maxi_depth,self.depths_sizeordered,self.clades_size,self.clades_alltips = getdepths_sizes(self.root)
        self.nodes = [node for node in tree.get_nonterminals()]
        self.tips = [tip for tip in tree.get_terminals()]
        self.checkdupids()
        self.Total_length = gettotallength(self.tree)
        self.Total_species = gettotalspecies(self.tree)
    def checkdupids(self):
        node_ids = [node.name for node in self.nodes if node.name is not None]
        tip_ids = [tip.name for tip in self.tips if tip.name is not None]
        all_ids = node_ids + tip_ids
        assert len(all_ids) == len(set(all_ids))
    def polardraw(self,polar,fs=(6,6),topologylw=3,userfig=None,userax=None,starttheta=0,tiplabelroffset=0.1,tiplabelthetaoffset=0,showtiplabel=True,plottipnode=True,shownodelabel=True,plotnnode=True,nodelabelroffset=0.1,nodelabelthetaoffset=0,plotnodeuncertainty=False,nucalpha=0.5,nuccolor='blue',nuclw=4,userbranchcolor=None,tiplabelalign='left',nodelabelalign='left',plotfulllengthscale=True,scaleinipoint=(0,0),scaleendpoint=(180,1),scalecolor='k',scalelw=2,tiplabelsize=1.5,tiplabelalpha=1,tiplabelcolor='k',tipnodesize=3,tipnodecolor='k',tipnodealpha=1,tiplabelstyle='normal',tipnodemarker='o',nodelabelsize=1.5,nodelabelalpha=1,nodelabelcolor='k',nnodesize=3,nnodecolor='k',nnodealpha=1,nodelabelstyle='normal',nnodemarker='o',wgdlw=4,fullscalelw=None,fullscalexticks=None,fullscalecolor='k',fullscalels='--'):
        logging.info("Plotting circular tree")
        if userfig is None and userax is None:
            fig, ax = plt.subplots(1,1,figsize=fs,subplot_kw={'projection': 'polar'})
        self.fig,self.ax = fig,ax
        self.starttheta,self.endtheta = starttheta,polar
        self.drawtipspolar(polar,tiplabelroffset,tiplabelthetaoffset,plotnode=plottipnode,showlabel=showtiplabel,starttheta=starttheta,labelalign=tiplabelalign,labelsize=tiplabelsize,labelalpha=tiplabelalpha,labelcolor=tiplabelcolor,nodesize=tipnodesize,nodecolor=tipnodecolor,nodealpha=tipnodealpha,labelstyle=tiplabelstyle,nodemarker=tipnodemarker)
        self.drawnodespolar(nodelabelroffset,nodelabelthetaoffset,showlabel=shownodelabel,plotnode=plotnnode,pnuc=plotnodeuncertainty,nuca=nucalpha,nucc=nuccolor,labelalign=nodelabelalign,labelsize=nodelabelsize,labelalpha=nodelabelalpha,labelcolor=nodelabelcolor,nodesize=nnodesize,nodecolor=nnodecolor,nodealpha=nnodealpha,labelstyle=nodelabelstyle,nodemarker=nnodemarker,nuclw=nuclw)
        self.drawlinespolar(ubr=userbranchcolor,topologylw=topologylw)
        self.drawscalepolar(plotfulllengthscale=plotfulllengthscale,inipoint=scaleinipoint,endpoint=scaleendpoint,cr=scalecolor,lw=scalelw,fullscalelw=fullscalelw,fullscalexticks=fullscalexticks,fullscalecolor=fullscalecolor,fullscalels=fullscalels)
        self.scaler_theta = abs(self.endtheta-self.starttheta)/100
        thetamin,thetamax = self.starttheta-self.scaler_theta,min([360,self.endtheta+self.scaler_theta])
        if thetamax-thetamin >=360: thetamax = 360 + thetamin
        self.ax.set_thetamin(thetamin);self.ax.set_thetamax(thetamax)
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
    def basicdraw(self,fs=(6,6),topologylw=3,tiplabelxoffset=0.1,tiplabelyoffset=0,nodelabelxoffset=0.1,nodelabelyoffset=0,userfig=None,userax=None,shownodelabel=True,showtiplabel=True,plottipnode=True,plotnnode=True,userbranchcolor=None,plotnodeuncertainty=False,nucalpha=0.5,nuccolor='blue',nulw=4,plotfulllengthscale=True,scaleinipoint=(0,-0.5),scaleendpoint=(1,-0.5),scalecolor='k',scalelw=2,tiplabelsize=1.5,tiplabelalpha=1,tiplabelcolor='k',tipnodesize=6,tipnodecolor='k',tipnodealpha=1,tiplabelstyle='normal',tipnodemarker='o',nodelabelsize=1.5,nodelabelalpha=1,nodelabelcolor='k',nnodesize=6,nnodecolor='k',nnodealpha=1,nodelabelstyle='normal',nnodemarker='o',wgdlw=4,fullscalelw=None,fullscaley=-1,fullscalexticks=None,fullscalecolor='k',fullscaleticklw=None,fullscaletickcolor='k',fullscaletickheight=None,fullscaleticklabels=None,fullscaleticklabelsize=None,fullscaleticklabelcolor='k'):
        logging.info("Plotting tree")
        if userfig is None and userax is None:
            fig, ax = plt.subplots(1,1,figsize=fs)
        else:
            fig, ax = userfig, userax
        #ax.set_ylim(0,len(self.tips)+1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])  # Remove x ticks
        ax.set_yticks([])  # Remove y ticks
        ax.set_xticklabels([])  # Remove x tick labels
        ax.set_yticklabels([])  # Remove y tick labels
        self.fig,self.ax = fig,ax
        self.drawtips(tiplabelxoffset,tiplabelyoffset,plotnode=plottipnode,showlabel=showtiplabel,labelsize=tiplabelsize,labelalpha=tiplabelalpha,labelcolor=tiplabelcolor,nodesize=tipnodesize,nodecolor=tipnodecolor,nodealpha=tipnodealpha,labelstyle=tiplabelstyle,nodemarker=tipnodemarker)
        self.drawnodes(nodelabelxoffset,nodelabelyoffset,showlabel=shownodelabel,plotnode=plotnnode,pnuc=plotnodeuncertainty,nuca=nucalpha,nucc=nuccolor,nulw=nulw,labelsize=nodelabelsize,labelalpha=nodelabelalpha,labelcolor=nodelabelcolor,nodesize=nnodesize,nodecolor=nnodecolor,nodealpha=nnodealpha,labelstyle=nodelabelstyle,nodemarker=nnodemarker)
        self.drawlines(ubr=userbranchcolor,topologylw=topologylw)
        self.scaler_y = len(self.tips)/1000
        ymin,ymax = self.drawscale(plotfulllengthscale=plotfulllengthscale,inipoint=scaleinipoint,endpoint=scaleendpoint,cr=scalecolor,lw=scalelw,fullscalelw=fullscalelw,fullscaley=fullscaley,fullscalexticks=fullscalexticks,fullscalecolor=fullscalecolor,fullscaleticklw=fullscaleticklw,fullscaletickcolor=fullscaletickcolor,fullscaletickheight=fullscaletickheight,fullscaleticklabels=fullscaleticklabels,fullscaleticklabelsize=fullscaleticklabelsize,fullscaleticklabelcolor=fullscaleticklabelcolor)
        ax.set_ylim(ymin-self.scaler_y,ymax+self.scaler_y)
    def highlightnode(self,nodes=[],colors=[],nodesizes=[],nodealphas=[],nodemarkers=[]):
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
            if node not in self.allnodes_xcoordinates or node not in self.allnodes_ycoordinates:
                logging.error("Cannot find {} in the tree!".format(node))
                exit(0)
            xcoor,ycoor = self.allnodes_xcoordinates[node],self.allnodes_ycoordinates[node]
            self.ax.plot((xcoor,xcoor),(ycoor,ycoor),marker=marker,alpha=al,markersize=ns,color=cr)
    def highlightclade(self,clades=[],facecolors=[],alphas=[],lws=[],gradual=True,leftoffset=None,rightoffset=None,bottomoffset=None,topoffset=None):
        if len(clades) == 0:
            return
        if facecolors == []:
            facecolors = ['gray' for i in range(len(clades))]
        if alphas == []:
            alphas = np.full(len(clades),0.5)
        if lws == []:
            lws = np.full(len(clades),self.topologylw)
        for clade,fcr,al,lw in zip(clades,facecolors,alphas,lws):
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
                color_limits_rgba = [to_rgba(fcr, alpha=0.1),to_rgba(fcr, alpha=1)]
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
    def saveplot(self,outpath):
        self.fig.tight_layout()
        self.fig.savefig(outpath)
        plt.close()
    def drawtipspolar(self,polar,labeloffset,labelthetaoffset,plotnode=False,showlabel=True,starttheta=0,labelalign='left',labelsize=1.5,labelstyle='normal',labelalpha=1,labelcolor='k',nodesize=3,nodecolor='k',nodealpha=1,nodemarker='o'):
        self.labelsize = labelsize
        tips_rcoordinates = {tip.name:self.root.distance(tip) for tip in self.tips}
        self.polar = polar
        self.Total_theta = self.polar/180*np.pi
        self.per_sp_theta = (polar-starttheta)/180*np.pi/(len(self.tips)-1)
        thetacoordinate = starttheta/180*np.pi - self.per_sp_theta
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
            if plotnode: self.ax.plot(tips_thetacoordinates[tip.name],tips_rcoordinates[tip.name],marker=nodemarker,markersize=nodesize,color=nodecolor,alpha=nodealpha)
            if showlabel:
                angle=thetatovalue(tips_thetacoordinates[tip.name])
                if angle <= 90 or angle > 270:
                    rotation = angle
                    labelalign_ = labelalign
                else:
                    rotation = angle + 180
                    if labelalign == 'left': labelalign_ = 'right'
                    if labelalign == 'right': labelalign_ = 'left'
                text = self.ax.text(tips_thetacoordinates[tip.name]+self.Total_theta*labelthetaoffset,tips_rcoordinates[tip.name]+self.Total_length*labeloffset,tip.name,rotation=rotation,rotation_mode='anchor',va='center',ha=labelalign_,fontsize=labelsize,fontstyle=labelstyle,color=labelcolor,alpha=labelalpha)
        self.tips_thetacoordinates,self.tips_rcoordinates = tips_thetacoordinates,tips_rcoordinates
        self.allnodes_thetacoordinates,self.allnodes_rcoordinates = {**tips_thetacoordinates},{**tips_rcoordinates}
    def drawnodespolar(self,labelroffset,labelthetaoffset,showlabel=False,plotnode=False,pnuc=False,nuca=0.5,nucc='blue',labelalign='left',labelsize=1.5,labelalpha=1,labelcolor='k',nodesize=3,nodecolor='k',nodealpha=1,labelstyle='normal',nodemarker='o',nuclw=4):
        self.nodes_rcoordinates = {}
        self.nodes_thetacoordinates = {}
        self.nodesize = nodesize
        if pnuc:
            logging.info("Adding node uncertainty")
        for node in self.nodes:
            children = findfirstchildren(node)
            self.nodes_thetacoordinates[node.name] = recursiongetycor(children,self.tips_thetacoordinates)
            self.nodes_rcoordinates[node.name] = self.root.distance(node)
            if plotnode: self.ax.plot(self.nodes_thetacoordinates[node.name],self.nodes_rcoordinates[node.name],marker=nodemarker,markersize=nodesize,color=nodecolor,alpha=nodealpha)
            angle=thetatovalue(self.nodes_thetacoordinates[node.name])
            if angle <= 90 or angle > 270:
                rotation = angle
                labelalign_ = labelalign
            else:
                rotation = angle + 180
                if labelalign == 'left': labelalign_ = 'right'
                if labelalign == 'right': labelalign_ = 'left'
            if showlabel: self.ax.text(self.nodes_thetacoordinates[node.name]+self.Total_theta*labelthetaoffset,self.nodes_rcoordinates[node.name]+self.Total_length*labelroffset,node.name,ha=labelalign_,va='center',rotation_mode='anchor',rotation=rotation,fontsize=labelsize,alpha=labelalpha,color=labelcolor,fontstyle=labelstyle)
            if pnuc:
                nodeuncertainty = getnuc(node)
                if None in nodeuncertainty:
                    continue
                nodeuncertainty = -np.array(nodeuncertainty)+self.Total_length
                self.ax.plot((self.nodes_thetacoordinates[node.name],self.nodes_thetacoordinates[node.name]),(nodeuncertainty[1],nodeuncertainty[0]),lw=nuclw,color=nucc,alpha=nuca)
        self.allnodes_thetacoordinates = {**self.allnodes_thetacoordinates,**self.nodes_thetacoordinates}
        self.allnodes_rcoordinates = {**self.allnodes_rcoordinates,**self.nodes_rcoordinates}
    def drawlinespolar(self,ubr=None,rbr=False,topologylw=3):
        self.topologylw = topologylw
        if ubr is None:
            branch_colors = {**{tip.name:'k' for tip in self.tips},**{node.name:'k' for node in self.nodes}}
            if rbr:
                for key in branch_colors: branch_colors[key] = random_color_hex()
        else:
            branch_colors = getubr(ubr)
        for tip in self.tips:
            firstparent = getfirstparent(tip,self.nodes)
            segment_length = self.root.distance(tip)-self.root.distance(firstparent)
            rmin,rmax = self.root.distance(firstparent),self.root.distance(tip)
            self.ax.plot((self.tips_thetacoordinates[tip.name],self.tips_thetacoordinates[tip.name]),(rmin,rmax),color=branch_colors.get(tip.name,'k'),lw=topologylw)
            thetamin, thetamax = sorted([self.tips_thetacoordinates[tip.name],self.nodes_thetacoordinates[firstparent.name]])
            thetas = np.linspace(thetamin, thetamax, 1000)
            self.ax.plot(thetas,np.full(len(thetas),self.nodes_rcoordinates[firstparent.name]),color=branch_colors.get(tip.name,'k'),lw=topologylw)
        for node in self.nodes:
            if self.root.distance(node) == 0: continue
            firstparent = getfirstparent(node,self.nodes)
            segment_length = self.root.distance(node)-self.root.distance(firstparent)
            rmin,rmax = self.root.distance(firstparent),self.root.distance(node)
            self.ax.plot((self.nodes_thetacoordinates[node.name],self.nodes_thetacoordinates[node.name]),(rmin,rmax),color=branch_colors.get(node.name,'k'),lw=topologylw)
            thetamin,thetamax = sorted([self.nodes_thetacoordinates[node.name],self.nodes_thetacoordinates[firstparent.name]])
            thetas = np.linspace(thetamin, thetamax, 100)
            self.ax.plot(thetas,np.full(len(thetas),self.nodes_rcoordinates[firstparent.name]),color=branch_colors.get(node.name,'k'),lw=topologylw)
    def drawscalepolar(self,plotfulllengthscale=False,inipoint=(0,0),endpoint=(0,0),cr='k',lw=1.5,fullscalelw=None,fullscalexticks=None,fullscalecolor='k',fullscalels='--'):
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
            self.ax.plot((theta1,theta2),(r1,r2),color=cr,lw=lw) 
    def drawscale(self,plotfulllengthscale=True,inipoint=(0,0),endpoint=(0,0),cr='k',lw=None,fullscalelw=None,fullscaley=-1,fullscalexticks=None,fullscalecolor='k',fullscaleticklw=None,fullscaletickcolor='k',fullscaletickheight=None,fullscaleticklabels=None,fullscaleticklabelsize=None,fullscaleticklabelcolor='k'):
        Ymin,Ymax = [],[]
        if plotfulllengthscale:
            if fullscalelw is None: fullscalelw = self.topologylw
            if fullscaleticklw is None: fullscaleticklw = self.topologylw
            if fullscaleticklabelsize is None: fullscaleticklabelsize = self.labelsize
            xmin,xmax = 0,self.Total_length
            ycoordi = fullscaley
            xticks = np.linspace(xmin,xmax,6) if fullscalexticks is None else fullscalexticks
            if fullscaleticklabels is None: fullscaleticklabels = ["{:.1f}".format(float(tick)) for tick in xticks[::-1]]
            self.ax.plot((xticks[-1],xticks[0]), (ycoordi,ycoordi), color=fullscalecolor, linewidth=fullscalelw)
            y2 = fullscaletickheight if fullscaletickheight is not None else self.scaler_y
            for tick,ticklabel in zip(xticks,fullscaleticklabels):
                self.ax.plot((tick,tick), (ycoordi,ycoordi-y2), color=fullscaletickcolor, linewidth=fullscaleticklw)
                self.ax.text(tick,ycoordi-y2-self.scaler_y,ticklabel,fontsize=fullscaleticklabelsize,color=fullscaleticklabelcolor,ha='center',va='top')
            y2+=self.scaler_y
        else:
            ycoordi,y2 = 0,0
        if inipoint!=endpoint:
            if lw is None: lw = self.topologylw
            x1,y1 = inipoint
            x2,y2 = endpoint
            self.ax.plot((x1,x2),(y1,y2),color=cr,linewidth=lw)
            ymin,ymax = sorted([y1,y2])
            ymin,ymax = min([ymin,0]),max([ymax,len(self.tips)+1])
            Ymin,Ymax = min([ymin,ycoordi-y2]),max([ymax,ycoordi-y2])
            return Ymin,Ymax
        else:
            Ymin,Ymax = min([0,ycoordi-y2]),max([len(self.tips)+1,ycoordi-y2])
            return Ymin,Ymax
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
    def drawtips(self,labelxoffset,labelyoffset,plotnode=False,showlabel=True,labelsize=1.5,labelstyle='normal',labelalpha=1,labelcolor='k',nodesize=3,nodecolor='k',nodealpha=1,nodemarker='o'):
        self.labelsize = labelsize
        tips_ycoordinates = {}
        ycoordinate = 0
        tips_xcoordinates = {tip.name:self.root.distance(tip) for tip in self.tips}
        clades_sizeordered = self.depths_sizeordered[1] # first children from root
        for clades_size in clades_sizeordered:
            clade_name,size = clades_size
            clade = findcladebyname(self.tree,clade_name)
            if clade.is_terminal():
                ycoordinate+=1
                tips_ycoordinates[clade_name] = ycoordinate
            else:
                ycoordinate,tips_ycoordinates = recursionuntilalltip(clade,ycoordinate,tips_ycoordinates,self.clades_size)
        #print(tips_xcoordinates,tips_ycoordinates)
        for tip in self.tips:
            if plotnode: self.ax.plot(tips_xcoordinates[tip.name],tips_ycoordinates[tip.name],marker=nodemarker,markersize=nodesize,color=nodecolor,alpha=nodealpha)
            if showlabel:
                text = self.ax.text(tips_xcoordinates[tip.name]+self.Total_length*labelxoffset,tips_ycoordinates[tip.name]+len(self.tips)*labelyoffset,tip.name,ha='left',va='center',fontsize=labelsize,fontstyle=labelstyle,alpha=labelalpha,color=labelcolor)
        self.tips_ycoordinates,self.tips_xcoordinates = tips_ycoordinates,tips_xcoordinates
        self.allnodes_ycoordinates,self.allnodes_xcoordinates = {**tips_ycoordinates},{**tips_xcoordinates}
    def drawnodes(self,labelxoffset,labelyoffset,showlabel=False,plotnode=False,pnuc=False,nuca=0.5,nucc='blue',nulw=4,labelsize=1.5,labelalpha=1,labelcolor='k',nodesize=3,nodecolor='k',nodealpha=1,labelstyle='normal',nodemarker='o'):
        self.nodes_ycoordinates = {}
        self.nodes_xcoordinates = {}
        self.nodesize = nodesize
        if pnuc:
            logging.info("Adding node uncertainty")
        for node in self.nodes:
            children = findfirstchildren(node)
            self.nodes_ycoordinates[node.name] = recursiongetycor(children,self.tips_ycoordinates)
            self.nodes_xcoordinates[node.name] = self.root.distance(node)
            if plotnode: self.ax.plot(self.nodes_xcoordinates[node.name],self.nodes_ycoordinates[node.name],marker=nodemarker,markersize=nodesize,color=nodecolor,alpha=nodealpha)
            if showlabel: self.ax.text(self.nodes_xcoordinates[node.name]+self.Total_length*labelxoffset,self.nodes_ycoordinates[node.name]+len(self.tips)*labelyoffset,node.name,ha='left',va='center',fontsize=labelsize,alpha=labelalpha,fontstyle=labelstyle)
            if pnuc:
                nodeuncertainty = getnuc(node)
                if None in nodeuncertainty:
                    continue
                nodeuncertainty = -np.array(nodeuncertainty)+self.Total_length
                self.ax.plot((nodeuncertainty[1],nodeuncertainty[0]),(self.nodes_ycoordinates[node.name],self.nodes_ycoordinates[node.name]),lw=nulw,color=nucc,alpha=nuca)
        self.allnodes_ycoordinates = {**self.allnodes_ycoordinates,**self.nodes_ycoordinates}
        self.allnodes_xcoordinates = {**self.allnodes_xcoordinates,**self.nodes_xcoordinates}
        #print(self.nodes_xcoordinates,self.nodes_ycoordinates)
    def drawlines(self,ubr=None,rbr=False,topologylw=3):
        drawed_nodes = []
        self.topologylw = topologylw
        if ubr is None:
            branch_colors = {**{tip.name:'k' for tip in self.tips},**{node.name:'k' for node in self.nodes}}
            if rbr:
                for key in branch_colors: branch_colors[key] = random_color_hex()
        else:
            branch_colors = getubr(ubr)
        for tip in self.tips:
            firstparent = getfirstparent(tip,self.nodes)
            segment_length = self.root.distance(tip)-self.root.distance(firstparent)
            xmin,xmax = self.root.distance(firstparent),self.root.distance(tip)
            self.ax.plot((xmin,xmax),(self.tips_ycoordinates[tip.name],self.tips_ycoordinates[tip.name]),color=branch_colors.get(tip.name,'k'),linewidth=topologylw)
            ymin, ymax = sorted([self.tips_ycoordinates[tip.name],self.nodes_ycoordinates[firstparent.name]])
            self.ax.plot((self.nodes_xcoordinates[firstparent.name],self.nodes_xcoordinates[firstparent.name]),(ymin, ymax),color=branch_colors.get(tip.name,'k'),linewidth=topologylw)
        for node in self.nodes:
            if self.root.distance(node) == 0: continue
            firstparent = getfirstparent(node,self.nodes)
            segment_length = self.root.distance(node)-self.root.distance(firstparent)
            xmin,xmax = self.root.distance(firstparent),self.root.distance(node)
            self.ax.plot((xmin,xmax),(self.nodes_ycoordinates[node.name],self.nodes_ycoordinates[node.name]),color=branch_colors.get(node.name,'k'),linewidth=topologylw)
            ymin, ymax = sorted([self.nodes_ycoordinates[node.name],self.nodes_ycoordinates[firstparent.name]])
            self.ax.plot((self.nodes_xcoordinates[firstparent.name],self.nodes_xcoordinates[firstparent.name]),(ymin, ymax),color=branch_colors.get(node.name,'k'),linewidth=topologylw)
    def drawtrait(self,trait=(),offset=0.2,usedata=()):
        if trait == ():
            return
        logging.info("Adding trait")
        trait_dics = {}
        colors = cm.viridis(np.linspace(0, 1, len(trait)))
        updated_offset_bound = 0
        for ind,trait_file in enumerate(trait):
            offset += updated_offset_bound
            df = pd.read_csv(trait_file,header=0,index_col=0,sep='\t')
            if usedata!=():
                trait_dic_orig = {i:j for i,j in zip(df.index,df.loc[:,usedata[ind]])}
                traitname = usedata[ind]
            else:
                trait_dic_orig = {i:j for i,j in zip(df.index,df.iloc[:,0])}
                traitname = df.columns[0]
            trait_dics[traitname] = trait_dic_orig
            trait_dic = copy.deepcopy(trait_dic_orig)
            scaling = 0.1*self.Total_length
            max_trait = sorted(trait_dic.values())[-1]
            self.ax.text(self.Total_length*(1+offset)+0.5*scaling,len(self.tips)+1,traitname,ha='center',va='center')
            for key,value in trait_dic.items(): trait_dic[key] = value/max_trait*scaling
            left_coordi = self.Total_length*(offset+1)
            widths = []
            ys = []
            real_trait_values = list(trait_dic_orig.values())
            real_Maxi = np.max(real_trait_values)
            scaled_trait_values = list(trait_dic.values())
            scaled_Maxi = np.max(scaled_trait_values)
            for tip in self.tips:
                widths.append(trait_dic.get(tip.name,0))
                #xmin,xmax = self.tips_xcoordinates[tip.name]+self.Total_length*offset, self.tips_xcoordinates[tip.name]+self.Total_length*offset+trait_dic.get(tip.name,0)
                if trait_dic.get(tip.name,0) > updated_offset_bound:
                    updated_offset_bound = trait_dic.get(tip.name,0)
                ys.append(self.tips_ycoordinates[tip.name])
                #self.ax.plot((xmin,xmax),(self.tips_ycoordinates[tip.name],self.tips_ycoordinates[tip.name]),color=colors[ind],lw=2)
            #self.ax.plot((self.Total_length*(1+offset),self.Total_length*(1+offset)),(0,len(self.tips)),color='k',lw=2)
            self.ax.barh(ys,widths,height=0.5,left=left_coordi,align='center',color=colors[ind])
            self.ax.plot((self.Total_length*(1+offset),self.Total_length*(1+offset)),(0.5,len(self.tips)+0.5),color='k',lw=2)
            self.ax.plot((self.Total_length*(1+offset),self.Total_length*(1+offset)+scaled_Maxi),(0.5,0.5),color='k',lw=2)
            self.ax.plot((self.Total_length*(1+offset)+scaled_Maxi,self.Total_length*(1+offset)+scaled_Maxi),(0.5,0),color='k',lw=2)
            self.ax.plot((self.Total_length*(1+offset)+scaled_Maxi/2,self.Total_length*(1+offset)+scaled_Maxi/2),(0.5,0),color='k',lw=2)
            self.ax.plot((self.Total_length*(1+offset),self.Total_length*(1+offset)),(0.5,0),color='k',lw=2)
            self.ax.text(self.Total_length*(1+offset),-0.5,0,ha='center',va='top')
            self.ax.text(self.Total_length*(1+offset)+scaled_Maxi/2,-0.5,'{:.0f}'.format(real_Maxi/2),ha='center',va='top')
            self.ax.text(self.Total_length*(1+offset)+scaled_Maxi,-0.5,'{:.0f}'.format(real_Maxi),ha='center',va='top')
    def drawwgd(self,wgd=None,cr='r',al=0.6):
        if wgd is None:
            return
        logging.info("Adding WGD")
        df = pd.read_csv(wgd,header=0,index_col=None,sep='\t').drop_duplicates(subset=['WGD ID'])
        self.allnodes_ycoordinates = {**self.nodes_ycoordinates,**self.tips_ycoordinates}
        for fullsp,hcr in zip(df["Full_Species"],df["90% HCR"]):
            sps = fullsp.split(", ")
            node = self.tree.common_ancestor(*sps)
            lower,upper = [float(i)/100 for i in hcr.split('-')]
            ycoordi = self.allnodes_ycoordinates[node.name]
            self.ax.plot((self.Total_length-upper,self.Total_length-lower),(ycoordi,ycoordi),lw=4,color=cr,alpha=al)

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

def plottree(tree=None,polar=None,trait=(),usedtraitcolumns=(),wgd=None,output='Plottree.pdf'):
    if tree is None:
        Tree = stringttophylo(Test_tree)
    else:
        Tree = Phylo.read(tree,format='newick')
    TB = TreeBuilder(Tree)
    if polar is not None:
        TB.polardraw(polar,fs=(30,30),topologylw=3,userfig=None,userax=None,tiplabelroffset=0.02,tiplabelthetaoffset=0,starttheta=0,showtiplabel=True,plottipnode=False,shownodelabel=False,plotnnode=False,nodelabelroffset=0.01,nodelabelthetaoffset=0,plotnodeuncertainty=True,nucalpha=0.4,nuccolor='blue',userbranchcolor=None,tiplabelalign='left',nodelabelalign='left',plotfulllengthscale=False,scaleinipoint=(0,0),scaleendpoint=(0,0),scalecolor='r',scalelw=3,tiplabelsize=10,tiplabelalpha=1,tiplabelcolor='k',tipnodesize=6,tipnodecolor='k',tipnodealpha=1,tiplabelstyle='normal',tipnodemarker='o',nodelabelsize=10,nodelabelalpha=1,nodelabelcolor='k',nnodesize=6,nnodecolor='k',nnodealpha=1,nodelabelstyle='normal',nnodemarker='o',fullscalelw=None,fullscalexticks=None,fullscalecolor='k',fullscalels='--')
        #TB.highlightnodepolar(nodes=[('Apostasia_shenzhenica','Asparagus_setaceus')],colors=['r'],nodesizes=[],nodealphas=[],nodemarkers=[])
        #TB.highlightcladepolar(clades=[('Apostasia_shenzhenica','Asparagus_setaceus')],facecolors=[],alphas=[],lws=[],leftoffset=None,rightoffset=None,bottomoffset=None,topoffset=None)
        TB.drawwgdpolar(wgd=wgd,cr='r',al=0.6,lw=4)
    else:
        TB.basicdraw(fs=(15,60),topologylw=3,tiplabelxoffset=0.02,tiplabelyoffset=0,nodelabelxoffset=0.02,nodelabelyoffset=0,userfig=None,userax=None,shownodelabel=False,showtiplabel=True,plottipnode=False,plotnnode=False,userbranchcolor=None,plotnodeuncertainty=True,nucalpha=0.4,nuccolor='blue',plotfulllengthscale=True,scaleinipoint=(0,0),scaleendpoint=(0,0),scalecolor='k',scalelw=None,tiplabelsize=10,tiplabelalpha=1,tiplabelcolor='k',tipnodesize=6,tipnodecolor='k',tipnodealpha=1,tiplabelstyle='normal',tipnodemarker='o',nodelabelsize=10,nodelabelalpha=1,nodelabelcolor='k',nnodesize=6,nnodecolor='k',nnodealpha=1,nodelabelstyle='normal',nnodemarker='o',wgdlw=4,fullscalelw=None,fullscaley=-1,fullscalexticks=None,fullscalecolor='k',fullscaleticklw=None,fullscaletickcolor='k',fullscaletickheight=None,fullscaleticklabels=None,fullscaleticklabelsize=None,fullscaleticklabelcolor='k')
        #TB.highlightnode(nodes=[('Apostasia_shenzhenica','Asparagus_setaceus')],colors=['r'],nodesizes=[],nodealphas=[],nodemarkers=[])
        #TB.highlightclade(clades=[('Apostasia_shenzhenica','Asparagus_setaceus')],facecolors=[],alphas=[],lws=[],leftoffset=None,rightoffset=None,bottomoffset=None,topoffset=None)
        TB.drawtrait(trait=trait,offset=0.3,usedata=usedtraitcolumns)
        TB.drawwgd(wgd=wgd,cr='r',al=0.6)
    if output is not None: TB.saveplot(output)
    return TB,Tree
