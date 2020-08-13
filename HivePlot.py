import numpy as np
import pandas as pd
import time
from math import cos, sin, atan2, sqrt, pi

from matplotlib.path import Path
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import PatchCollection

import matplotlib.pyplot as plt


def _pol2xy(rho, theta):
    x = rho * np.sin(theta)
    y = rho * np.cos(theta)
    return (x, y)


def _xy2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(x, y)
    
    return rho, theta


def _bezier_point(x1, y1, x2, y2, verbose=False):
    r1, t1 = _xy2pol(x1, y1)
    r2, t2 = _xy2pol(x2, y2)

    if t1*t2 < 0 and abs(t1)+abs(t2) > pi:
        if verbose: print("foo %s %s [%s]" % (t1, t2, np.mean([t1, t2])))
        t1 = (t1+2*pi) % (2*pi)
        t2 = (t2+2*pi) % (2*pi)
        if verbose: print("bar %s %s" % (t1, t2))

    tc = np.mean([t1, t2])
    if verbose: print(tc)
    
    rc = np.mean([r1, r2])

    xc, yc = _pol2xy(rc, tc)
    if verbose: print("buz %s" % yc)
    
    if verbose: print("[%s,%s]" % (x1, y1), "[%s,%s]" % (x2, y2), "[%s,%s]" % (xc, yc))
    return [xc, yc]


def _bezierify(row):
    return pd.Series(_bezier_point(row.start_x, row.start_y, row.end_x, row.end_y))


def curvify(row):
    p = Path(
        [(row.start_x, row.start_y), (row.bezier_x, row.bezier_y), (row.end_x, row.end_y)],
        [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    )
    return p


def arrowify(row):
    arrow = FancyArrowPatch(path=row["path"],
                            arrowstyle="Simple,tail_width=0.005,head_width=0.2,head_length=0.4",
                            alpha=row.alpha,
                            color=row.edgecolor)
    return pd.Series(arrow)


######################################################################################################################

class HivePlot:
    
    def __init__(self, split,
                 uniform=True,
                 alpha=0.01,
                 edgecol="black",
                 rmin=2.,
                 rmax=12.,
                 classcolors=("#e41a1c", "#377eb8", "#4daf4a")
                 ):
        """

        :param split: boolean, controls axis splitting for intra-class links
        :param uniform: boolean (default True), indicates if axes should have same scale
        :param alpha:
        :param edgecol:
        :param rmin:
        :param rmax:
        :param classcolors:
        """
        self.split = split
        self.uniform = uniform
        
        self.AXRANGED = False
        self.AXDICT = False
        self.BUILT = False
        
        self.rmin = rmin
        self.rmax = rmax
        
        self.rdelta = rmax - rmin
        
        if alpha == "weight":
            self.alpha = None
            self.alphamin = 0.01
        else: self.alpha = alpha
        
        if edgecol == "classes":
            self.classcols = {k:col for k,col in zip(["0","1","2"],classcolors)}
            self.edgecol = None
        else: self.edgecol = edgecol
        
        # create axis plotpoints
        if self.split: 
            self.axnum = 6
            ts_a = [-2*pi/3. + pi/18. + 2*pi/3. * i for i in [0,1,2]]
            ts_b = [2*pi/3. - pi/18. - 2*pi/3. * i for i in [0,1,2]]
            ts = ts_a + ts_b
            self.axpoints = zip([_pol2xy(self.rmin, t) for t in ts], [_pol2xy(self.rmax, t) for t in ts])
        else: 
            self.axnum = 3
            ts = [-2*pi/3. + 2*pi/3. * i for i in [0,1,2]]
            self.axpoints = zip([_pol2xy(self.rmin, t) for t in ts], [_pol2xy(self.rmax, t) for t in ts])
        
        self.node_df = pd.DataFrame(columns=["id_str","adj_ax","axis_id","value","x","y","rho","theta","markdict"])
        
        if self.split:
            self.node_df.set_index(["id_str","adj_ax"], inplace=True)
        else: 
            self.node_df.drop("adj_ax",axis=1,inplace=True)
            self.node_df.set_index("id_str", inplace=True)
        
        self.edge_df = pd.DataFrame(columns=["start_id","end_id","weight","start_x","start_y","end_x","end_y",
                                             "bezier_x","bezier_y","markdict","path","arrowpatch"])
        self.edge_df.set_index(["start_id","end_id"],inplace=True)
        
        
        self.axdict = dict() # coversion from class string to ax index, ax index always 0,1,2
        if self.uniform: self.axranges = [None,None]
        else: self.axranges = [None]*3 # list of ax
        
        self.adj_dict = dict() # start_id : [ (end_id1, weight), ...]
    
    
    def set_axdict(self, class_ids_list):
        if self.AXDICT: 
            print("AxDict already configured")
            return
        
        assert len(class_ids_list) == 3
    
        self.axdict = {k: i for i,k in enumerate(class_ids_list)}
        self.AXDICT = True
    
    def set_axrange(self, axmin, axmax, class_id =None):        
        """ Set max and min range values for axis
            for uniform case called only ONCE, axdict must be set outside
            for !uniform case called FOR EACH CLASS, setting axdict in the process
            - axmin/axmax: <numbers> values of min/max axis attribute
            - class_id: <string>  class identifier, each value is assigned a separate axis (no more than 3 possible)
        """
        assert (class_id is None) == (self.uniform)
        
        if self.AXRANGED :
            print("AxRange already configured")
            return

         # uniform case
        if self.uniform or (class_id is None):
            self.axranges = [axmin,axmax]
            self.AXRANGED = True
            return
        # not uniform case
        else:
            # class_id already in axdict but axrange not set
            if class_id in self.axdict.keys():
                # only for load_df, should have check here
                ax_id = self.axdict[class_id]
            # new class_id, hence axdict incomplete
            else:
                if not len(self.axdict.keys()) < 3:
                    raise RuntimeError("Cannot have more than three classes of data")
                    return -1
                
                ax_id = len(self.axdict.keys())
                self.axdict[class_id] = ax_id
            
            self.axranges[ax_id] = (axmin,axmax)
            if (not self.uniform) and (not (None in self.axranges)): 
                self.AXRANGED = True
                self.AXDICT = True
    
    #######################################################################################################
    # BUILDERS
    ############################
    def build_bezier(self):
        if self.split:
            self.edge_df[["bezier_x","bezier_y"]] = self.edge_df.apply(_bezierify, axis=1)
        else:
            self.edge_df[["bezier_x","bezier_y"]] = self.edge_df.apply(_bezierify, axis=1)
        
    def build_curves(self):
        if self.edge_df[["bezier_x","bezier_y"]].isnull().values.any():
            raise RuntimeError("Cannot build curve without all bezier control points")
            return -1
        else:
            self.edge_df["path"] = self.edge_df.apply(curvify, axis=1)
            
    def build_arrows(self):
        if self.edge_df["path"].isnull().values.any():
            raise RuntimeError("Cannot build arrows without all curves paths")
            return -1
        else:
            self.edge_df["arrowpatch"] = self.edge_df.apply(arrowify, axis=1)            
    
    def build_graph(self, verbose = False):
        
        if verbose: 
            t0 = time.time()
            print("Building graph..." )
        self.build_bezier()
        if verbose: 
            t1 = time.time()
            print("...bezier done (%.3f s)" % (t1 - t0) )
            t0 = t1
        self.build_curves()
        if verbose: 
            t1 = time.time()
            print("...curves done (%.3f s)" % (t1 - t0) )
            t0 = t1
        self.build_arrows()
        if verbose: 
            t1 = time.time()
            print("...arrows done (%.3f s)" % (t1 - t0) )
            print("Graph done." )
        self.BUILT = True
    
    ########################################################################################################
    # NODES
    #####################
    def load_nodedf(self, input_df, class_col, value_col, logvalue = False):
        """ Use dataframe as node_df using class_col for axis definition and value_col for attribute 
            "id_str","adj_ax","axis_id","value","x","y","markdict"
            """
        self.logvalue = logvalue
        temp_df = input_df[[class_col,value_col]].copy()
        class_ids = temp_df[class_col].unique()
        if len(class_ids) != 3:
            raise RuntimeError("Class column must have exactly 3 unique values for separating data")
        
        # set AXDICT
        self.set_axdict(class_ids)
        
        temp_df['axis_id'] = temp_df[class_col].map(self.axdict)

        if not self.uniform:
            # each axis has own range and translation
            for c in class_ids:
                cmin = temp_df[temp_df[class_col]==c][value_col].min()
                cmax = temp_df[temp_df[class_col]==c][value_col].max()
                
                if cmin == cmax:
                    raise RuntimeError("Value column must have at least 2 unique values for spacing nodes")
                
                # handle logscaling
                if self.logvalue:
                    # if <0 translate
                    if cmin <=0: 
                        delta = 1 - cmin
                        cmin += delta
                        cmax += delta
                        temp_df[temp_df[class_col]==c,value_col] += delta
                    
                    temp_df[temp_df[class_col]==c,value_col] = np.log10(temp_df[temp_df[class_col]==c,value_col])
                    
                    cmin = np.log10(cmin)
                    cmax = np.log10(cmax)
                
                temp_df.loc[temp_df[class_col]==c,"v_min"] = cmin
                temp_df.loc[temp_df[class_col]==c,"v_max"] = cmax
                
                # call for each class
                self.set_axrange(cmin,cmax,c)
        else:
            # one range for all
            allmin = temp_df[value_col].min()
            allmax = temp_df[value_col].max()
            
            if allmax == allmin:
                raise RuntimeError("Value column must have at least 2 unique values for spacing nodes")
            
            if self.logvalue:
                if allmin <=0:
                    delta = 1 - allmin
                    allmin += delta
                    allmax += delta
                    temp_df[value_col] += delta
                temp_df[value_col] = np.log10(temp_df[value_col])
                
                allmin = np.log10(allmin)
                allmax = np.log10(allmax)

            temp_df['v_min'] = allmin
            temp_df['v_max'] = allmax
            
            # call once
            self.set_axrange(allmin, allmax)

        temp_df["rho"] = self.rmin + self.rdelta * (temp_df[value_col] - temp_df.v_min )/ (temp_df.v_max - temp_df.v_min)
        
        
        if self.split:
            temp_df_a = temp_df.copy()
            temp_df_a["adj_ax"] = (temp_df["axis_id"] - 1 ) %3
            temp_df_a.set_index([temp_df_a.index,"adj_ax"], inplace=True)
            
            temp_df_b = temp_df.copy()
            temp_df_b["adj_ax"] = (temp_df["axis_id"] + 1 ) %3
            temp_df_b.set_index([temp_df_b.index,"adj_ax"], inplace=True)
            
            full_df = pd.concat([temp_df_a,temp_df_b])
            full_df.sort_index(inplace=True)
            
            # rename multi-index levels
            full_df.rename_axis(index=['id_str', 'adj_ax'], inplace=True)
            
            # cast id_str to string
            idx_full = full_df.index
            full_df.index = full_df.index.set_levels([idx_full.levels[0].astype(str),idx_full.levels[1] ])
            
            # ((ax-adj)%3 -3/2.)*2
            full_df["theta"] = (- 2*pi/3.) + (full_df.axis_id + 1) * 2*pi/3. + \
                        (pi/9.)*( (full_df.axis_id - full_df.index.get_level_values('adj_ax'))% 3 - 3/2. )
        else:
            full_df = temp_df
            full_df.rename_axis(index='id_str', inplace=True)
            
            #TODO: CHECK
            full_df.index = full_df.index.astype(str)
            
            full_df["theta"] = (- 2*pi/3.) + (full_df.axis_id + 1) * 2*pi/3.
        
        full_df["x"] = full_df.rho * np.sin(full_df.theta)
        full_df["y"] = full_df.rho * np.cos(full_df.theta)
        
        full_df["markdict"] = None
        full_df.rename(columns={value_col: "value"}, inplace=True)
        self.node_df = full_df[["axis_id", "value","x","y","rho","theta","markdict"]].copy()
        assert self.AXRANGED
        assert self.AXDICT
    
    #################################################################################################
    def add_node(self, id_str, class_id, value):
        """ Adds node to Hiveplot using external data (class identifier and attribute value)
            - compute rho from min/max on axis attribute values
            - assign theta from axis id
            -- if split compute angles for demi-axis
            - compute x,y as pol2xy
            - stores XY coordinates 
            assumes axranges already set
        """
        raise NotImplementedError("Old implementation, needs refactoring")

        if (not self.AXRANGED) or (not self.AXDICT):
            raise RuntimeError("Cannot add nodes with unconfigured axis ranges or dictionary")
        
        # get axis id from class and retrieve axrange
        ax_id = self.axdict[class_id]
        axmin, axmax = self.axranges[ax_id]
        
        r = 1 + (value - axmin) / float(axmax - axmin) * 10.
        t = - 2*pi/3. + (ax_id+1) * 2*pi/3.
        
        if self.split:
            """add node twice, one for each demi-axis using <id_str>__<adjaxid> multi-index as index entry"""
            ta = t - pi/18.
            tb = t + pi/18.
            
            xa,ya = _pol2xy(r, ta)
            xb,yb = _pol2xy(r, tb)
            
            self.node_df.loc[(id_str, (ax_id - 1)%3), ] = {"axis_id":ax_id, 
                                                           "value":value, 
                                                           "x":xa, "y":ya,
                                                           "rho":r, "theta":ta,
                                                           "markdict":None}
            
            self.node_df.loc[(id_str, (ax_id + 1)%3), ] = {"axis_id":ax_id, 
                                                           "value":value, 
                                                           "x":xb, "y":yb,
                                                           "rho":r,"theta":tb,
                                                           "markdict":None }
        else:
            """ add node once """
            x,y = _pol2xy(r, t)
            
            self.node_df.loc[id_str] = {"axis_id":ax_id, 
                                        "value":value, 
                                        "x":x, "y":y,
                                        "rho":r,"theta":t,
                                        "markdict":None}        
    
    
    ########################################################################################################
    # EDGES
    ####################################
    def load_edgedf(self, input_df, startid_col, endid_col, weightcol = None, alpha = None, edgecol = None):
        
        if weightcol is None: weight = 1
        if alpha is None: alpha = self.alpha
        if edgecol is None: edgecol = self.edgecol
        
        temp_nodes = self.node_df.copy()
        temp_nodes["temp_id_str"] = temp_nodes.index.get_level_values('id_str')
        if self.split: temp_nodes["temp_adj_ax"] = temp_nodes.index.get_level_values('adj_ax')
        
        edgeplus_df = input_df.merge(temp_nodes, left_on=startid_col, right_on="temp_id_str")
        edgeplus_df.rename({"temp_id_str":"start_id", 
                            "axis_id":"start_ax", "x":"start_x", "y":"start_y",
                            "rho":"start_rho","theta":"start_theta"},axis=1,inplace=True)
        
        if weightcol is not None:
            edgeplus_df.rename({weightcol : "edgeweight"},axis=1,inplace=True)
        else:
            edgeplus_df['edgeweight'] = weight
        
        if self.split: edgeplus_df.rename({"temp_adj_ax": "start_adj_axis"}, axis=1, inplace=True)
        edgeplus_df.drop(["value","markdict"],axis=1,inplace=True)
        
        edgeplus_df = edgeplus_df.merge(temp_nodes, left_on=endid_col, right_on="temp_id_str")
        edgeplus_df.rename({"temp_id_str":"end_id", 
                            "axis_id":"end_ax", "x":"end_x", "y":"end_y",
                            "rho":"end_rho","theta":"end_theta"},axis=1,inplace=True)
        if self.split: edgeplus_df.rename({"temp_adj_ax": "end_adj_axis"}, axis=1, inplace=True)
        edgeplus_df.drop(["value","markdict"],axis=1,inplace=True)
        
        
        # boolean index for edges between nodes on same axis
        sameax_idx = edgeplus_df.start_ax == edgeplus_df.end_ax
        
        # splitted, need to handle demi-axes
        if self.split:
            
            # handel sameax
            temp_same = edgeplus_df.loc[sameax_idx].copy()
            temp_same = temp_same[(temp_same.start_adj_axis == (temp_same.start_ax-1)%3 ) & \
                                 (temp_same.end_adj_axis == (temp_same.end_ax+1)%3 )]
            
            # handle diff ax
            temp_diff = edgeplus_df.loc[~sameax_idx].copy()
            temp_diff = temp_diff[(temp_diff.end_adj_axis == temp_diff.start_ax) & \
                                  (temp_diff.start_adj_axis == temp_diff.end_ax)]
            
            edgeplus_df = pd.concat([temp_same,temp_diff])
        # merged
        else:
            # enrich edge_df
            if any(sameax_idx):
                print("Warning, ignoring same-class links since unsplitted Hiveplot")
            edgeplus_df.drop(edgeplus_df[sameax_idx].index,inplace=True)

        edgeplus_df.set_index([startid_col,endid_col], inplace=True)
                        
        if edgecol is None:
            edgeplus_df["edgecolor"] = edgeplus_df.start_ax.astype(str).map(self.classcols)
        else:
            edgeplus_df["edgecolor"] = edgecol

        if alpha is None:
            edgeplus_df["alpha"] = np.minimum(1,edgeplus_df.edgeweight * self.alphamin)
        else:
            edgeplus_df["alpha"] = self.alpha
            
            
        edgeplus_df["bezier_x"] = None
        edgeplus_df["bezier_y"] = None

        self.edge_df = edgeplus_df.copy()

    ####################################################################################################################
    def add_link(self, start_id, end_id, weight=None, update_adjlist=False, bezier= False, alpha = None, edgecol = None):
        """ Adds link between "start" and "end" nodes, storing weight if necessary
        """
        raise NotImplementedError("Old implementation, needs refactoring")
        try:
            assert type(start_id) == str
            assert type(end_id) == str
        except:
            raise RuntimeError("Edge nodes IDs must be <string> type (%s, %s)" % (type(start_id),type(end_id)))
            return -1
        
        if update_adjlist: self.adj_dict[start_id].append((end_id,w))
            
        if weight is None: weight = 1
        if alpha is None: aplha = self.alpha
        if edgecol is None: edgecol = self.edgecol
        
        if self.split:
            s_subax = self.node_df.loc[end_id]["axis_id"].iloc[0]
            e_subax = self.node_df.loc[start_id]["axis_id"].iloc[0]

            # handle same-class links
            if s_subax == e_subax:
                s_subax = (e_subax - 1) % 3
                e_subax = (s_subax + 2) % 3

            sx, sy = self.node_df.loc[(start_id, s_subax),["x","y"]]
            ex, ey = self.node_df.loc[(end_id, e_subax),["x","y"]]

        else:
            if self.node_df.loc[start_id,"axis_id"] == self.node_df.loc[end_id,"axis_id"]: 
                print("WARNING: suppressed same-axis link since not in split mode")
                return -1
            sx, sy = self.node_df.loc[start_id,["x","y"]]
            ex, ey = self.node_df.loc[end_id,["x","y"]]
        
        if bezier: cx, cy = _bezier_point(sx, sy, ex, ey)
        else: cx, cy = None, None
        
        line = {"weight" : 1,
         "start_x": sx,"start_y" :sy,
         "end_x" :ex,"end_y":ey,
         "bezier_x" : cx,"bezier_y" :cy,
         "markdict" : {"color": edgecol, "alpha": alpha},"path" : None, "arrowpatch": None}
        
        #edge_idx = start_id + "__" + end_id
        self.edge_df.loc[(start_id,end_id),] = line

    #############################################################################################
    # MARKERS
    ###############
    def mark_node(self, id_str, markdict):
        if self.split:
            self.node_df.loc[(id_str,),"markdict"] = [markdict]*2
        else:
            self.node_df.loc[id_str,"markdict"] = markdict
    
    def mark_link_in(self, id_str, markdict):
        raise NotImplementedError
        self.edge_df.loc[id_str,"markdict"] = markdict
        self.BUILT = False
        
    def mark_link_out(self, id_str, markdict):
        raise NotImplementedError
        self.edge_df.loc[id_str,"markdict"] = markdict
        self.BUILT = False
    
    ###############################################################################################
    def plot(self, reverse=False,verbose=True, background = "white"):
        if not self.BUILT:
            self.build_graph(verbose)
        
        self.reverse = reverse
        self.fig, self.ax = plt.subplots(facecolor=background)
        
        for a in self.axpoints:
            self.ax.plot(*zip(a[0],a[1]),color="grey",lw=2)
        
        self.patch_coll = PatchCollection(self.edge_df["arrowpatch"].values, match_original=True)
        #return self.patch_coll
        self.ax.add_collection(self.patch_coll)        
        
        labelpos = [{"rotation_mode":"default","ha":"center","va":"top", "weight":"bold"} , 
                    {"rotation_mode":"default","ha":"center","va":"center", "weight":"bold"}, 
                    {"rotation_mode":"default","ha":"center","va":"center", "weight":"bold"} ]
        mintickpos = [{"rotation_mode":"default","ha":"center","va":"top"} , 
                    {"rotation_mode":"default","ha":"right","va":"bottom"}, 
                    {"rotation_mode":"default","ha":"left","va":"bottom"} ]
        maxtickpos = [{"rotation_mode":"default","ha":"center","va":"center"} , 
                    {"rotation_mode":"default","ha":"center","va":"center"}, 
                    {"rotation_mode":"default","ha":"center","va":"center"} ]
        
        nodemarkpos = [{"rotation_mode":"default","ha":"left","va":"center"} , 
                    {"rotation_mode":"default","ha":"right","va":"top"}, 
                    {"rotation_mode":"default","ha":"right","va":"bottom"} ]
        
        
        nodemarkfunc = [lambda a,b: (a+0.5,b),
                       lambda a,b: (a-0.5,b-0.5),
                       lambda a,b: (a-0.5,b+0.5)]
        
        nodemarkfunc_r = [lambda a,b: (a-0.5,b),
                       lambda a,b: (a+0.5,b+0.5),
                       lambda a,b: (a+0.5,b-0.5)]
        
        if background == "black": textcol = "white"
        else: textcol = "black"
        
        for k,i in self.axdict.items():
            
            r_lab = self.rmax + 2
            t = (- 2*pi/3.) + ((i + 1) * 2*pi/3.)
            x,y = _pol2xy(r_lab, t)
            text_rot = -( np.rad2deg(t))
            
            # labels
            self.ax.text(x,y,k,color = textcol,rotation=text_rot,**labelpos[i])
            
            # ticks
            if self.uniform: 
                tickmin = self.axranges[0]
                tickmax = self.axranges[1]
            else:
                tickmin = self.axranges[i][0]
                tickmax = self.axranges[i][1]
                
            xmin,ymin = _pol2xy(self.rmin - 0.5, t)
            xmax,ymax = _pol2xy(self.rmax + 0.7, t)
            
            if self.logvalue:
                tickmin = 10**tickmin
                tickmax = 10**tickmax
            
            self.ax.text(xmin,ymin,int(tickmin),color = textcol, rotation=text_rot,**mintickpos[i])
            self.ax.text(xmax,ymax,int(tickmax),color = textcol, rotation=text_rot,**maxtickpos[i])
            
        # marked nodes
        if not all(self.node_df.markdict.isna()):
            
            nodemarks_idx = self.node_df.loc[~self.node_df.markdict.isna()].index
            
            if self.split:
                self.node_df["adj_ax"] = self.node_df.index.get_level_values('adj_ax')
                textnodemarks_idx = self.node_df.loc[~(self.node_df.markdict.isna()) & \
                                                 (self.node_df.adj_ax == (self.node_df.axis_id + 1)%3)].index
            
            for idx in nodemarks_idx:
                x,y = self.node_df.loc[idx,["x","y"]]
                
                
                self.ax.scatter([x],[y], marker="o", facecolors=textcol, edgecolors=background, zorder=10)
                
                i = self.node_df.loc[idx,"axis_id"]
                
                if self.reverse: x,y = nodemarkfunc_r[i](x,y)
                else: x,y = nodemarkfunc[i](x,y)
                
                markdict = self.node_df.loc[idx,"markdict"]

                if self.split:
                    if idx in textnodemarks_idx: self.ax.text(x,y,markdict['text'], color=background, bbox={'facecolor':textcol, 'alpha':0.5, 'pad':2},**nodemarkpos[i])
                else:
                    self.ax.text(x,y,markdict['text'], color=background, bbox={'facecolor':textcol, 'alpha':0.5, 'pad':2},**nodemarkpos[i])
        
        self.ax.set_xlim(-self.rmax,self.rmax)
        self.ax.set_ylim(-self.rmax,self.rmax)
        self.fig.set_figheight(7)
        self.fig.set_figwidth(7)
        
        if self.reverse:
            self.ax.invert_yaxis()
            self.ax.invert_xaxis()

        
        plt.axis('off')
        plt.show()
        #return self.fig

