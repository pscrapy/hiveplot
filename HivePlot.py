import numpy as np
import pandas as pd
from math import cos, sin, atan2, sqrt, pi

import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def pol2xy(rho, theta):
    x = rho * np.sin(theta)
    y = rho * np.cos(theta)
    return (x,y)
    
def xy2pol(x,y):
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2( x ,y)
    return rho, theta

def bezier_point(x1,y1,x2,y2,verbose=False):
    r1,t1 = xy2pol(x1,y1)
    r2,t2 = xy2pol(x2,y2)        

    if t1*t2 <0 and abs(t1)+abs(t2)>pi: 
        if verbose: print("foo %s %s [%s]" % (t1,t2, np.mean(t1,t2)) )
        t1= (t1+2*pi) % (2*pi)
        t2= (t2+2*pi) % (2*pi)
        if verbose: print("bar %s %s" % (t1,t2))

    tc = np.mean(t1,t2)
    if verbose: print(tc)
    
    rc = np.mean(r1,r2)

    xc,yc = pol2xy(rc,tc)
    if verbose: print("buz %s" %yc)
    
    
    if verbose: print("[%s,%s]" %(x1,y1),"[%s,%s]"%(x2,y2),"[%s,%s]"%(xc,yc) )
    return [xc,yc]

def coordify(row):
    pass
    
    

def bezierify(row):
    return pd.Series(bezier_point(row.start_x, row.start_y, row.end_x, row.end_y))

def curvify(row):
    p = Path(
        [(row.start_x, row.start_y), (row.bezier_x, row.bezier_y), (row.end_x, row.end_y)],
        [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    )
    return p

def arrowify(row):
    arrow = mpatches.FancyArrowPatch(path=row["path"],
                                     arrowstyle="Simple,tail_width=0.005,head_width=0.2,head_length=0.4",
                                     **row.markdict)
    return pd.Series(arrow)


######################################################################################################################

class HivePlot:
    
    def __init__(self, split):
        self.split = split
        
        self.AXRANGED = False
        self.BUILT = False
        
        if self.split: 
            self.axnum = 6
            ts_a = [-2*pi/3. + pi/18. + 2*pi/3. * i for i in [0,1,2]]
            ts_b = [2*pi/3. - pi/18. - 2*pi/3. * i for i in [0,1,2]]
            ts = ts_a + ts_b
            self.axpoints = zip([pol2xy(1,t) for t in ts],[pol2xy(11,t) for t in ts] )
        else: 
            self.axnum = 3
            ts = [-2*pi/3. + 2*pi/3. * i for i in [0,1,2]]
            self.axpoints = zip([pol2xy(1,t) for t in ts],[pol2xy(11,t) for t in ts] )
        
        self.node_df = pd.DataFrame(columns=["id_str","adj_ax","axis_id","value","x","y","markdict"])
        
        if self.split:
            self.node_df.set_index(["id_str","adj_ax"], inplace=True)
        else: 
            self.node_df.drop("adj_ax",axis=1,inplace=True)
            self.node_df.set_index("id_str", inplace=True)
        
        self.edge_df = pd.DataFrame(columns=["start_id","end_id","weight","start_x","start_y","end_x","end_y",
                                             "bezier_x","bezier_y","markdict","path","arrowpatch"])
        self.edge_df.set_index(["start_id","end_id"],inplace=True)
        
        
        self.axdict = dict() # coversion from class string to ax index, ax index always 0,1,2
        self.axranges = [(None,None)]*3
        
        self.adj_dict = dict() # start_id : [ (end_id1, weight), ...]
        
    
    def set_axrange(self, class_id, axmin, axmax):        
        """ Set max and min range values for axis
            - class_id: <string>  class identifier, each value is assigned a separate axis (no more than 3 possible)
            - axmin/axmax: <numbers> values of min/max axis attribute
        """
        if self.AXRANGED : 
            print("Already configured")
            return
        
        if class_id in self.axdict.keys():
            ax_id = self.axdict[class_id]
        else:
            assert len(self.axdict.keys()) < 3
            ax_id = len(self.axdict.keys())
            self.axdict[class_id] = ax_id
        
        self.axranges[ax_id] = (axmin,axmax)
        if len(self.axdict.keys()) == 3: self.AXRANGED = True
    
    
    def add_node(self, id_str, class_id, value):
        """ Adds node to Hiveplot using external data (class identifier and attribute value)
            - compute rho from min/max on axis attribute values
            - assign theta from axis id
            -- if split compute angles for demi-axis
            - compute x,y as pol2xy
            - stores XY coordinates 
            assumes axranges already set
        """
        
        if not self.AXRANGED:
            raise RuntimeError("Cannot add nodes with unconfigured axis ranges")
        
        # get axis id from class and retrieve axrange
        ax_id = self.axdict[class_id]
        axmin, axmax = self.axranges[ax_id]
        
        r = 1 + (value - axmin) / float(axmax - axmin) * 10.
        t = - 2*pi/3. + (ax_id+1) * 2*pi/3.
        
        if self.split:
            """add node twice, one for each demi-axis using <id_str>__<adjaxid> as index entry"""
            ta = t - pi/18.
            tb = t + pi/18.
            
            xa,ya = pol2xy(r,ta)
            xb,yb = pol2xy(r,tb)
            
            #row_id_a = id_str + "__" + str((ax_id - 1)%3)
            #self.node_df.loc[row_id_a] = {"axis_id":ax_id, "value":value, "x":xa,"y":ya,"markdict":None}
            self.node_df.loc[(id_str, (ax_id - 1)%3), ] = {"axis_id":ax_id, "value":value, "x":xa,"y":ya,"markdict":None}
            
            #row_id_b = id_str + "__" + str((ax_id + 1)%3)                                         
            #self.node_df.loc[row_id_b] = {"axis_id":ax_id, "value":value, "x":xb,"y":yb,"markdict":None }
            self.node_df.loc[(id_str, (ax_id + 1)%3), ] = {"axis_id":ax_id, "value":value, "x":xb,"y":yb,"markdict":None }
        else:
            """ add node once """
            x,y = pol2xy(r,t)
            
            self.node_df.loc[id_str] = {"axis_id":ax_id, "value":value, "x":x,"y":y,"markdict":None}
    
    def build_nodexy(self):
        """ called on node_df without x,y"""
        if self.split:
            pass
        else:
            #def coordifier(row):
            self.node_df[["x","y"]] = self.node_df.apply(coordify,axis=1)
    
    
    def add_link(self, start_id, end_id, weight=None, update_adjlist=False, bezier= False):
        """ Adds link between "start" and "end" nodes, storing weight if necessary
        """
        try:
            assert type(start_id) == str
            assert type(end_id) == str
        except:
            raise RuntimeError("Edge nodes IDs must be <string> type (%s, %s)" % (type(start_id),type(end_id)))
            return -1
        
        if update_adjlist: self.adj_dict[start_id].append((end_id,w))
            
        if weight is None: weight = 1
        
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
        
        if bezier: cx, cy = bezier_point(sx,sy,ex,ey)
        else: cx, cy = None, None
        
        line = {"weight" : 1,
         "start_x": sx,"start_y" :sy,
         "end_x" :ex,"end_y":ey,
         "bezier_x" : cx,"bezier_y" :cy,
         "markdict" : {"color":"black", "alpha":0.2},"path" : None, "arrowpatch": None}
        
        #edge_idx = start_id + "__" + end_id
        self.edge_df.loc[(start_id,end_id),] = line
        
        
    def build_bezier(self):
        self.edge_df[["bezier_x","bezier_y"]] = self.edge_df.apply(bezierify, axis=1)
    
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
            print("Building graph...")
        self.build_bezier()
        if verbose: 
            t1 = time.time()
            print("...bezier done (%.3f)" % (t1 - t0))
            t0 = t1
        self.build_curves()
        if verbose: 
            t1 = time.time()
            print("...curves done (%.3f)" % (t1 - t0))
            t0 = t1
        self.build_arrows()
        if verbose: 
            t1 = time.time()
            print("...arrows done (%.3f)" % (t1 - t0))
            print("Graph done.")
        self.BUILT = True
            
    def add_df(self, node_df, column):
        
        #self.set_axrange()
        
        pass
    
    def add_adj(self, adjlist, weight = False):
        pass
    
    def mark_node(self, id_str, markdict):
        self.node_df.loc[id_str,"markdict"] = markdict
    
    def mark_link(self, id_str, markdict):
        pass
    
    def plot(self, reverse=False):
        if not self.BUILT:
            self.build_graph()
        
        self.reverse = reverse
        self.fig, self.ax = plt.subplots()
        
        for a in self.axpoints:
            self.ax.plot(*zip(a[0],a[1]),color="black",lw=1)
        
        self.patch_coll = PatchCollection(self.edge_df["arrowpatch"].values, match_original=True)
        #return self.patch_coll
        self.ax.add_collection(self.patch_coll)
        
        if self.reverse:
            self.ax.invert_yaxis()
            self.ax.invert_xaxis()
        
        self.ax.set_xlim(-11,11)
        self.ax.set_ylim(-9,13)
        self.fig.set_figheight(7)
        self.fig.set_figwidth(7)
        
        plt.axis('off')
        plt.show()
        #return self.fig
    