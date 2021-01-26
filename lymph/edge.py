import numpy as np
import scipy as sp 
import scipy.stats

from .node import Node

class Edge(object):
    """Class for the connections between lymph node levels (LNLs) represented
    by the Node class.

    Args:
        start: Parent node

        end: Child node

        t: Transition probability in case start-Node has state 1 (microscopic 
            involvement).
    """
    def __init__(self, 
                 start: Node, 
                 end: Node, 
                 t:float = 0.):
        if type(start) is not Node:
            raise TypeError("Start must be instance of Node")
        if type(end) is not Node:
            raise TypeError("End must be instance of Node!")

        self.start = start 
        self.start.out.append(self)
        self.end = end 
        self.end.inc.append(self)
        self.t = t



    def report(self):
        """Just quickly prints infos about the edge
        """
        print("start: {}".format(self.start.name))
        print("end: {}".format(self.end.name))
        print("t = {}".format(self.t))