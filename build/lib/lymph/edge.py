import numpy as np
import scipy as sp 
import scipy.stats

from .node import Node

class Edge(object):
    """Class for the connections between lymph node levels (LNLs) represented
    by the Node class.

    Args:
        start (Node instance): Parent node

        end (Node instance): Child node

        t (float): Transition probability in case start-Node has state 1 
            (microscopic involvement).
    """
    def __init__(self, start, end, t=None, narity=3):
        if type(start) is not Node:
            raise Exception("Start must be Node!")
        if type(end) is not Node:
            raise Exception("End must be Node!")

        self.start = start 
        self.start.out.append(self)
        self.end = end 
        self.end.inc.append(self)

        if t is None:
            self.t = np.zeros(shape=(narity,))
        else:
            self.t = t
            self.t[0] = 0
        for i in range(1, len(self.t)):
            if self.t[i] < self.t[i-1]:
                raise Exception("t cannot decrease!")



    def report(self):
        """Just quickly prints infos about the edge"""
        print("start: {}".format(self.start.name))
        print("end: {}".format(self.end.name))
        print("t = {}".format(self.t))