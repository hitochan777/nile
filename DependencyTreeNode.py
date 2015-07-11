#!/usr/bin/env python
# otsuki@nlp (Otsuki Hitoshi)

import sys
import weakref
from TerminalNode import TerminalNode
from NLPTree import NLPTree
from Tree import Tree

class DependencyTreeNode(NLPTree):
    def __init__(self, data = None, children = None):
        self.setup(data,children)

    def setup(self, data, children = None):
        self.partialAlignments = []
        self.partialAlignments_hope = []
        self.partialAlignments_fear = []
        self.data = data
        self.parent = None
        self.children = []
        if children is not None:
            self.children = children
        for ci, child in enumerate(self.children):
            child.parent = weakref.ref(self)
            child.order = ci
        self.terminals = [ ]
        self.eIndex = -1
        self.hope = None
        self.oracle = None
        self.fear = None
        self.i = -1
        self.j = -1
        self.order = 0
        self.span = None
        self.nodeList = None # list including weakref to each node; sorted by id
	
    def setTerminals(self):
        if len(self.children) > 0:
            for child in self.children:
                self.terminals += child.setTerminals()
        else:
            self.terminals = [weakref.ref(self)]
        return self.terminals

    def getAllNodes(self):
        return self.nodeList

    def getNodeByIndex(self, i):
        """
        Return node with index i
        Store only weak references to node
        """
        return self.nodeList[i]

    def getTreeTerminals(self):
        """
        Iterator over terminals.
        """
        # print len(self.terminals)
        if len(self.terminals)==0:
            self.setTerminals()
        for t in self.terminals:
            yield t()

    def span_start(self):
        return self.i
        # if(self.children):
        #     return self.children[0].span_start()
        # if(len(self.children) == 0):
        #     return self.eIndex

    def span_end(self):
        return self.j-1
        # if(self.children):
        #     return self.children[-1].span_end()
        # if(len(self.children) == 0):
        #     return self.eIndex
