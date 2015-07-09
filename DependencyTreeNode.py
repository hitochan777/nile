#!/usr/bin/env python
# otsuki@nlp (Otsuki Hitoshi)

import sys
import weakref
from TerminalNode import TerminalNode

class DependencyTreeNode(TerminalNode):
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
        self.oracle = None

    def setTerminals(self):
        if len(self.children) > 0:
            for child in self.children:
                self.terminals += child.setTerminals()
        else:
            self.terminals = [weakref.ref(self)]
        return self.terminals

    def getTerminal(self, i):
        """
        Return terminal with index i
        Store only weak references to terminals
        """
        if len(self.terminals)==0:
            self.setTerminals()
        return self.terminals[i]()

    def getTerminals(self):
        """
        Iterator over terminals.
        """
        print len(self.terminals)
        if len(self.terminals)==0:
            self.setTerminals()
        for t in self.terminals:
            yield t()

    def span_start(self):
        if(self.children):
            return self.children[0].span_start()
        if(len(self.children) == 0):
            return self.eIndex

    def span_end(self):
        if(self.children):
            return self.children[-1].span_end()
        if(len(self.children) == 0):
            return self.eIndex
