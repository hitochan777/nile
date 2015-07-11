import re
import sys
import weakref
# http://stackoverflow.com/questions/9908013/weak-references-in-python

from NLPTree import NLPTree
from TerminalNode import TerminalNode
from DependencyTreeNode import DependencyTreeNode

def readDependencyFile(filename):
    dep=""
    with open(filename,"r") as f:
        for line in f:
            if line.startswith("#"):
                if dep!="":
                    yield dep
                    dep=""
                continue
            if line.startswith("\n"):
                continue
            dep += line
        if dep!="":
            yield dep

def stringToDependencyTreeWeakRef(string):
    """
    Read dependency parser output style tree string and return the tree that the string encodes. 
    """
    # Reset current class members
    eIndex = -1
    rootId = -1  
    infos = [word_infos.split() for word_infos in string.strip().split("\n")]
    # print(infos)
    strlen = len(infos)
    nodeList = [DependencyTreeNode() for i in range(0,strlen)]
    for info in infos: 
        id = int(info[0])
        dependency_id = int(info[1])
        surface = info[2] 
        dict_form = info[3].split("/")[0]
        if "/" in info[3]:
            pronunciation = info[3].split("/")[1]
        else:
            pronunciation = None
        if ":" in info[4]:
            pos2 = info[4].split(":")[1] # pos in detail
        else:
            pos2 = None
        pos = info[4].split(":")[0]
        isContent = info[5]=="1"
        nodeList[id].eIndex = id # because eIndex and data["id"] is identica we can remove either, but this might lead to runtime error so for now I keep both of them
        nodeList[id].data = {"id":id, "dep_id":dependency_id, "surface":surface, "dict_form": dict_form, "pronunciation": pronunciation, "isContent":isContent, "pos":pos, "pos2":pos2}
        if dependency_id >= 0:
            nodeList[id].parent = weakref.ref(nodeList[dependency_id])
            nodeList[dependency_id].addChild(nodeList[id])
        else:
            if rootId != -1:
                sys.exit("root appeared more than once!!")
            rootId = id
            nodeList[id].parent = None
    if rootId == -1:
        sys.exit("root didn't appeare!")
    addSpans(nodeList[rootId])
    nodeList[rootId].nodeList = nodeList
    return nodeList[rootId]
    
def _addSpans(tree):
  i = 0
  for node in tree.bottomup():
    if len(node.children) == 0:
      node.i = i
      node.j = i+1
      i += 1
    else:
      node.i = node.children[0].i
      node.j = node.children[-1].j

  return tree

def addSpans(tree):
    for node in tree.bottomup():
        if len(node.children) == 0:
            node.i = node.data["id"]
            node.j = node.i + 1
        else:
            node.i = min(node.children[0].i,node.data["id"])
            node.j = max(node.children[-1].j,node.data["id"]+1)
    return tree

def containsSpan(currentNode, fspan):
    """
    Does span of node currentNode wholly contain span fspan?
    """
    span = currentNode.get_span()
    return span[0] <= fspan[0] and span[1] >= fspan[1]

if __name__ == "__main__":
    fname=sys.argv[1]
    y = readDependencyFile(fname)
    for d in y:
        tree = stringToDependencyTreeWeakRef(d)
        # print(tree)
        for node in tree.bottomup():
            print(node.data["surface"],node.i, node.j)
            # print(node.setTerminals())
        # print(d.split("\n"))
