import os,sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import ROOT as root



## Read file and tree
dirname='/Users/ekargian/Documents/g-2/Trackers/ML_tracking/MC/'
rootfilename='gm2tracker_particle_gun_ana_newprod.root'
f = root.TFile(dirname+rootfilename)
tree=f.Get("MCNtuple/MCNtuple")

nEntries=tree.GetEntries()
branches=tree.GetListOfBranches()

## data structure
data={}
for branch in branches:
    data[branch.GetName()]=[]

## Iterate over entries
for i in range(nEntries):
    tree.GetEntry(i)
    if getattr(tree, 'hasTrack') and\
       getattr(tree, 'hasDecayVertex') and\
       getattr(tree, 'pValue')>0:
        for branch in branches:
            data[branch.GetName()] += [ getattr(tree, branch.GetName()) ]

outfilename = dirname+'treedata.pkl'
pickle.dump(data,open(outfilename,'wb'))
