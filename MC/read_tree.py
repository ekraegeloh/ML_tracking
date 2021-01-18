import os,sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import ROOT as root


## Arg parser
import argparse
parser = argparse.ArgumentParser(description=__doc__, epilog=' ', formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('rootfile', type=str, nargs='+', help='ROOT files to be processed')
args = parser.parse_args()

rootfile_path = args.rootfile[0]
#n_files = len(infiles)


def _convertToPython(value):
    """ 
    convert value from ROOT tree to python friendly type 
    """ 
    problem_types = [ 
        '<type \'ROOT.PyIntBuffer\'>', 
        '<type \'ROOT.PyFloatBuffer\'>' ,
        '<type \'ROOT.PyBoolBuffer\'>' 
    ] 
    value_type = type(value) 
    return_value = value 
    if str(value_type) in problem_types: 
        return_value = list(value) 
    return return_value


## Read file and tree
f = root.TFile(rootfile_path)
if ('MC/' in rootfile_path) or ('MCntuple' in rootfile_path):
    ## MC data 
    tree=f.Get("MCNtuple/MCNtuple")
    hasTrack='hasTrack'
    hasDecayVertex='hasDecayVertex'
    pValue='pValue'
else:
    ## Real tracker data
    tree=f.Get("trackAndTrackCalo/tree")
    hasTrack='passTrackQuality'
    hasDecayVertex='passDecayVertexQuality'
    pValue='trackPValue'

nEntries=tree.GetEntries()
branches=tree.GetListOfBranches()

## data structure
data={}
for branch in branches:
    data[branch.GetName()]=[]
    print 'Found branch : ' , branch.GetName()

## Iterate over entries
for i in range(nEntries):
    tree.GetEntry(i)
    if getattr(tree, hasTrack) and\
       getattr(tree, hasDecayVertex) and\
       getattr(tree, pValue)>0:
        for branch in branches:
            value = _convertToPython( getattr(tree, branch.GetName()) )
            data[branch.GetName()] += [ value ]

'''
for key in data.keys():
    print key, data[key]
'''

indir, filename = os.path.split(rootfile_path)
outfilename = filename.replace('.root', '.pkl')
pickle.dump(data,open(outfilename,'wb')) ## output in current dir
