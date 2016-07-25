###Make a false sentence using a true sentence through querying a tensor KB
#!/usr/bin/env python

import pickle
from collections import defaultdict as ddict
import numpy as np
import ast
import nltk
import sys
import os
from nltk.stem.wordnet import WordNetLemmatizer

k = int(sys.argv[1])
input_file = "data/sample-positive-sentences.txt"
tensor_truetriples = "data/animals-tensor-july10-yesonly-wo-thing.csv"
out_file = "data/animals-july10-schemadicts.bin"
candidatesfilename = "data/candidatesentences.txt"


class Dictionary(object):
    def __init__(self):
        self.strings = dict()
        self.array = [-1]
        self.current_index = 1

    def getIndex(self, string, _force_add=False):
        if not _force_add:
            index = self.strings.get(string, None)
            if index:
                return index
        self.strings[string] = self.current_index
        self.array.append(string)
        self.current_index += 1
        return self.current_index

    def getString(self, index):
        return self.array[index]

    def getAllStrings(self):
        return self.array[1:]

def getTriplesFromFile(filename, entityDict, relDict):
    triples = []
    for line in open(filename):
        fields = line.split(',')
        source = fields[0]
        relation = fields[1]
        target = fields[2]
        sourceIndex = entityDict.getIndex(source)
        targetIndex = entityDict.getIndex(target)
        relationIndex = relDict.getIndex(relation)
        triple = (sourceIndex, relationIndex, targetIndex)
        triples.append(triple)

    return triples



def makeTypedict(filenameall,entityDict):
    #Note that this depends on form of the given tensor and where each column is.
    targets = []
    sources = []
    typelists = ddict(list)
    typedict = ddict(list)
    entitytypedict = ddict(list)
    schematriples=[]
    for line in open(filenameall):
        fields = line.split(',')
        source = fields[0]
        relation= fields[1]
        target = fields[2]
        stype = fields[8]
        ttype = fields[9].replace('\n', '')
        codeditem = entityDict.getIndex(source)
        schematriple = (stype, relation, ttype)
        if schematriple not in schematriples:
           schematriples.append(schematriple)
        if stype not in typelists:
            typelists.update({stype:codeditem})
        if codeditem not in typedict[stype]:
            typedict[stype].append(codeditem)
            entitytypedict[codeditem]= stype

        codeditem = entityDict.getIndex(target)
        if ttype not in typelists:
            typelists.update({ttype:codeditem})
        if codeditem not in typedict[ttype]:
            typedict[ttype].append(codeditem)
            entitytypedict[codeditem] = ttype
        if source not in sources:
            sources.append(source)
        if target not in targets:
            targets.append(target)
    return typedict , entitytypedict, schematriples, sources, targets

def Findreplacement(location1, location2, words, typedict, entitytypedict,entityDict, relDict, allTriples, inputsentence):
    codeditem1 = entityDict.getIndex(words[location1])
    type1 = entitytypedict[codeditem1]
    codeditem2 = entityDict.getIndex(words[location2])
    type2 = entitytypedict[codeditem2]


    #get predicatelist
    predicatelist = []
    for relationIndex in range(len(relDict.getAllStrings())):
        if (codeditem1, relationIndex , codeditem2) in allTriples:
            predicatelist.append(relationIndex)

    replacementlist_firstitem=[]
    replacementlist_seconditem=[]
    flag = 0
    for candidateitem in typedict[type1]:
        for predicate in predicatelist:
            if (candidateitem, predicate, codeditem2) in allTriples:
                flag=1
                break
        if not flag:
            replacementlist_firstitem.append(candidateitem)
            replacementlist_seconditem.append(codeditem2)

        else:
            flag = 0
    flag = 0
    for candidateitem in typedict[type2]:
        for predicate in predicatelist:
            if (codeditem1, predicate, candidateitem) in allTriples:
                flag = 1
                break
        if not flag:
            replacementlist_firstitem.append(codeditem1)
            replacementlist_seconditem.append(candidateitem)
        else:
            flag = 0
    if not replacementlist_firstitem:
        return -1

    else:
        with open(candidatesfilename,'a') as f:
            num_lines = sum(1 for line in open(candidatesfilename))
            if os.stat(candidatesfilename).st_size == 0:
                num_lines = 0
            for i in range(k):
                newsentence = inputsentence.replace(words[location1], entityDict.getString(replacementlist_firstitem[i]))
                newsentence = newsentence.replace(words[location2], entityDict.getString(replacementlist_seconditem[i]))
                f.write(str(num_lines+i) + ' '+ newsentence )
        f.close()
        return 1

def MakeNegativeSentence(inputsentence, sources, targets, typedict, entitytypedict, entityDict, relDict, allTriples):
    lmtzr = WordNetLemmatizer()
    words = inputsentence.split(' ')
    words = [word.replace(".", "") for word in words]
    words = [word.replace("\n", "") for word in words]
    words = [lmtzr.lemmatize(word) for word in words]

    for i in range(len(words)):
        if words[i] in sources:
            for j in range(i + 1, len(words)):
                if words[j] in targets:
                    location1 = i
                    location2 = j
                    Findreplacement(location1, location2, words, typedict, entitytypedict, entityDict, relDict,
                                    allTriples, inputsentence)
                    Findreplacement(location2, location1, words, typedict, entitytypedict, entityDict, relDict,
                                    allTriples, inputsentence)
    for i in range(len(words)):
        if words[i] in targets:
            for j in range(i + 1, len(words)):
                if words[j] in sources:
                    location1 = j
                    location2 = i
                    Findreplacement(location1, location2, words, typedict, entitytypedict, entityDict, relDict,
                                    allTriples, inputsentence)
                    Findreplacement(location2, location1, words, typedict, entitytypedict, entityDict, relDict,
                                    allTriples, inputsentence)


    return


def main():
    entityDict = Dictionary()
    relDict = Dictionary()
    allTriples = getTriplesFromFile(tensor_truetriples, entityDict, relDict)
    typedict , entitytypedict, schematriples,sources, targets = makeTypedict(tensor_truetriples, entityDict)

    '''
    data = {}
    data['entities'] = entityDict.getAllStrings()
    data['relations'] = relDict.getAllStrings()
    data['typedict'] = typedict
    data['entitytypedict'] = entitytypedict
    data['schematriples'] = schematriples

    with open(out_file, 'wb') as out:
       pickle.dump(data, out, protocol=2)
    '''

    for line in open(input_file):
        inputsentence = line
        inputsentence = inputsentence.lower()
        MakeNegativeSentence(inputsentence,sources, targets, typedict, entitytypedict,entityDict, relDict, allTriples)



if __name__ == '__main__':
    main()

# vim: et sw=4 sts=4
