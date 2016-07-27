
from collections import defaultdict as ddict
import nltk
from nltk.stem.wordnet import WordNetLemmatizer




def getTriplesFromFile(filename):
    entity_pair_relations = ddict(set)
    triples = []
    for line in open(filename):
        fields = line.split(',')
        source = fields[0]
        relation = fields[1]
        target = fields[2]
        triple = (source, relation, target)
        triples.append(triple)
        entity_pair_relations[(source, target)].add(relation)

    return entity_pair_relations



def maketype_dict(filename_all):
    #Note that this depends on form of the given tensor and where each column is.
    #The format of input CSV file is source,relation,target, in column 0,1,2 and source_type,target_type in columns
    # 8,9, respectively.
    targets = set()
    sources = set()
    type_dict = ddict(set)
    entity_type_dict = dict()
    schema_triples = set()
    for line in open(filename_all):
        fields = line.strip().split(',')
        source = fields[0]
        relation = fields[1]
        target = fields[2]
        source_type = fields[8]
        target_type = fields[9]
        schema_triple = (source_type, relation, target_type)
        schema_triples.add(schema_triple)
        type_dict[source_type].add(source)
        entity_type_dict[source] = source_type
        type_dict[target_type].add(target)
        entity_type_dict[target] = target_type
        sources.add(source)
        targets.add(target)
    return type_dict , entity_type_dict, schema_triples, sources, targets

def Findreplacement(location1, location2, words, type_dict, entity_type_dict, entity_pair_relations, input_sentence):
    #The method receives the sentence in words list and takes the location of the two words that we want to consider
    #Given these two words, the goal is to generate two sets of perturbations, by replacing one of these words at a time
    #Let us call these two, as word1, word2.
    #We look into triple KB and find a list of predicates->predicate_list that connect word1,word2.
    # note that we consider both orders
    #Then, given word1, we look into all the words that have the same type(type1) and make a list of the words
    # that have type1  but never appeared with word2 and any of the predicates in the KB. These are our candidate replacements.
    # The same approach is repeated for replacing word2.
    replacementlist_firstitem = []
    replacementlist_seconditem = []
    new_sentences_per_sentence = []
    type1 = entity_type_dict[words[location1]]
    type2 = entity_type_dict[words[location2]]


    predicate_list = entity_pair_relations[(words[location1], words[location2])]



    for candidateitem in type_dict[type1]:
        if len(entity_pair_relations[(candidateitem, words[location2])].intersection(predicate_list)) == 0:
            replacementlist_firstitem.append(candidateitem)
            replacementlist_seconditem.append(words[location2])

    for candidateitem in type_dict[type2]:
        if len(entity_pair_relations[(words[location1], candidateitem)].intersection(predicate_list)) == 0:
            replacementlist_firstitem.append(words[location1])
            replacementlist_seconditem.append(candidateitem)

    for i in range(max(len(replacementlist_firstitem), no_perturb)):
        new_sentence = input_sentence.replace(words[location1], replacementlist_firstitem[i])
        new_sentence = new_sentence.replace(words[location2], replacementlist_seconditem[i])
        new_sentences_per_sentence.append(new_sentence)
    return new_sentences_per_sentence

def MakeNegativeSentence(input_sentence, sources, targets, type_dict, entity_type_dict, entity_pair_relations):
    lmtzr = WordNetLemmatizer()
    words = nltk.word_tokenize(input_sentence)
    words = [lmtzr.lemmatize(word) for word in words]

    new_sentences = []
    for i in range(len(words)):
        if words[i] in sources:
            for j in range(i + 1, len(words)):
                if words[j] in targets:
                    location1 = i
                    location2 = j
                    new_sentences.extend(Findreplacement(location1, location2, words, type_dict, entity_type_dict,
                                                         entity_pair_relations, input_sentence))
                    new_sentences.extend(Findreplacement(location2, location1, words, type_dict, entity_type_dict,
                                                         entity_pair_relations, input_sentence))
    for i in range(len(words)):
        if words[i] in targets:
            for j in range(i + 1, len(words)):
                if words[j] in sources:
                    location1 = j
                    location2 = i
                    new_sentences.extend(Findreplacement(location1, location2, words, type_dict, entity_type_dict,
                                                         entity_pair_relations, input_sentence))
                    new_sentences.extend(Findreplacement(location2, location1, words, type_dict, entity_type_dict,
                                                         entity_pair_relations, input_sentence))

    with open(candidates_file, 'a') as f:
        for i in range(len(new_sentences)):
            f.write(str(i) + ' ' + new_sentences[i])
    f.close()
    return


def main():
    entity_pair_relations = getTriplesFromFile(KBtensor_file)
    type_dict, entity_type_dict, schema_triples, sources, targets = maketype_dict(KBtensor_file)


    for line in open(input_file):
        input_sentence = line
        input_sentence = input_sentence.lower()
        MakeNegativeSentence(input_sentence, sources, targets, type_dict, entity_type_dict, entity_pair_relations)




if __name__ == '__main__':
    main()


    argparser = argparse.ArgumentParser(description="Perturb sentences using KB and type information")
    argparser.add_argument("--input_file", type=str, help="File with sentences to perturb,\
            one per line.")
    argparser.add_argument("--candidates_file", type=str, help="File with purturbed sentences along with an id,\
            one per line.")
    argparser.add_argument("--KBtensor_file", type=str, help="File with sentences to replace words,\
        one per line.")
    argparser.add_argument("--no_perturb", type=int, help="no. of word replacements per word combination in sentence, default=20",
                       default=20)

    args = argparser.parse_args()


# vim: et sw=4 sts=4
