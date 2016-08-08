from __future__ import print_function
import argparse
import codecs
import random
from collections import defaultdict as ddict

import nltk
from nltk.stem.wordnet import WordNetLemmatizer


def get_triples_from_file(filename):
    entity_pair_relations = ddict(set)
    for line in codecs.open(filename, "r", "utf-8"):
        fields = line.split(',')
        source = fields[0]
        relation = fields[1]
        target = fields[2]
        entity_pair_relations[(source, target)].add(relation)
    return entity_pair_relations


def create_type_dict(filename):
    # Note that this depends on form of the given tensor and where each column is.
    # The format of input CSV file is source,relation,target, in column 0,1,2 and source_type,target_type in columns
    # 8,9, respectively.
    entities = set()
    type_entity_dict = ddict(set)
    entity_type_dict = ddict(set)
    for line in codecs.open(filename, "r", "utf-8"):
        fields = line.strip().split(',')
        source = fields[0]
        target = fields[2]
        source_type = fields[8]
        target_type = fields[9]
        type_entity_dict[source_type].add(source)
        entity_type_dict[source].add(source_type)
        type_entity_dict[target_type].add(target)
        entity_type_dict[target].add(target_type)
        entities.add(source)
        entities.add(target)
    return type_entity_dict, entity_type_dict, entities


def find_replacement(location1, location2, words, type_entity_dict, entity_type_dict,
                     entity_pair_relations, input_sentence):
    # The method receives the sentence in words list and takes the location of the two words that we want to consider
    # Given these two words, the goal is to generate two sets of perturbations, by replacing one of these words at a time
    # Let us call these two, as word1, word2.
    # We look into triple KB and find a list of predicates->predicate_list that connect word1,word2.
    # note that we consider both orders
    # Then, given word1, we look into all the words that have the same type(type1) and make a list of the words
    # that have type1  but never appeared with word2 and any of the predicates in the KB. These are our candidate replacements.
    # The same approach is repeated for replacing word2.
    lemmatizer = WordNetLemmatizer()
    replacement_list = []

    negative_sentences_per_sentence = []
    lemma1 = lemmatizer.lemmatize(words[location1])
    lemma2 = lemmatizer.lemmatize(words[location2])

    for type1 in entity_type_dict[lemma1]:
        for type2 in entity_type_dict[lemma2]:
            predicate_list = entity_pair_relations[(lemma1, lemma2)]
            for candidate_item in type_entity_dict[type1]:
                if len(entity_pair_relations[(candidate_item, lemma2)].intersection(predicate_list)) == 0:
                    replacement_list.append((candidate_item, words[location2]))
            for candidate_item in type_entity_dict[type2]:
                if len(entity_pair_relations[(lemma1, candidate_item)].intersection(predicate_list)) == 0:
                    replacement_list.append((words[location1], candidate_item))

    for (replacement1, replacement2) in replacement_list:
        new_sentence = input_sentence.replace(words[location1], replacement1).replace(words[location2], replacement2)
        negative_sentences_per_sentence.append(new_sentence)
    return negative_sentences_per_sentence


def create_negative_sentence(input_sentence, entities, type_entity_dict, entity_type_dict,
                             entity_pair_relations):
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(input_sentence)
    negative_sentences = []

    for i in range(len(words)):  # pylint: disable=consider-using-enumerate
        if lemmatizer.lemmatize(words[i]) in entities:
            for j in range(i + 1, len(words)):
                if lemmatizer.lemmatize(words[j]) in entities:
                    negative_sentences.extend(
                            find_replacement(i, j, words, type_entity_dict, entity_type_dict,
                                             entity_pair_relations, input_sentence))
                    negative_sentences.extend(
                            find_replacement(j, i, words, type_entity_dict, entity_type_dict,
                                             entity_pair_relations, input_sentence))
    return negative_sentences


def main():
    '''Takes as input a list of sentences and a KB, and produces as output a collection of
    corrupted sentences.

    The input sentences are assumed formatted as one sentence per line, possibly with an index:
    either "[sentence]" or "[sentence id][tab][sentence".

    The input KB format is described in the comment to create_type_dict.

    The output format is a tab-separated list of corruptions per sentence.  For every sentence for
    which we found a corruption, we output a line formatted as "[sentence][tab][sentence][tab]...".
    Sentences for which we found no corruption are just skipped.
    '''
    argparser = argparse.ArgumentParser(description="Perturb sentences using KB and type information")
    argparser.add_argument("--input_file", type=str, help="File with sentences to perturb, one per line.")
    argparser.add_argument("--output_file", type=str, help="File with purturbed sentences along with an id, one per line.")
    argparser.add_argument("--kb_tensor_file", type=str, help="input KB tensor in csv format with type information,one per line.")
    argparser.add_argument("--num_perturbation", type=int,\
                           help="no. of word replacements per word combination in sentence, default=20",\
                            default=20)
    args = argparser.parse_args()

    entity_pair_relations = get_triples_from_file(args.kb_tensor_file)
    type_entity_dict, entity_type_dict, entities = create_type_dict(args.kb_tensor_file)

    negative_sentences = []
    for line in codecs.open(args.input_file, "r", "utf-8"):
        if '\t' in line:
            input_sentence = line.strip().split('\t')[1]
        else:
            input_sentence = line.strip()
        input_sentence = input_sentence.lower()
        negatives = create_negative_sentence(input_sentence, entities, type_entity_dict,
                                             entity_type_dict, entity_pair_relations)
        if negatives:
            random.shuffle(negatives)
            negative_sentences.append(negatives[:args.num_perturbation])
    with codecs.open(args.output_file, 'w', 'utf-8') as out_file:
        for sentences in negative_sentences:
            out_file.write('\t'.join(sentences) + '\n')


if __name__ == '__main__':
    main()
