import argparse #imports the argparse module so it can be used
from collections import defaultdict as ddict
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


def getTriplesFromFile(filename):
    entity_pair_relations = ddict(set)

    for line in open(filename):
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
    schema_triples = set()
    for line in open(filename):
        fields = line.strip().split(',')
        source = fields[0]
        relation = fields[1]
        target = fields[2]
        source_type = fields[8]
        target_type = fields[9]
        schema_triple = (source_type, relation, target_type)
        schema_triples.add(schema_triple)
        type_entity_dict[source_type].add(source)
        entity_type_dict[source].add(source_type)
        type_entity_dict[target_type].add(target)
        entity_type_dict[target].add(target_type)
        entities.add(source)
        entities.add(target)
    return type_entity_dict, entity_type_dict, schema_triples, entities

def find_replacement(location1, location2, words, type_entity_dict, entity_type_dict, entity_pair_relations, 
                    input_sentence, num_perturbation):

    # The method receives the sentence in words list and takes the location of the two words that we want to consider
    # Given these two words, the goal is to generate two sets of perturbations, by replacing one of these words at a time
    # Let us call these two, as word1, word2.
    # We look into triple KB and find a list of predicates->predicate_list that connect word1,word2.
    # note that we consider both orders
    # Then, given word1, we look into all the words that have the same type(type1) and make a list of the words
    # that have type1  but never appeared with word2 and any of the predicates in the KB. These are our candidate replacements.
    # The same approach is repeated for replacing word2.
    replacement_list = []

    negative_sentences_per_sentence = []

    for type1 in entity_type_dict[lemmatizer.lemmatize(words[location1])]:
        for type2 in entity_type_dict[lemmatizer.lemmatize(words[location2])]:
            predicate_list = entity_pair_relations[(lemmatizer.lemmatize(words[location1]), lemmatizer.lemmatize(words[location2]))]

            for candidate_item in type_entity_dict[type1]:
                if len(entity_pair_relations[(candidate_item, words[location2])].intersection(predicate_list)) == 0:
                    replacement_list.append((candidate_item, words[location2]))

            for candidate_item in type_entity_dict[type2]:
                if len(entity_pair_relations[(words[location1], candidate_item)].intersection(predicate_list)) == 0:
                    replacement_list.append((words[location1], candidate_item))


    num_replacements = min(len(replacement_list), num_perturbation)
    for (replacement1, replacement2) in replacement_list[:num_replacements]:
        new_sentence = input_sentence.replace(words[location1], replacement1).replace(words[location2], replacement2)
        negative_sentences_per_sentence.append(new_sentence)
    return negative_sentences_per_sentence

def create_negative_sentence(input_sentence, entities, type_entity_dict, entity_type_dict, entity_pair_relations
                             , num_perturbation):

    words = nltk.word_tokenize(input_sentence)
    negative_sentences = []

    for i in range(len(words)):
        if lemmatizer.lemmatize(words[i]) in entities:
            for j in range(i + 1, len(words)):
                if words[j] in entities:
                    location1 = i
                    location2 = j

                    negative_sentences.extend(find_replacement(location1, location2, words, type_entity_dict, entity_type_dict,
                                                         entity_pair_relations, input_sentence, num_perturbation))
                    negative_sentences.extend(find_replacement(location2, location1, words, type_entity_dict, entity_type_dict,
                                                         entity_pair_relations, input_sentence, num_perturbation))

    return negative_sentences

def main():

    argparser = argparse.ArgumentParser(description="Perturb sentences using KB and type information")
    argparser.add_argument("--input_file", type=str, help="File with sentences to perturb, one per line.")
    argparser.add_argument("--output_file", type=str, help="File with purturbed sentences along with an id, one per line.")
    argparser.add_argument("--kb_tensor_file", type=str, help="input KB tensor in csv format with type information,one per line.")
    argparser.add_argument("--num_perturbation", type=int,\
                           help="no. of word replacements per word combination in sentence, default=20",\
                            default=20)
    args = argparser.parse_args()

    entity_pair_relations = getTriplesFromFile(filename = args.kb_tensor_file)
    type_entity_dict, entity_type_dict, schema_triples, entities = create_type_dict(filename = args.kb_tensor_file)

    negative_sentences = []
    for line in open(args.input_file):
        input_sentence = line
        input_sentence = input_sentence.lower()
        negative_sentences.extend(create_negative_sentence(input_sentence, entities, type_entity_dict, entity_type_dict, entity_pair_relations, num_perturbation = args.num_perturbation))
    with open(args.output_file, 'a') as f:
        for i, sentence in enumerate(negative_sentences):
            f.write(str(i) + ' ' + sentence)
    f.close()
if __name__ == '__main__':
    main()

# vim: et sw=4 sts=4
