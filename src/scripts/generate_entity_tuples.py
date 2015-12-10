#!/usr/bin/python

import re
import json
import sys

from collections import defaultdict

train_filename = sys.argv[1]
entity_filename = sys.argv[2]
word_filename = sys.argv[3]
lf_filename = sys.argv[4]
word_count_threshold = int(sys.argv[5])
uniq = (int(sys.argv[6]) != 0)
inv = (int(sys.argv[7]) != 0)
joint = (int(sys.argv[8]) != 0)

INVERSE_REL_FORMAT="inverse:%s"

arg1_arg2_map = {}
arg2_arg1_map = {}
entity_counts = {}
entity_tuple_counts = {}
lfs = []

rel_occurring_map = defaultdict(set)
cat_occurring_map = defaultdict(set)
joint_occurring_map = defaultdict(set)
joint_qk_index = {}

cat_word_counts = {}
rel_word_counts = {}


def json_to_query_key(json_obj, mid):
    query_key_list = []
    for part in json_obj:
        if len(part) == 2:
            if part[1] == mid:
                query_key_list.append( (part[0], None) )
        elif len(part) == 3:
            if part[1] == mid:
                query_key_list.append( (part[0], None, part[2]) )
            elif part[2] == mid:
                query_key_list.append( (part[0], part[1], None) )

    query_key_list.sort()
    return tuple(query_key_list)

def read_words(part):
    return [x.replace("\\","\\\\") for x in part.split(" ")]

with open(train_filename, 'r') as train_file:
    for line in train_file:
        parts = line.strip().split("\t")
        cat_word_part = parts[2].strip()
        rel_word_part = parts[3].strip()

        if not cat_word_part == "":
            cat_words = read_words(cat_word_part)
            for cat_word in cat_words:
                if not cat_word_counts.has_key(cat_word):
                    cat_word_counts[cat_word] = 0
                cat_word_counts[cat_word] += 1

        if not rel_word_part == "":
            rel_words = read_words(rel_word_part)
            for rel_word in rel_words:
                if not rel_word_counts.has_key(rel_word):
                    rel_word_counts[rel_word] = 0
                rel_word_counts[rel_word] += 1

                if inv:
                    inv_rel_word = INVERSE_REL_FORMAT % rel_word
                    if not rel_word_counts.has_key(inv_rel_word):
                        rel_word_counts[inv_rel_word] = 0
                    rel_word_counts[inv_rel_word] += 1

seen_lfs = set()
with open(train_filename, 'r') as train_file:
    for line in train_file:
        parts = line.strip().split("\t")
        mid_part = parts[0]
        cat_word_part = parts[2].strip()
        rel_word_part = parts[3].strip()

        cat_words = []
        if not cat_word_part == "":
            cat_words = read_words(cat_word_part)

        rel_words = []
        if not rel_word_part == "":
            rel_words = read_words(rel_word_part)

        if not (all([cat_word_counts[x] > word_count_threshold for x in cat_words]) and all([rel_word_counts[x] > word_count_threshold for x in rel_words])):
            continue

        if uniq and parts[4] in seen_lfs:
            continue

        try:
            json_obj = json.loads(parts[5])
        except ValueError:
            continue

        lfs.append((parts[4], json_obj))
        if uniq:
            seen_lfs.add(parts[4])

        mids = mid_part.split(" ")
        for mid in mids:
            if not entity_counts.has_key(mid):
                entity_counts[mid] = 0
            entity_counts[mid] += 1

        if joint:
            for mid in mids:
                query_key = json_to_query_key(json_obj, mid)
                joint_occurring_map[query_key].add(mid)

        for cat_word in cat_words:
            # lfs.append("(random-cat-neg-example (word-cat \"%s\"))" % cat_word)
            # lfs.append("(weighted-random-cat-neg-example (word-cat \"%s\"))" % cat_word)
            pass

        for rel_word in rel_words:
            # lfs.append("(random-rel-neg-example (word-rel \"%s\"))" % rel_word)
            # lfs.append("(weighted-random-rel-neg-example (word-rel \"%s\"))" % rel_word)
            pass

            if inv:
                # Generate training examples for inverse relations
                inv_rel_name = INVERSE_REL_FORMAT % rel_word
                lfs.append('((word-rel "%s") "%s" "%s")' % (inv_rel_name, mids[1], mids[0]))
                lfs.append("(weighted-random-rel-neg-example (word-rel \"%s\"))" % inv_rel_name)

        for cat_word in cat_words:
            assert len(mids) == 1
            cat_occurring_map[cat_word].add(mids[0])

        for rel_word in rel_words:
            assert len(mids) == 2
            rel_occurring_map[rel_word].add((mids[0], mids[1]))

        if len(mids) == 2:
            arg1 = mids[0]
            arg2 = mids[1]

            if not arg1_arg2_map.has_key(arg1):
                arg1_arg2_map[arg1] = set()
            if not arg2_arg1_map.has_key(arg2):
                arg2_arg1_map[arg2] = set()

            arg1_arg2_map[arg1].add(arg2)
            arg2_arg1_map[arg2].add(arg1)

            if not entity_tuple_counts.has_key((arg1, arg2)):
                entity_tuple_counts[(arg1, arg2)] = 0
            entity_tuple_counts[(arg1, arg2)] += 1


with open(entity_filename, 'w') as entity_file:
    entity_list = sorted(entity_counts.keys())
    print >> entity_file, "(define entity-histogram (make-histogram "
    for entity in entity_list:
        print >> entity_file, '(list "%s" %d)' % (entity, entity_counts[entity])
    print >> entity_file, '))'
    print >> entity_file, '(define entities (histogram-to-dictionary entity-histogram))'

    print >> entity_file, "(define entity-tuple-histogram (make-histogram "
    for arg1 in arg1_arg2_map.keys():
        for arg2 in arg1_arg2_map[arg1]:
            print >> entity_file, '(list (list "%s" "%s") %d)' % (arg1, arg2, entity_tuple_counts[(arg1, arg2)])
    print >> entity_file, '))'
    print >> entity_file, '(define entity-tuples (histogram-to-dictionary entity-tuple-histogram))'

    print >> entity_file, '(define related-entities (array '
    for entity in entity_list:
        related_entities = set()
        if arg1_arg2_map.has_key(entity):
            related_entities.update(arg1_arg2_map[entity])
        if arg2_arg1_map.has_key(entity):
            related_entities.update(arg2_arg1_map[entity])

        print >> entity_file, "(array", " ".join(['"%s"' % x for x in related_entities]),")"
    print >> entity_file, '))'

    print >> entity_file, '(define arg1-arg2-map (array '
    for entity in entity_list:
        related_entities = set()
        if arg1_arg2_map.has_key(entity):
            related_entities.update(arg1_arg2_map[entity])

        print >> entity_file, "(make-dset entities (array", " ".join(['"%s"' % x for x in related_entities]),"))"
    print >> entity_file, '))'

    print >> entity_file, '(define arg2-arg1-map (array '
    for entity in entity_list:
        related_entities = set()
        if arg2_arg1_map.has_key(entity):
            related_entities.update(arg2_arg1_map[entity])

        print >> entity_file, "(make-dset entities (array", " ".join(['"%s"' % x for x in related_entities]),"))"
    print >> entity_file, '))'
        
with open(word_filename, 'w') as word_file:
    cat_word_list = [x for (x, y) in cat_word_counts.iteritems() if y > word_count_threshold]
    cat_word_list.sort()
    cat_word_list = ['<UNK>'] + cat_word_list

    rel_word_list = [x for (x, y) in rel_word_counts.iteritems() if y > word_count_threshold]
    rel_word_list.sort()
    rel_word_list = ['<UNK>'] + rel_word_list

    print >> word_file, '(define cat-words (make-dictionary '
    for cat_word in cat_word_list:
        print >> word_file, '"%s"' % cat_word
    print >> word_file, "))"

    print >> word_file, '(define rel-words (make-dictionary '
    for rel_word in rel_word_list:
        print >> word_file, '"%s"' % rel_word
    print >> word_file, '))'

    cat_word_set = set(cat_word_list)
    unk_observations = set()
    for word in cat_occurring_map.iterkeys():
        if word not in cat_word_set:
            unk_observations.update(cat_occurring_map[word])
    cat_occurring_map['<UNK>'] = unk_observations

    rel_word_set = set(rel_word_list)
    unk_observations = set()
    for word in rel_occurring_map.iterkeys():
        if word not in rel_word_set:
            unk_observations.update(rel_occurring_map[word])
    rel_occurring_map['<UNK>'] = unk_observations

    print >> word_file, '(define cat-word-entities (array '
    for word in cat_word_list:
        positive_occurrences = cat_occurring_map[word]
        positive_example_str = "(make-dset entities (array "  + " ".join(['"%s"' % x for x in positive_occurrences]) + "))"
        print >> word_file, positive_example_str
    print >> word_file, '))'

    print >> word_file, '(define rel-word-entities (array '
    for word in rel_word_list:
        positive_occurrences = rel_occurring_map[word]
        positive_example_str = "(make-dset entity-tuples (array "  + " ".join(['(list "%s" "%s")' % x for x in positive_occurrences]) + "))"
        print >> word_file, positive_example_str
    print >> word_file, '))'

    query_key_list = sorted(joint_occurring_map.keys())
    print >> word_file, '(define joint-entities (array '
    for (i, qk) in enumerate(query_key_list):
        joint_qk_index[qk] = i
        positive_occurrences = joint_occurring_map[qk]
        positive_example_str = "(make-dset entities (array "  + " ".join(['"%s"' % x for x in positive_occurrences]) + "))"
        print >> word_file, positive_example_str
    print >> word_file, '))'

with open(lf_filename, 'w') as lf_file:
    entity_set = set(entity_list)

    print >> lf_file, "(define training-inputs (array "
    for (lf, json_obj) in lfs:
        # print >> lf_file, "(list (quote %s) (list))" % lf

        if joint:
            rels = [x for x in json_obj if (len(x) == 3 and x[1] in entity_set and x[2] in entity_set)]
            arg1s = defaultdict(set)
            arg2s = defaultdict(set)

            for rel in rels:
                first = '"%s"' % rel[1]
                second = '"%s"' % rel[2]
                
                arg1s[first].add(second)
                arg2s[second].add(first)

            entity_names = re.findall('"/m/[^"]*"', lf)
            entity_name_set = set(entity_names)
            if len(entity_name_set) < len(entity_names):
                continue

            entity_names_uniq = list(entity_name_set)

            for entity_name in entity_names_uniq:
                var_name = 'var'
                neg_var_name = 'neg-var'
                subbed_lf = re.sub(entity_name, "%s %s" % (var_name, neg_var_name), lf)

                for other_entity_name in entity_names_uniq:
                    subbed_lf = re.sub(other_entity_name, "%s %s" % (other_entity_name, other_entity_name), subbed_lf)

                arg1_list = "(list " + " ".join(arg2s[entity_name]) + ")"
                arg2_list = "(list " + " ".join(arg1s[entity_name]) + ")"

                qk = json_to_query_key(json_obj, entity_name.strip('"'))
                ind = joint_qk_index[qk]

                print >> lf_file, "(list (quote (lambda (var neg-var)", subbed_lf, ")) (list", entity_name, ")", arg1_list, " ", arg2_list, " ", ind, ")"
        else:
            entity_names = re.findall('"/m/[^"]*"', lf)
            entity_name_set = set(entity_names)
            if len(entity_name_set) < len(entity_names):
                continue

            var_names = []
            subbed_lf = lf
            for i in xrange(len(entity_names)):
                var_name = 'var%d' % i
                neg_var_name = 'neg-var%d' % i
                subbed_lf = re.sub(entity_names[i], "%s %s" % (var_name, neg_var_name), subbed_lf)
                var_names.append(var_name)
                var_names.append(neg_var_name)

            positive_example_str = ""
            if len(entity_names) == 1:
                word = re.search('word-cat "([^"]*)"', subbed_lf).group(1)
                positive_example_str = '"%s"' % word
            else:
                word = re.search('word-rel "([^"]*)"', subbed_lf).group(1)
                positive_example_str = '"%s"' % word
                
            print >> lf_file, "(list (quote (lambda (", " ".join(var_names), ")", subbed_lf, ")) (list", " ".join(entity_names), ")", positive_example_str, ")"

    print >> lf_file, "))"

