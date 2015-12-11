#!/usr/bin/python

import sys

raw_data_filename = sys.argv[1]

baseline_out_filename = sys.argv[2]

cat_words = {}
cat_word_list = []
num_cat_words = 0
rel_words = {}
rel_word_list = []
num_rel_words = 0

cat_clusters = []
rel_clusters = []
with open(raw_data_filename, 'r') as raw_data_file:
    for line in raw_data_file:
        parts = line.strip().split('\t')

        midpart = parts[0]
        cat_word_part = parts[2].replace("\\","\\\\")
        rel_word_part = parts[3].replace("\\","\\\\")

        mids = midpart.split(" ")

        if (cat_word_part.strip() != ""):
            if not cat_words.has_key(cat_word_part):
                cat_words[cat_word_part] = num_cat_words
                cat_word_list.append(cat_word_part)
                cat_clusters.append(set())
                num_cat_words += 1

            ind = cat_words[cat_word_part]
            mid = mids[0]
            cat_clusters[ind].add(mid)

        if (rel_word_part.strip() != ""):
            if not rel_words.has_key(rel_word_part):
                rel_words[rel_word_part] = num_rel_words
                rel_word_list.append(rel_word_part)
                rel_clusters.append(set())
                num_rel_words += 1

            ind = rel_words[rel_word_part]
            rel_clusters[ind].add((mids[0], mids[1]))
    
def dump_word_list_to_file(word_list, f):
    for word in word_list:
        print >> f, ('"' + word + '"')
    print >> f, "\"<UNK>\"",

with open(baseline_out_filename, 'w') as baseline_out_file:
    print >> baseline_out_file, "(define cat-word-dict (make-dictionary "
    dump_word_list_to_file(cat_word_list, baseline_out_file)
    print >> baseline_out_file, "))"

    print >> baseline_out_file, "(define cat-word-cluster-names (array "
    dump_word_list_to_file(cat_word_list, baseline_out_file)
    print >> baseline_out_file, "))"

    print >> baseline_out_file, "(define cat-cluster-dict (make-dictionary "
    dump_word_list_to_file(cat_word_list, baseline_out_file)
    print >> baseline_out_file, "))"

    print >> baseline_out_file, "(define cat-clusters (array"
    for i in xrange(len(cat_clusters)):
        print >> baseline_out_file, "(make-dictionary ", " ".join(['"'+ x + '"' for x in cat_clusters[i]]) ,")"
    print >> baseline_out_file, "(make-dictionary)"
    print >> baseline_out_file, "))"

    print >> baseline_out_file, "(define rel-word-dict (make-dictionary "
    dump_word_list_to_file(rel_word_list, baseline_out_file)
    print >> baseline_out_file, "))"

    print >> baseline_out_file, "(define rel-word-cluster-names (array "
    dump_word_list_to_file(rel_word_list, baseline_out_file)
    print >> baseline_out_file, "))"

    print >> baseline_out_file, "(define rel-cluster-dict (make-dictionary "
    dump_word_list_to_file(rel_word_list, baseline_out_file)
    print >> baseline_out_file, "))"

    print >> baseline_out_file, "(define rel-clusters (array"
    for i in xrange(len(rel_clusters)):
        print >> baseline_out_file, "(make-dictionary ", " ".join(['(list "'+ x + '" "' + y + '")' for (x, y) in rel_clusters[i]]) ,")"
    print >> baseline_out_file, "(make-dictionary)"
    print >> baseline_out_file, "))"
