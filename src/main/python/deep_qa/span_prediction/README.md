The script in 'span_model.py' runs a model for predicting spans of
fill-the-gap questions.

It is for now only training on the cases present in Omnibus-Gr04 and
Omnibus-Gr08.

Example prediction on Barron's statements can be generated to inspect
the model's outputs on exemplary science facts.


Note:
Stanford CoreNLP is used to compute parse trees. It can be downloaded from http://nlp.stanford.edu/software/corenlp.shtml

After calling 'span_model.py' once, sentences are written into two files,
line by line.
These serve as input files for CoreNLP ('SENTENCE_FILE.txt').

CoreNLP can be run with

'cd path_to_CoreNLP_downlowd/stanford-corenlp-full-2015-12-09'
'java -cp "\*" -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse, -file SENTENCE_FILE.txt -outputFormat json'

'cp SENTENCE_FILE.txt.json directory/of/span_model.py'
