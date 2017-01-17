This module contains code for a model that identifies simple declarative science statements from a text.
The filter can be applied to single sentences or to paragraphs. 

Requirements:
- Download two text files with lists of first names from [here](http://www.cs.cmu.edu/Groups/AI/util/areas/nlp/corpora/names/). The files are called _male.txt_ and _female.txt_. 

Before you get started you need to specify a few file paths in the main function of _filter.py_.
Insert your local paths pointing to the names files (see above), and to a file containing the text you would like to filter (with one sentence/paragraph per line).

If combined with output from a Mechanical Turk task, a file can be specified which contains paragraphs that _shouldn't_ be used (e.g. because they were used already).
