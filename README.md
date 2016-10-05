[![Build Status](https://semaphoreci.com/api/v1/projects/b3480192-615d-4981-ba34-62afeb9d9ae6/953929/shields_badge.svg)](https://semaphoreci.com/allenai/deep_qa)

# Deep QA

This repository contains code for training deep learning systems to do question answering tasks.
Our primary focus is on Aristo's science questions, though we can run various models on several
popular datasets.

This code is a mix of scala (for data processing / pipeline management) and python (for actually training and executing deep models with Keras / Theano / TensorFlow).

# Implemented models

This repository implements several variants of memory networks, including the models found in these papers:

- The original MemNN, from [Memory Networks](https://arxiv.org/abs/1410.3916), by Weston, Chopra and Bordes
- [End-to-end memory networks](https://www.semanticscholar.org/paper/End-To-End-Memory-Networks-Sukhbaatar-Szlam/10ebd5c40277ecba4ed45d3dc12f9f1226720523), by Sukhbaatar and others (close, but still in progress)
- [Dynamic memory networks](https://www.semanticscholar.org/paper/Ask-Me-Anything-Dynamic-Memory-Networks-for-Kumar-Irsoy/04ee77ef1143af8b19f71c63b8c5b077c5387855), by Kumar and others (close, but still in progress)
- DMN+, from [Dynamic Memory Networks for Visual and Textual Question Answering](https://www.semanticscholar.org/paper/Dynamic-Memory-Networks-for-Visual-and-Textual-Xiong-Merity/b2624c3cb508bf053e620a090332abce904099a1), by Xiong, Merity and Socher (close, but still in progress)
- The attentive reader, from [Teaching Machines to Read and Comprehend](https://www.semanticscholar.org/paper/Teaching-Machines-to-Read-and-Comprehend-Hermann-Kocisk%C3%BD/2cb8497f9214735ffd1bd57db645794459b8ff41), by Hermann and others
- Windowed-memory MemNNs, from [The Goldilocks Principle: Reading Children's Books with Explicit Memory Representations](https://www.semanticscholar.org/paper/The-Goldilocks-Principle-Reading-Children-s-Books-Hill-Bordes/1ee46c3b71ebe336d0b278de9093cfca7af7390b) (in progress)

As well as some of our own, as-yet-unpublished variants.  There is a lot of similarity between the models in these papers, and our code is structured in a way to allow for easily switching between these models.

# Datasets

This code allows for easy experimentation with the following datasets:

- [AI2 Elementary school science questions (no diagrams)](http://allenai.org/data.html)
- [The Facebook Children's book test dataset](https://research.facebook.com/research/babi/#cbt)
- [The Facebook bAbI dataset](https://research.facebook.com/research/babi/)

And more to come...  In the near future, we hope to also include easy experimentation with [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/), [CNN/Daily Mail](http://cs.nyu.edu/~kcho/DMQA/), and [SimpleQuestions](https://research.facebook.com/research/babi/).

# Usage Guide

TODO.  Sorry.  This will be done soon.

# License

This code is released under the terms of the [Apache 2 license](https://www.apache.org/licenses/LICENSE-2.0).
