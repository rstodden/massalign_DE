# Introduction:

This repository contains the code of the extended version of **MASSAlign**, to which a Doc2Vec language model has been added, as well as a new alignment method. 
In addition, this repository contains a new dataset to be used for text simplification systems, which can be found in the dataset folder.

**Dataset**
* The dataset is composed of pairs of aligned text units retrieved from the original works of Early Modern Philosophers such as *George Berkeley*, *David Hume*, *John Locke* and *John Stuart Mill* and their simplified versions of the works obtained from and edited by [EarlyModernTexts](https://www.earlymoderntexts.com/).
* There are two files available: 
    - *dataset_paragraphs.txt* in which the text is aligned at paragraph level
    - *dataset_sentences.txt* in which the text is aligned at sentence level
* The structure of the files is as follows (both for sentence and paragraph level):
    - First line is the original text unit of the aligned pair
    - Second line is the simplified text unit of the aligned pair
    - Each pair of aligned text units is separated by an empty line 
    - In cases where there is more than one text unit per line, that marks a splitting or concatenating text simplification operation
* More details: Stefan Paun. **Parallel Text Alignment and Monolingual Parallel Corpus Creation from Philosophical Texts for Text Simplification**.

**MASSAlign** is a *Python 2* library for the alignment and annotation of comparable documents.
It offers **3** main functionalities:
* Paragraph-level alignment between comparable documents
* Sentence-level alignment between comparable paragraphs or documents
* Word-level annotation of modification operations between aligned sentences

**MASSAlign** offers 2 different alignment techniques suitable for either a TF-IDF language model or Doc2Vec language model. The language models are used as a basis for text similarity computation:
* *VicinityDrivenParagraphAligner* and *VicinityDrivenSentenceAligner* for use with TF-IDF model. More details: Gustavo H. Paetzold and Lucia Specia. **Vicinity-Driven Paragraph and Sentence Alignment for Comparable Corpora**. arXiv preprint arXiv:1612.04113.
* *ExpandingAlingmentParagraphAligner* and *ExpandingAlingmentSentenceAligner* for use with the Doc2Vec model. More details: Stefan Paun. **Parallel Text Alignment and Monolingual Parallel Corpus Creation from Philosophical Texts for Text Simplification**.

**MASSAlign** is excellent for extracting simplifications from complex/simple parallel documents!
These papers are evidence:
* Gustavo H. Paetzold and Lucia Specia. **Lexical Simplification with Neural Ranking**. Proceedings of the 2017 EACL.
* Fernando Alva-Manchego, Joachim Bingel, Gustavo H. Paetzold, Carolina Scarton and Lucia Specia. **Learning how to Simplify from Explicit Labeling of Complex-Simplified Text Pairs**. Proceedings of the 2017 IJCNLP.

# Installation:

To install **MASSAlign**, you must:
1. Download and unpack MASSAlign's github [repository](https://github.com/ghpaetzold/massalign/archive/master.zip).
2. Navigate to the root folder.
3. Run the following command line:

```
python setup.py install
```

# Documentation:

**MASSAlign's** documentation can be found [here](http://ghpaetzold.github.io/massalign_docs).

# Examples:

An example for the *ExpandingAlignment* methods can be found in [/example/d2v_example.py](https://github.com/stefanpaun/massalign/blob/master/example/d2v_example.py).
You can learn how to use **MASSAlign** [here](http://ghpaetzold.github.io/massalign_docs/examples.html).

# Citing:

If you use **MASSAlign**, please cite this paper:
* Gustavo H. Paetzold, Fernando Alva-Manchego and Lucia Specia. **MASSAlign: Alignment and Annotation for Comparable Documents**. Proceedings of the 2017 IJCNLP.

If you use the Doc2Vec language model or the *ExpandingAlignment* method, please cite, in addition, this paper:
* Stefan Paun. **Parallel Text Alignment and Monolingual Parallel Corpus Creation from Philosophical Texts for Text Simplification**.

# License:

**MASSAlign** is distributed under the LGPL v3.0 license.
