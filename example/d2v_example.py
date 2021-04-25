#%%

from massalign.core import *
from time import time
import numpy as np
import statistics
import scipy.stats as st


#Get files to align:
path =  ' '
file_name1 = ' '
file_name2 = ' '


file1 = path + file_name1
file2 = path + file_name2


#D2V parameters
vector_size = 300
window = 15
min_count = 3
epochs = 60
infer_epochs = 60

#Initialize MASSAligner object, model with the parameteres and two empty aligners to determine the thresholds
m = MASSAligner()
model = D2VModel(vector_size=vector_size, window_size = window, min_count = min_count, epochs = epochs, infer_epochs = infer_epochs, input_files=[file1, file2])
paragraph_aligner = ExpandingAlingmentParagraphAligner(similarity_model=model)
sentence_aligner = ExpandingAlingmentSentenceAligner(similarity_model=model)


#Read paragraphs from document file
p1s = m.getParagraphsFromDocument(file1)
p2s = m.getParagraphsFromDocument(file2)


#Compute paragraph aligner thresholds based on the similarity matrix
mat = m.getSimMatrixPar(p1s, p2s, paragraph_aligner)
par_history = []
for line in mat:
        par_history.append(max(line))
(low, high) = st.t.interval(0.95, len(par_history)-1, loc=np.mean(par_history), scale=st.sem(par_history))

hard_treshold = low-statistics.stdev(par_history)
soft_threshold = max(min(par_history),0)
certain_threshold = high+statistics.stdev(par_history)

#Reintialize aligner with the thresholds determined earlier and align paragraphs
paragraph_aligner = ExpandingAlingmentParagraphAligner(similarity_model=model, certain_threshold=certain_threshold, hard_threshold=hard_treshold, soft_threshold=soft_threshold)
alignments_par, aligned_paragraphs = m.getParagraphAlignments(p1s, p2s, paragraph_aligner)




#Compute sentence aligner thresholds based on the similarity matrix
sent_history = []
for a in aligned_paragraphs:
    p1 = a[0]
    p2 = a[1]
    mat = m.getSimMatrixSent(p1, p2, sentence_aligner)
    mat_history = []
    for line in mat:
        mat_history.append(max(line))
        if max(line) > 0:
            sent_history.append(max(line))


(low, high) = st.t.interval(0.95, len(sent_history)-1, loc=np.mean(sent_history), scale=st.sem(sent_history))

hard_treshold_s = low-statistics.stdev(sent_history)
soft_threshold_s = max(min(sent_history),0)
certain_threshold_s = high+statistics.stdev(sent_history)

#Reintialize aligner with the thresholds determined earlier and align sentences
sentence_aligner = ExpandingAlingmentSentenceAligner(similarity_model=model, certain_threshold=certain_threshold_s, hard_threshold=hard_treshold_s, soft_threshold=soft_threshold_s)
sentence_alignments = []
for a in aligned_paragraphs:
        p1 = a[0]
        p2 = a[1]
        alignments, aligned_sentences = m.getSentenceAlignments(p1, p2, sentence_aligner)
        sentence_alignments.append((alignments, aligned_sentences))

par_tuple_list = zip(alignments_par, sentence_alignments)


#Print aligned sentences
for alignment, sentences in sentence_alignments:
        for sent in zip(alignment, sentences):
                print(sent[0])
                print(sent[1][0])
                print(sent[1][1])
                print('\n')
        print('\n\n\n')

  