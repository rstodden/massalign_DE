from massalign.core import *

#Get files to align:
file1 = "sample_data/test_document_complex_2.txt"  # 'https://ghpaetzold.github.io/massalign_data/complex_document.txt'
file2 = "sample_data/test_document_simple_2.txt"  # 'https://ghpaetzold.github.io/massalign_data/simple_document.txt'

#Train model over them:
model = TFIDFModel([file1, file2])

#Get paragraph aligner:
paragraph_aligner = VicinityDrivenParagraphAligner(similarity_model=model, acceptable_similarity=0.3)

#Get sentence aligner:
sentence_aligner = VicinityDrivenSentenceAligner(similarity_model=model, acceptable_similarity=0.2, similarity_slack=0.05)

#Get MASSA aligner for convenience:
m = MASSAligner()

#Get paragraphs from the document:
p1s = m.getParagraphsFromDocument(file1)
p2s = m.getParagraphsFromDocument(file2)

#Align paragraphs:
alignments, aligned_paragraphs = m.getParagraphAlignments(p1s, p2s, paragraph_aligner)
print("PAR", aligned_paragraphs)

#Align sentences in each pair of aligned paragraphs:
alignmentsl = []
for a in aligned_paragraphs:
        p1 = a[0]
        p2 = a[1]
        alignments, aligned_sentences = m.getSentenceAlignments(p1, p2, sentence_aligner)
	print(alignments, aligned_sentences)
        #Display sentence alignments:
        # m.visualizeSentenceAlignments(p1, p2, alignments)
        # m.visualizeListOfSentenceAlignments([p1, p1], [p2, p2], [alignments, alignments])
