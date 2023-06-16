import pandas
from massalign.core import *
import os
from massalign.evaluate import evaluate
from massalign.params import *
import statistics
import scipy.stats as st
from time import time

def compute_sent_aligner_threshold(m, aligned_paragraphs, sentence_aligner):
	# Compute sentence aligner thresholds based on the similarity matrix
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

	(low, high) = st.t.interval(0.95, len(sent_history) - 1, loc=np.mean(sent_history), scale=st.sem(sent_history))

	hard_treshold_s = low - statistics.stdev(sent_history)
	soft_threshold_s = max(min(sent_history), 0)
	certain_threshold_s = high + statistics.stdev(sent_history)
	return hard_treshold_s, soft_threshold_s, certain_threshold_s


def align_sentences_of_files_tfidf(file1, file2):

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

	#Align sentences in each pair of aligned paragraphs:
	alignmentsl = []
	for a in aligned_paragraphs:
		p1 = a[0]
		p2 = a[1]
		alignments, aligned_sentences = m.getSentenceAlignments(p1, p2, sentence_aligner)
		alignmentsl.append(aligned_sentences)
	return alignmentsl

def align_sentences_of_files_doc2vec(file1, file2, vector_size=None, window=None, min_count=None, epochs=None,
							 infer_epochs=None):
	#Get MASSA aligner for convenience:
	m = MASSAligner()

	#Get sentence aligner:
	model = D2VModel(vector_size=vector_size, window_size=window, min_count=min_count, epochs=epochs,
						 infer_epochs=infer_epochs, input_files=[file1, file2])
	paragraph_aligner = ExpandingAlingmentParagraphAligner(similarity_model=model)
	sentence_aligner = ExpandingAlingmentSentenceAligner(similarity_model=model)



	#Get paragraphs from the document:
	p1s = m.getParagraphsFromDocument(file1)
	p2s = m.getParagraphsFromDocument(file2)

	# #Align paragraphs:
	# alignments, aligned_paragraphs = m.getParagraphAlignments(p1s, p2s, paragraph_aligner)
	aligned_paragraphs = [[p1s[0], p2s[0]]]
	hard_treshold_s, soft_threshold_s, certain_threshold_s = compute_sent_aligner_threshold(m, aligned_paragraphs, sentence_aligner)
	sentence_aligner = ExpandingAlingmentSentenceAligner(similarity_model=model, certain_threshold=certain_threshold_s,
														 hard_threshold=hard_treshold_s,
														 soft_threshold=soft_threshold_s)

	#Align sentences in each pair of aligned paragraphs:
	alignmentsl = []
	for a in aligned_paragraphs:
		p1 = a[0]
		p2 = a[1]
		alignments, aligned_sentences = m.getSentenceAlignments(p1, p2, sentence_aligner)
		alignmentsl.append(aligned_sentences)
	return alignmentsl

# def merge_n_to_1(complex_lines, simple_lines):
# 	output_complex = list()
# 	output_simple = list()
# 	i_simple = 0
# 	i_complex = 0
# 	while i_complex < len(complex_lines) and i_simple < len(simple_lines):
# 		complex_sent = complex_lines[i_complex].strip()
# 		j_complex = 0
# 		while i_complex+ j_complex < len(complex_lines) and complex_sent == complex_lines[i_complex+j_complex].strip():
# 			j_complex += 1
# 		if j_complex != 1:
# 			output_complex.append(" ".join([sent.strip() for sent in complex_lines[i_complex]]))
# 		else:
# 			output_complex.append(" ".join([sent.strip() for sent in complex_lines[i_complex:i_complex + j_complex]]))
# 		output_simple.append(" ".join([sent.strip() for sent in simple_lines[i_simple:i_simple + j_complex]]))
# 		i_simple = i_simple + j_complex
# 		i_complex = i_complex + j_complex
# 	return output_complex, output_simple
#
#
# def merge_n_to_n(complex_filename, simple_filename):
# 	# todo problem n:m currently only 1:n and n:1
# 	with open(complex_filename) as f:
# 		complex_lines = f.readlines()
# 	with open(simple_filename) as f:
# 		simple_lines = f.readlines()
# 	output_complex, output_simple = merge_n_to_1(complex_lines, simple_lines)
# 	output_complex, output_simple = merge_n_to_1(output_simple, output_complex)
# 	with open(complex_filename.split(".")[0]+"_complex_clean.txt", "w") as f:
# 		f.write("\n".join(output_complex))
# 	with open(simple_filename.split(".")[0]+"_simple_clean.txt", "w") as f:
# 		f.write("\n".join(output_simple))
# 	return complex_filename.split(".")[0]+"_complex_clean.txt", simple_filename.split(".")[0]+"_simple_clean.txt"


def align_and_evaluate(method_name="tfidf", without_identical=False, vector_size=None, window=None, min_count=None, epochs=None,
							 infer_epochs=None, csv_path=None, input_path="dataset/documents/", output_path="dataset/", evaluate_data=True):
	start_time = time()
	result_original = list()
	result_simplification = list()
	meta_cols = ["pair_id", "complex_document_id", "simple_document_id", "domain", "corpus", "simple_url", "complex_url",
		 "simple_level", "complex_level", "simple_location_html", "complex_location_html", "simple_location_txt",
		 "complex_location_txt", "alignment_location", "simple_author", "complex_author", "simple_title",
		 "complex_title", "license", "last_access", "language_level_original", "license_summary"]
	results_meta = ["\t".join(meta_cols)]

	all_files = [filename for filename in os.listdir(input_path) if filename.endswith(".src")]
	df = pandas.read_csv(csv_path)
	df[['pair_id', 'x']] = df['simple_document_id'].str.split('-', 1, expand=True)
	df_columns = df.columns

	for filename in all_files:
		# print(filename)
		file1 = input_path+filename
		file2 = input_path+filename[:-4]+".tgt"
		pair_id = filename[:-4].split("_")[-1]
		doc_row = df[df["pair_id"]==pair_id].iloc[0]  # .tolist()
		if method_name == "tfidf":
			aligned_par_sents = align_sentences_of_files_tfidf(file1, file2)
		elif method_name == "doc2vec":
			aligned_par_sents = align_sentences_of_files_doc2vec(file1, file2, vector_size=vector_size, window=window,
													 min_count=min_count, epochs=epochs, infer_epochs=infer_epochs)
		else:
			aligned_par_sents = None

		for par_sents in aligned_par_sents:
			for pair in par_sents:
				if without_identical:
					if pair[0].strip() != pair[1].strip():
						result_original.append(pair[0])
						result_simplification.append(pair[1])
						# results_meta.append(pair_id)

						output_row = list()
						for col in meta_cols:
							if col in df_columns:
								output_row.append(doc_row[col])
							else:
								output_row.append("")
						results_meta.append("\t".join([str(item) for item in output_row]))
				else:
					result_original.append(pair[0])
					result_simplification.append(pair[1])
					# results_meta.append(pair_id)
					output_row = list()
					for col in meta_cols:
						if col in df_columns:
							output_row.append(doc_row[col])
						else:
							output_row.append("")
					results_meta.append("\t".join([str(item) for item in output_row]))

	if without_identical:
		with open(output_path+method_name+"_without.complex", "w") as f:
			f.write("\n".join(result_original))
		with open(output_path+method_name+"_without.simple", "w") as f:
			f.write("\n".join(result_simplification))
		with open(output_path+method_name+"_without.meta", "w") as f:
			f.write("\n".join(results_meta))
	else:
		with open(output_path+method_name+".complex", "w") as f:
			f.write("\n".join(result_original))
		with open(output_path+method_name+".simple", "w") as f:
			f.write("\n".join(result_simplification))
		with open(output_path+method_name+".meta", "w") as f:
			f.write("\n".join(results_meta))

	# # simple_f, complex_f  = merge_n_to_n("dataset/"+method_name+".complex", "dataset/"+method_name+".simple")
	# # print("n:m", "without_identical", without_identical, evaluate("dataset/gold_data.src", "dataset/gold_data.tgt", complex_f, simple_f, without_identical=without_identical))
	# # complex_f, simple_f = merge_n_to_n("dataset/" + "dummy" + ".complex", "dataset/" + "dummy" + ".simple")
	# # print(evaluate("dataset/dummy.complex", "dataset/dummy.complex", complex_f, simple_f))
	print("--- %s seconds ---" % (time() - start_time))
	if evaluate_data:
		return evaluate("dataset/gold_data.src", "dataset/gold_data.tgt", output_path+method_name+".complex", output_path+method_name+".simple", without_identical=without_identical)
	else:
		return 1

# print("tfidf", "default", "without identical", False, align_and_evaluate("tfidf"))
# print("tfidf", "default", "without identical", True,  align_and_evaluate("tfidf", without_identical=True))
# vector_size = 300
# window = 15
# min_count = 3
# epochs = 60
# infer_epochs = 60
# print("doc2vec", align_and_evaluate("doc2vec", vector_size=vector_size, window=window, min_count=min_count, epochs=epochs, infer_epochs=infer_epochs))

print("tfidf", "default", "without identical", False, align_and_evaluate("tfidf", csv_path="DEplain-web-auto-private/all.csv", input_path="DEplain-web-auto-private/documents/", output_path="DEplain-web-auto-private/", without_identical=True, evaluate_data=False))
print("tfidf", "default", "without identical", False, align_and_evaluate("tfidf", csv_path="DEplain-web-auto-public/all.csv", input_path="DEplain-web-auto-public/documents/", output_path="DEplain-web-auto-public/", without_identical=True, evaluate_data=False))
print("tfidf", "default", "without identical", False, align_and_evaluate("tfidf", csv_path="DEplain-web-manual-private/test.csv", input_path="DEplain-web-manual-private/documents/", output_path="DEplain-web-manual-private/", without_identical=True, evaluate_data=False))
print("tfidf", "default", "without identical", False, align_and_evaluate("tfidf", csv_path="DEplain-web-manual-public/test.csv", input_path="DEplain-web-manual-public/documents/", output_path="DEplain-web-manual-public/", without_identical=True, evaluate_data=True))

print("tfidf", "default", "with identical", True, align_and_evaluate("tfidf", csv_path="DEplain-web-auto-private/all.csv", input_path="DEplain-web-auto-private/documents/", output_path="DEplain-web-auto-private/", without_identical=False, evaluate_data=False))
print("tfidf", "default", "with identical", True, align_and_evaluate("tfidf", csv_path="DEplain-web-auto-public/all.csv", input_path="DEplain-web-auto-public/documents/", output_path="DEplain-web-auto-public/", without_identical=False, evaluate_data=False))
print("tfidf", "default", "with identical", True, align_and_evaluate("tfidf", csv_path="DEplain-web-manual-private/test.csv", input_path="DEplain-web-manual-private/documents/", output_path="DEplain-web-manual-private/", without_identical=False, evaluate_data=False))
print("tfidf", "default", "with identical", True, align_and_evaluate("tfidf", csv_path="DEplain-web-manual-public/test.csv", input_path="DEplain-web-manual-public/documents/", output_path="DEplain-web-manual-public/", without_identical=False, evaluate_data=True))
