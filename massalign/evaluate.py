def evaluate(gold_src, gold_tgt, out_complex, out_simpl, without_identical=False):
	correct, n_out, n_gold = get_num_correct_aligns(gold_src, gold_tgt,
													out_complex, out_simpl, without_identical=without_identical)

	precision = correct / n_out
	recall = correct / n_gold

	print((2 * precision * recall), (precision + recall), precision, recall, n_out, n_gold, correct)
	f1 = (2 * precision * recall) / (precision + recall)

	return precision, recall, n_gold, n_out, correct, f1


def get_num_correct_aligns(gold_src, gold_tgt, out_complex, out_simpl, without_identical=False):
	with open(gold_src, 'r') as gold_src_file:
		gold_complex_sents = gold_src_file.readlines()

	with open(gold_tgt, 'r') as gold_tgt_file:
		gold_simpl_sents = gold_tgt_file.readlines()

	gold_complex_sents_cln = []
	for lin in gold_complex_sents:
		if '.eoa' in lin:
			pass
		else:
			gold_complex_sents_cln.append(lin.replace('.eoa', '').strip())

	gold_simpl_sents_cln = []
	for lin in gold_simpl_sents:
		if '.eoa' in lin:
			pass
		else:
			gold_simpl_sents_cln.append(lin.replace('.eoa', '').strip())

	if len(gold_simpl_sents_cln) != len(gold_complex_sents_cln):
		raise ValueError("Wrong input, gold files have different length of content.")

	with open(out_complex, 'r') as out_complex_file:
		out_complex_sents = out_complex_file.readlines()

	with open(out_simpl, 'r') as out_simpl_file:
		out_simpl_sents = out_simpl_file.readlines()

	out_complex_sents_cln = []
	for lin in out_complex_sents:
		if '.eoa' in lin:
			pass
		else:
			out_complex_sents_cln.append(lin.replace('.eoa', '').strip())

	out_simpl_sents_cln = []
	for lin in out_simpl_sents:
		if '.eoa' in lin:
			pass
		else:
			out_simpl_sents_cln.append(lin.replace('.eoa', '').strip())

	if len(out_complex_sents_cln) != len(out_simpl_sents_cln):
		raise ValueError("Wrong input, output files have different length of content. Complex: "+str(len(out_complex_sents_cln))+", Simple: "+str(len(out_simpl_sents_cln)))

	golds = list(zip(gold_complex_sents_cln, gold_simpl_sents_cln))
	# if len(gold_complex_sents_cln) != len(golds.keys()):
	# 	print("gold complex:", len(gold_complex_sents_cln), "dictionary:", len(golds.keys()))
	# 	raise KeyError("Problem with method, dictionary has different size than input.")
	correct = 0
	aligned = len(out_complex_sents_cln)
	gold_aligned = len(gold_complex_sents_cln)
	if without_identical:
		for gold_complex, gold_simple in zip(gold_complex_sents_cln, gold_simpl_sents_cln):
			if gold_complex == gold_simple:
				gold_aligned -= 1

	for out_cmplx, out_simpl in zip(out_complex_sents_cln, out_simpl_sents_cln):
		if without_identical and out_cmplx == out_simpl:
			aligned -= 1
			continue
		if (out_cmplx, out_simpl) in golds:
			# print(out_simpl, '-----------', golds[out_cmplx])
			#if out_simpl == golds[out_cmplx]:
				# print(out_simpl, '-----------', golds[out_cmplx])
			correct += 1

	return correct, aligned, gold_aligned
