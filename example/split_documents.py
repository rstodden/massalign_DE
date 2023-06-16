import os
import pandas as pd

deplain_web_auto_private = pd.read_csv("DEplain-web-auto-private/all.csv")
deplain_web_auto_public = pd.read_csv("DEplain-web-auto-public/all.csv")
deplain_web_manual_private = pd.read_csv("DEplain-web-manual-private/test.csv")
deplain_web_manual_public = pd.read_csv("DEplain-web-manual-public/test.csv")

print(deplain_web_auto_private.head())

def split_documents(data, output_path):
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	for i, row in data[~(data["original"].isna()) & ~(data["simplification"].isna())].iterrows():
		print(row["complex_document_id"], row["simple_document_id"])
		complex_sentences = row["original"].strip().split("|||")
		simple_sentences = row["simplification"].strip().split("|||")
		# for n, sent in complex_sentences:
		pair_id = row["complex_document_id"].split("-")[0]
		with open(output_path+"file_"+str(pair_id)+".src", "w") as f:
			f.write("\n".join([sent.strip() for sent in complex_sentences]))
		with open(output_path+"file_"+str(pair_id)+".tgt", "w") as f:
			f.write("\n".join([sent.strip() for sent in simple_sentences]))
		# with open(output_path+"file_"+str(i)+".src_meta", "w") as f:
		# 	f.write("\n".join([row["complex_document_id"] for sent in complex_sentences]))
		# with open(output_path+"file_"+str(i)+".tgt_meta", "w") as f:
		# 	f.write("\n".join([row["simple_document_id"] for sent in simple_sentences]))
	return 1


split_documents(deplain_web_auto_private, "DEplain-web-auto-private/documents/")
split_documents(deplain_web_auto_public, "DEplain-web-auto-public/documents/")
split_documents(deplain_web_manual_private, "DEplain-web-manual-private/documents/")
split_documents(deplain_web_manual_public, "DEplain-web-manual-public/documents/")