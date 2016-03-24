#/user/local/bin/python
# -*- coding: utf-8 -*-

from concrete.util.file_io import CommunicationReader
from concrete.inspect import * 
import sys
import codecs as cs


def convert_concrete_to_conll(infilename, outfilename):
	reader = CommunicationReader(infilename)
	with cs.open(outfilename, 'w', encoding='utf-8') as outf:
		for (comm, _) in reader:
			comm_annotations = annotation_to_conll(comm)
			outf.write('id:'+comm.id+'\n')
			for line in comm_annotations:
				outf.write(line+'\n')
			outf.write('\n')


def annotation_to_conll(comm):
	tokenizations = get_tokenizations(comm)
	#print len(tokenizations)
	comm_annotations = []
	for tokenization in tokenizations:
		predict_ner_tags = get_ner_tags_for_tokenization(tokenization)
		gold_ner_tags = get_goldNER_for_tokenization(tokenization)
		sent_annotations = recover_annotation(tokenization.tokenList.tokenList, gold_ner_tags, predict_ner_tags)	
		comm_annotations.extend(sent_annotations)
		#print ner_tags
		#print tokenization.tokenList.tokenList
	return comm_annotations


def get_goldNER_for_tokenization(tokenization):
	if tokenization.tokenList:
		ner_tags = [""]*len(tokenization.tokenList.tokenList)
		ner_tokentaggings = get_tokentaggings_of_type(tokenization, u"GoldNER")
		if ner_tokentaggings and len(ner_tokentaggings) > 0:
			tag_for_tokenIndex = {}
			for taggedToken in ner_tokentaggings[0].taggedTokenList:
				tag_for_tokenIndex[taggedToken.tokenIndex] = taggedToken.tag
			for i, token in enumerate(tokenization.tokenList.tokenList):
				try:
					ner_tags[i] = tag_for_tokenIndex[i]
				except IndexError:
					ner_tags[i] = u""
				if ner_tags[i] == u"NONE":
					ner_tags[i] = u""
		return ner_tags
	

def recover_annotation(tokenlist, goldNER, predictNER):
	annotation_array = []
	for token, gold_tag, pred_tag in zip(tokenlist, goldNER, predictNER):
		token_text = token.text
		gold_tag_array = gold_tag.split(' ')
		for i, tk in enumerate(token_text):   #.decode('utf-8')
			prefix = ('I-' if i==0 else 'B-')
			predict_tag = ("O" if len(pred_tag) == 1 else prefix+pred_tag[:3]+'.NAM')
			oneline = [tk, gold_tag_array[i], predict_tag]
			annotation_array.append('\t'.join(oneline))
	return annotation_array



if __name__ == '__main__':
	convert_concrete_to_conll(sys.argv[1], sys.argv[2])
