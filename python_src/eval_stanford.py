#!/usr/bin/python
import codecs as cs
import os
import sys
import time
import uuid
from collections import defaultdict
from concrete.util.file_io import *
from concrete.util import generate_UUID
from concrete.inspect import *
from concrete import (
	AnnotationMetadata,
	TextSpan,
	Token,
	Tokenization,
	TokenizationKind,
	TokenList,
	TokenTagging,
	TaggedToken
	)


OOV = '_OOV_'
GOLD_TAG = 'GoldNER'
PRED_TAG = 'NER'

def evaluate_stanford(input_tar, annotated_tar):
    gold_array = []
    anno_dict = {}
    gold_tags = []
    pred_tags = []
    for comm, file_name in CommunicationReader(input_tar):
	    #print 'processing', comm.id
	    gold_array.append((comm, comm.id))
    print 'finished load gold data!'
    for comm, file_name in CommunicationReader(annotated_tar):
	    #print 'processing', os.path.splitext(file_name)[0]
	    anno_dict[os.path.splitext(file_name)[0]]=comm
    print 'finished load annotated data!'
    for comm, uuid in gold_array:
	    anno_comm = anno_dict[uuid]
	    gold_tags.extend(read_concrete(comm, GOLD_TAG))
	    pred_tags.extend(read_concrete(anno_comm, PRED_TAG))
    assert len(pred_tags)==len(gold_tags)
    eval_ner(pred_tags, gold_tags)

def read_concrete(comm, tag_name):
    #comm = read_communication_from_file(concrete_file)
    tokenizations = get_tokenizations(comm)
    all_tags = []
    ne_preceeded = None
    pre_ne = None
    for tokenization in tokenizations:
    	tags = get_tokentaggings_of_type(tokenization, tag_name)
	if tag_name == GOLD_TAG:
	    ctags = convert_tags_gold(tokenization, tags[0])
        else:
	    ctags, ne_preceeded, pre_ne = convert_tags_stanford(tokenization, tags[0], ne_preceeded, pre_ne)
	all_tags.extend(ctags)
    return all_tags 


def convert_tags_gold(tokenization, token_tags):
    result_tags = []
    if tokenization.tokenList:
	for i, token in enumerate(tokenization.tokenList.tokenList):
	    token_tag = token_tags.taggedTokenList[i] #[unicode(token_tag_list[i]) for token_tag_list in token_tags]
	    tags = token_tag.tag.split(' ')
	    assert len(tags) == len(token.text)
   	    result_tags.extend(tags)
    return result_tags


def convert_tags_stanford(tokenization, token_tags, ne_preceeded, pre_ne=None):
    result_tags = []
    if tokenization.tokenList:
	for i, token in enumerate(tokenization.tokenList.tokenList):
	    token_tag = token_tags.taggedTokenList[i].tag #[unicode(token_tag_list[i]) for token_tag_list in token_tags]
	    tags = ['O'] * len(token.text)
	    #print token_tag
	    if token_tag == 'PERSON' or token_tag == 'LOC' or token_tag == 'ORG': #or token_tag == 'GPE':
		    print 'recognized entity:', token_tag
		    if ne_preceeded and pre_ne == token_tag:
			    tags = ['I-'+token_tag[:3]]*len(token.text)
		    else:
			    tags = ['B-'+token_tag[:3]] + ['I-'+ token_tag[:3]]*(len(token.text)-1)
		    ne_preceeded = True
		    pre_ne = token_tag
		    print 'tags:', tags
            else:
		    ne_preceeded = None
	    assert len(tags) == len(token.text)
   	    result_tags.extend(tags)
    return result_tags, ne_preceeded, pre_ne


# utilities for evaluation:
def eval_ner(pred, gold):
	eval_dict = {}    # value=[#match, #pred, #gold]
	in_correct_chunk = False
	last_pair = ['^', '$']
	for p, g in zip(pred, gold):
		tp = p.split('-')
		tg = g.split('-')
		if len(tp) == 2:
			if tp[1] not in eval_dict:
				eval_dict[tp[1]] = [0]*3
			if tp[0] == 'B':
				eval_dict[tp[1]][1] += 1
		if len(tg) == 2:
			if tg[1] not in eval_dict:
				eval_dict[tg[1]] = [0]*3 
			if tg[0] == 'B':
				eval_dict[tg[1]][2] += 1
		# hit a row outside of a contineous NE chunk
		# either p == g == 'O' or p != g
		# in this case, it does not enter another NE chunk, change in_correct_chunk to false and change last_pair to default different value. 
		# check whether previous block could be a correctly predicted NE chunk.
		if p != g or len(tp) == 1:
			if in_correct_chunk and tp[0] != 'I' and tg[0] != 'I':
				assert last_pair[0] == last_pair[1]
				eval_dict[last_pair[0]][0] += 1
			in_correct_chunk = False
			last_pair = ['^', '$'] 
		# p == g and p == g != 'O'
		else:
			# start a new chunk
			if tg[0] == 'B':
				if in_correct_chunk:
					assert (last_pair[0] == last_pair[1])
					eval_dict[last_pair[0]][0] += 1
				in_correct_chunk = True
				last_pair = [tp[-1], tg[-1]]
	if in_correct_chunk:
	    assert last_pair[0] == last_pair[1]
	    eval_dict[last_pair[0]][0] += 1
	agg_measure = [0.0]*3
	agg_counts = [0]*3
	print eval_dict
	#print 'violet\'s evaluation: #match\t #pred\t #gold\t precision\t recall\t F1'
	for k, v in eval_dict.items():
		agg_counts = [sum(x) for x in zip(agg_counts, v)]
		prec = float(v[0])/v[1] if v[1] != 0 else 0.0 
		recall = float(v[0])/v[2] if v[2] != 0 else 0.0
		F1 = 2*prec*recall/(prec+recall) if prec != 0 and recall != 0 else 0.0
		agg_measure[0] += prec
		agg_measure[1] += recall
		agg_measure[2] += F1
		print k+':', v[0], '\t', v[1], '\t', v[2], '\t', prec, '\t', recall, '\t', F1
	agg_measure = [v/len(eval_dict) for v in agg_measure]
	print 'Macro average:', '\t', agg_measure[0], '\t', agg_measure[1], '\t', agg_measure[2]
	prec = float(agg_counts[0])/agg_counts[1] if agg_counts[1] != 0 else 0.0
	recall = float(agg_counts[0])/agg_counts[2] if agg_counts[2] != 0 else 0.0
	F1 = 2*prec*recall/(prec+recall) if prec != 0 and recall != 0 else 0.0
	print 'Micro average:', agg_counts[0], '\t', agg_counts[1], '\t', agg_counts[2], '\t', prec, '\t', recall, '\t', F1 
	return {'p': prec, 'r': recall, 'f1': F1} 


if __name__ == '__main__':
	evaluate_stanford(sys.argv[1], sys.argv[2])
