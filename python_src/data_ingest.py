#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import time
import codecs as cs
import re
import os
import jieba
import uuid
from concrete.inspect import get_tokenizations
from concrete import (
	AnnotationMetadata,
	Communication,
	Section,
	Sentence,
	TextSpan,
	Token,
	Tokenization,
	TokenizationKind,
	TokenList,
	TokenTagging,
	TaggedToken
	)
from concrete.util import generate_UUID
from concrete.util.file_io import *  #CommunicationWriterTGZ


GOLD_TAG = 'GoldNER'
def write_comms_to_tarfile(tarfile_name, comm_array):
	writer = CommunicationWriterTGZ()
	writer.open(tarfile_name)
	for i, comm in enumerate(comm_array):
		writer.write(comm, 'document_'+str(i)+'.comm')
	writer.close()


def convert_labels_more(infile, outfile):
	with cs.open(infile, 'r', encoding='utf-8') as inf, cs.open(outfile, 'w', encoding='utf-8') as outf:
		lines = []
		for line in inf:
			if line.strip() == '':
				if lines[-1][1] == 'B':
					lines[-1][1] = 'S'
				elif lines[-1][1] == 'I':
					lines[-1][1] = 'E'
				lines.append(line.strip())
			else :
				elems = re.split('[- ]+', line)
				if elems[1] == 'B' or elems[1][0] == 'O':
					if len(lines) == 0 or lines[-1] == '':
						pass
					elif lines[-1][1] == 'B':
						lines[-1][1] = 'S'
					elif lines[-1][1] == 'I':
						lines[-1][1] = 'E'
				lines.append(elems)
		for elems in lines:
			if elems == '':
				outf.write('\n')
			elif len(elems) == 2:
				outf.write(' '.join(elems))
			else:
				outf.write(' '.join([elems[0], '-'.join(elems[1:])]))


def convert_labels_less(infile, outfile):
	with cs.open(infile, 'r', encoding='utf-8') as inf, cs.open(outfile, 'w', encoding='utf-8') as outf:
		lines = []
		for line in inf:
			elems = re.split('[- ]+', line)
			if elems[-2] == 'S':
				elems[-2] = 'B'
			elif elems[-2] == 'E':
				elems[-2] = 'I'
			else:
				outf.write(line)
				continue
			if len(elems) == 5:
				outf.write(' '.join([elems[0], '-'.join(elems[1:3]), '-'.join(elems[3:])]))
			else:
				assert len(elems) == 4
				outf.write(' '.join([elems[0], elems[1], '-'.join(elems[2:])]))


def convert_word_representation(infile, outfile, segment=False, charpos=False, binarize=False):
	with cs.open(infile, 'r', encoding='utf-8') as inf, cs.open(outfile, 'w', encoding='utf-8') as outf:
		char_arry = []
		tag_arry = []
		line_count = 0
		for line in inf:
			if line.startswith("tweet_id"):
				if segment and line_count != 0:
					new_char_arry = []
					sentence = ''.join(char_arry)
					token_list = jieba.cut(sentence)
					for tk in token_list:
						for i, char in enumerate(tk):
							if charpos:
								new_char_arry.append(char+str(i))
							else:
								new_char_arry.append(tk)
					assert len(new_char_arry) == len(tag_arry)
					for char, tag in zip(new_char_arry, tag_arry):
						outf.write(char+'\t'+tag+'\n')
					outf.write('\n')
					char_arry = []
					tag_arry = []
				else:
					continue
			elif len(line.strip().split('\t')) < 2 and line != "\n" or line.startswith("\t"):
				#print line
				continue
			elif segment and line != "\n":
				elems = line.strip().split('\t')
				assert len(elems) == 2
				char_arry.append(elems[0])
				if elems[1].endswith('PRO') or elems[1].endswith('TTL.NA'):  #or elems[1].endswith('NOM') 
					tag_arry.append('O')
				else:
					if binarize and elems[1] != "O":
						tag_arry.append("I")
					else:
						tag_arry.append(elems[1])
			elif not segment:
				if line.rstrip().endswith("PRO") or line.rstrip().endswith("TTL.NA"):  # or line.rstrip().endswith("NOM")
					elems = line.strip().split('\t')
					assert len(elems) == 2
					elems[1] = 'O'
					outf.write('\t'.join(elems)+'\n')
				else:
					if binarize and line.strip() != '':
						elems = line.strip().split('\t')
						if len(elems) != 2:
							print elems
							exit(0)
						if elems[1] != 'O':
							elems[1] = 'I'
						outf.write('\t'.join(elems)+'\n')
					else:
						outf.write(line)
			line_count += 1
		if segment:
			new_char_arry = []
			sentence = ''.join(char_arry)
			token_list = jieba.cut(sentence)
			for tk in token_list:
				for i, char in enumerate(tk):
					if charpos:
						new_char_arry.append(char+str(i))
					else:
						new_char_arry.append(tk)
			assert len(new_char_arry) == len(tag_arry)
			for char, tag in zip(new_char_arry, tag_arry):
				outf.write(char+'\t'+tag+'\n')
			outf.write('\n')


def convert_to_conll(infile, outfile, segment=False, charpos=False, binarize=False):
	with cs.open(infile, 'r', encoding='utf-8') as inf, cs.open(outfile, 'w', encoding='utf-8') as outf:
		char_arry = []
		tag_arry = []
		line_count = 0
		for line in inf:
			if line.startswith("tweet_id"):
				if segment and line_count != 0:
					new_char_arry = []
					sentence = ''.join(char_arry)
					token_list = jieba.cut(sentence)
					for tk in token_list:
						for i, char in enumerate(tk):
							if charpos:
								new_char_arry.append(char+str(i))
							else:
								new_char_arry.append(tk)
					assert len(new_char_arry) == len(tag_arry)
					for char, tag in zip(new_char_arry, tag_arry):
						outf.write(char+'\t'+tag+'\n')
					outf.write('\n')
					char_arry = []
					tag_arry = []
				else:
					continue
			elif len(line.strip().split('\t')) < 2 and line != "\n" or line.startswith("\t"):
				#print line
				continue
			elif segment and line != "\n":
				elems = line.strip().split('\t')
				assert len(elems) == 2
				char_arry.append(elems[0])
				if elems[1].endswith('PRO') or elems[1].endswith('TTL.NA'):  #or elems[1].endswith('NOM') 
					tag_arry.append('O')
				else:
					if binarize and elems[1] != "O":
						tag_arry.append("I")
					else:
						tag_arry.append(elems[1])
			elif not segment:
				if line.rstrip().endswith("PRO") or line.rstrip().endswith("TTL.NA"):  # or line.rstrip().endswith("NOM")
					elems = line.strip().split('\t')
					assert len(elems) == 2
					elems[1] = 'O'
					outf.write('\t'.join(elems)+'\n')
				else:
					if binarize and line.strip() != '':
						elems = line.strip().split('\t')
						if len(elems) != 2:
							print elems
							exit(0)
						if elems[1] != 'O':
							elems[1] = 'I'
						outf.write('\t'.join(elems)+'\n')
					else:
						outf.write(line)
			line_count += 1
		if segment:
			new_char_arry = []
			sentence = ''.join(char_arry)
			token_list = jieba.cut(sentence)
			for tk in token_list:
				for i, char in enumerate(tk):
					if charpos:
						new_char_arry.append(char+str(i))
					else:
						new_char_arry.append(tk)
			assert len(new_char_arry) == len(tag_arry)
			for char, tag in zip(new_char_arry, tag_arry):
				outf.write(char+'\t'+tag+'\n')
			outf.write('\n')


def read_data_to_concrete(infile, separator = '\t'):
	comm_array = []
	with cs.open(infile, 'r', encoding='utf-8') as inf:
		uuid_str = ''
		content_arry = []
		tag_arry = []
		line_num = 0
		for line in inf:
			line_num += 1
			if line.startswith('tweet_id'):
				uuid_str = 'document_'+str(len(comm_array))+'.comm'   #line.strip().split(':')[1]
			elif line.strip() == '':
				content_str = ''.join(content_arry)
				print content_str 
				print tag_arry
				if uuid_str == '':
					uuid_str = 'document_'+str(len(comm_array))+'.comm'  #str(uuid.uuid4())
					print uuid_str
				comm=create_comm_from_sent(uuid_str, content_str, tag_arry)
				del content_arry[:]
				del tag_arry[:]
				uuid_str = ''
				comm_array.append(comm)
			else:
				elems = line.split(separator)
				try:
					assert len(elems) == 2
				except:
					continue	
				content_arry.append(elems[0])
				if elems[1].strip().endswith('PRO') or elems[1].strip().endswith('TTL.NA'):  #or elems[1].endswith('NOM') 
					tag_arry.append('O')
				else:
					tag_arry.append(elems[1].strip())
	return comm_array


def mentionSetList_to_tokenTagging(comm):
	toolname = 'mentionSet to tokenTagging'
	timestamp = int(round(time.time() * 1000))
	entity_mention_list = comm.entityMentionSetList[-1].mentionList
	mention_uuid_dict = {}
	for entity_mention in entity_mention_list:
		uuid = entity_mention.tokens.tokenizationId
		if uuid.uuidString not in mention_uuid_dict:
			mention_uuid_dict[uuid.uuidString] = [(entity_mention.tokens.tokenIndexList, entity_mention.entityType, entity_mention.phraseType)]
		else:
			mention_uuid_dict[uuid.uuidString].append((entity_mention.tokens.tokenIndexList, entity_mention.entityType, entity_mention.phraseType))	
	#print mention_uuid_dict
	tokenizations = get_tokenizations(comm)
	for tknzation in tokenizations:
	    tag_list = []
	    for i, token in enumerate(tknzation.tokenList.tokenList):
		tags = ['O'] * len(token.text)
		tag_list.append(tags)
	    if tknzation.uuid.uuidString in mention_uuid_dict:
		    for (token_idx_list, entity_type, phrase_type) in mention_uuid_dict[tknzation.uuid.uuidString]:
			    #print 'annotated mention!', token_idx_list
			    for i, idx in enumerate(token_idx_list):
				    length = len(tag_list[idx])
				    if tag_list[idx][0] != 'O':
					    continue
				    if i == 0:
					    tag_list[idx] = ['B-'+entity_type[:3]+'.'+('NAM' if phrase_type==None else phrase_type)] + ['I-'+entity_type[:3]+'.'+('NAM' if phrase_type==None else phrase_type)] * (length-1)
				    else:
					    tag_list[idx] = ['I-'+entity_type[:3]+'.'+('NAM' if phrase_type==None else phrase_type)] * length
	    token_tags = []
	    for i, tags in enumerate(tag_list):
		token_tags.append(TaggedToken(tokenIndex=i, tag=' '.join(tags)))
	    ner_tokentagging = TokenTagging(
			taggingType=GOLD_TAG,
			taggedTokenList=token_tags,
			metadata=AnnotationMetadata(tool=toolname, timestamp=timestamp),
			uuid=generate_UUID())
	    if tknzation.tokenTaggingList == None:
		tknzation.tokenTaggingList = [ner_tokentagging]
	    else:
	    	tknzation.tokenTaggingList.append(ner_tokentagging)
	return comm


# New version of data ingestion. No punctuation replacement anymore, each tweet is a sentence(?).
def create_comm_from_sent(comm_id, content_str, tag_arry):
	toolname = 'create_concrete_sent_from_weiboNER_jieba'
	timestamp = int(time.time())
	sent_end = u'[\.!?。？！…；]+'
	# Note: here sent_end_set and sent_end(regex) are made different intentionally
	sent_end_set = set([u'.', u'!', u'?', u'。', u'？', u'！', u'…', u'；'])
	comm = Communication(
		id=comm_id,
		metadata=AnnotationMetadata(tool=toolname, timestamp=timestamp),
		text=content_str,
		type=toolname,
		uuid=generate_UUID())
	sentences = []
	offsetInContent=0
	for sent_str in re.split(sent_end, content_str):
		if len(sent_str) == 0:
			continue
		try: 
			sstart = content_str.index(sent_str[0], offsetInContent)
		except:
			print 'cannot find sentence substring!!!' 
			print 'sentence:', sent_str
			print 'section:', content_str
		while sstart+len(sent_str) < len(content_str) and content_str[sstart+len(sent_str)] in sent_end_set:
			sent_str+=content_str[sstart+len(sent_str)]
		send = sstart + len(sent_str)
		offsetInContent = send
		#print 'sentence:', sent_str, 'start=', sstart, 'end=', send
		token_string_list = jieba.cut(sent_str)
		token_num = 0
		tokens = []
		token_tags = []
		offsetInSent = sstart
		#print 'processing sentence:', sent_str, 'offsetInSent:', offsetInSent
		for token_string in token_string_list: 
			if token_string == '':
				continue
			try:
				startIdx = content_str.index(token_string[0], offsetInSent)
			except:
				print 'cannot find token substring!!!' 
				print 'token:', token_string, 'offset:', offsetInSent
				print 'sentence:', sent_str
			endIdx = startIdx
			tempIdx = 0
			while endIdx < send and tempIdx < len(token_string):
				if token_string[tempIdx] == content_str[endIdx]:
					tempIdx += 1
				endIdx += 1
			#print 'processing token:', token_string, 'start idx=', startIdx, 'end idx=', endIdx
			tokens.append(Token(text=token_string, tokenIndex=token_num, textSpan=TextSpan(startIdx, endIdx)))
			token_tags.append(TaggedToken(tokenIndex=token_num, tag=' '.join(tag_arry[startIdx: endIdx])))
			offsetInSent = endIdx
			token_num += 1
		ner_tokentagging = TokenTagging(
			taggingType=GOLD_TAG,
			taggedTokenList=token_tags,
			metadata=AnnotationMetadata(tool=toolname, timestamp=timestamp),
			uuid=generate_UUID())
		tokenization = Tokenization(
			kind=TokenizationKind.TOKEN_LIST,
			metadata=AnnotationMetadata(tool=toolname, timestamp=timestamp),
			tokenList=TokenList(tokenList=tokens),
			tokenTaggingList=[ner_tokentagging],
			uuid=generate_UUID())
		sentence = Sentence(
			textSpan=TextSpan(sstart, send),
			tokenization=tokenization,
			uuid=generate_UUID())
		sentences.append(sentence)
	section = Section(
		kind="SectionKind",
		sentenceList=sentences,
		textSpan=TextSpan(0, len(content_str)),
		uuid=generate_UUID())
	comm.sectionList=[section]
	return comm

def quick_split(infile):
	for split in range(1,5):
	    with cs.open(infile, 'rb', encoding='utf-8') as inf, cs.open(infile+'.train.'+str(split), 'wb', encoding='utf-8') as trf, cs.open(infile+'.dev.'+str(split), 'wb', encoding='utf-8') as df, cs.open(infile+'.test.'+str(split), 'wb', encoding='utf-8') as tf:
		    count = 1
		    dev_test = 1
		    for line in inf:
			    if count % 5 < split:
				    trf.write(line)
			    else:
				    if dev_test % 2 == 0:
				    	tf.write(line)
				    else:
				        df.write(line)
			    if line.strip() == '':
				    count += 1
				    if count % 5 >= split:
				    	dev_test += 1


if __name__ == '__main__':
	writer = CommunicationWriterTGZ()
	writer.open(os.path.join(sys.argv[2], os.path.basename(sys.argv[1])))
	print os.path.join(sys.argv[2], os.path.basename(sys.argv[1]))
	for comm, filename in CommunicationReader(sys.argv[1]):
		mentionSetList_to_tokenTagging(comm)
		writer.write(comm, filename)
	writer.close()
	#comm_array = read_data_to_concrete(sys.argv[1], separator = '\t')
	#write_comms_to_tarfile(sys.argv[2], comm_array)
	
	#convert_labels_more(sys.argv[1], sys.argv[2])
	#func = eval(sys.argv[3])
	#func(sys.argv[1], sys.argv[2])
	#input_file, output_file, segment, charpos, binarize
	#convert_to_conll(sys.argv[1], sys.argv[2], sys.argv[3]=='true', sys.argv[4]=='true', sys.argv[5]=='true')
	#quick_split(sys.argv[1])
