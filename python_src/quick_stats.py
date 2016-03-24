#/user/local/bin/python
# -*- coding: utf-8 -*-

import sys
import codecs as cs
import csv
import ast

def unicode_csv_reader(utf8_data, dialect=csv.excel, **kwargs):
	csv_reader = csv.reader(utf8_data, dialect=dialect, **kwargs)
        for row in csv_reader:
	        yield [unicode(cell, 'utf-8') for cell in row]


def load_new_tweet_id(filename):
	tweet_id_set = set()
	with open(filename, 'rb') as inf:
		reader = unicode_csv_reader(inf)
		line_num = 0
		for line in reader:
			line_num += 1
			if line_num == 1:
				for i, c in enumerate(line):
					if c == 'Input.tweets':
						start_num = i
				gold_llen = len(line)
				continue
			assert start_num == 27
			assert len(line) == gold_llen-2
			original_content = line[start_num]
			original_content_map = ast.literal_eval(original_content)
			original_content_map = ast.literal_eval(original_content_map)
			for j in range(1, 11):
				tweet_id = str(original_content_map['tweet_id_'+str(j)])
				tweet_id_set.add(tweet_id)

	return tweet_id_set


def load_old_tweet_id(filename):
	tweet_id_set = set()
	with open(filename, 'rb') as inf:
		reader = unicode_csv_reader(inf)
		line_num = 0
		for line in reader:
			line_num += 1
			if line_num == 1:
				for i, c in enumerate(line):
					if c == 'Input.sentence_1':
						start_num = i
				gold_llen = len(line)
				continue
			assert start_num == 27
			assert len(line) == gold_llen-2
			for j in range(10):
				tweet_id_set.add(line[start_num+10+j])
				#data_map[line[start_num+10+j]] = [line[start_num+j].split(' '), line[field_map['Answer.named_entity_mask_'+str(j+1)]]]
	#print field_map
	return tweet_id_set 



if __name__ == '__main__':
	old_anno_file = sys.argv[1]
	new_anno_file = sys.argv[2]
	old_tweet_idmap = load_old_tweet_id(old_anno_file)
	new_tweet_idmap = load_new_tweet_id(new_anno_file)
	count = 0
	for el in old_tweet_idmap:
		if el not in new_tweet_idmap:
			print el
			count += 1
	print count, len(old_tweet_idmap)
