#/user/local/bin/python
# -*- coding: utf-8 -*-

import sys
import os
import gzip
import codecs as cs
import csv
import re

''' Generate the final hits file.
    @Param: input file dir
    @Param: line number (total line you want to generate)
    @Param: labeled gold data
    Sample usage: python python_src/final_generate.py weibo_all_segmented_sample/ 190 Batch_88873_batch_results.csv > output
'''
def load_all_file(file_dir):
	#symbols = u'[,]+'
	files = os.listdir(file_dir)
	data = []
	for f in files:
		with gzip.open(file_dir+'/'+f) as inf:
			for line in inf:
				#line = re.sub(symbols, '', line.decode('utf-8'))
				data.append(line.decode('utf-8'))
	print 'data size:', len(data)
	return data


def load_labeled_file(filename):
	ldata = []
	line_num = 0
	with cs.open(filename, 'r', encoding='utf-8') as f:
		start_num = 0
		for line in f:
			line_num += 1
			if line_num == 1:
				content = line.split(',')
				for i, c in enumerate(content):
					if c == '\"Input.sentence_1\"':
						start_num = i
				continue
			content = line.split(',')
			#print len(content), content
			#print start_num
			#sys.exit(0)
			assert start_num == 27
			assert len(content) == 481
			#start_num += 2
			for j in range(10):
				ldata.append((content[start_num+2+j][1:-1], content[start_num+12+j][1:-1]))
	print 'labeled data:', ldata, 'size:', len(ldata)
	return ldata

def make_header():
	sent_arry = []
	sent_id_arry = []
	for i in range(1,11):
		sent_arry.append('sentence_'+str(i))
		sent_id_arry.append('sentence_'+str(i)+'_id')
	header = ','.join(sent_arry)+','+','.join(sent_id_arry)
	print 'generated header:', header
	return header

def write_to_file(data_arry, header, filename):
	with cs.open(filename, 'w', encoding='utf-8') as outf:
		outf.write(header+'\n')
		for line in data_arry:
			ostr = ','.join(line)
			outf.write(ostr)


if __name__ == '__main__':
	file_dir = sys.argv[1]
	lin_num = int(sys.argv[2])
	lfile_name = sys.argv[3]
	data = load_all_file(file_dir)
	ldata = load_labeled_file(lfile_name)
	bin_size = len(data)/lin_num
	rotate_gold_size = len(ldata)
	print 'data size=', len(data), 'bin size=', bin_size
	out_data = []
	total_line = 0
	file_num = 0
	header = make_header() 
	for i, line in enumerate(data):
		if i%bin_size == 0:
			elems = line.split(',')
			#print elems
			assert len(elems) == 20
			num = (total_line%(rotate_gold_size/2))*2 
			#print 'padding line', num
			#print 'content', ldata[num], ldata[num+1]
			elems[0] = ldata[num][0]
			elems[1] = ldata[num+1][0]
			elems[10] = ldata[num][1]
			elems[11] = ldata[num+1][1]
			out_data.append(elems)
			total_line += 1
			if len(out_data) == 63:
				outfile_name = 'Chinese_weibo_' + str(file_num) +'.csv'
				write_to_file(out_data, header, outfile_name)
				out_data = []
				file_num += 1
