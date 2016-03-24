#/user/local/bin/python
# -*- coding: utf-8 -*-

import sys
import os
import gzip
import codecs as cs
import csv
import ast
import cStringIO
from collections import Counter


''' Load labeled file into a Map structure:
	key: weibo id.
	value: (text, label, tags)
'''
def load_gold_file(filename):
	field_map = {}
	data_map = {}
	with open(filename, 'rb') as inf:
		reader = unicode_csv_reader(inf)
		line_num = 0
		for line in reader:
			line_num += 1
			if line_num == 1:
				for i, c in enumerate(line):
					if c.startswith('Answer'):
						field_map[c] = i
					if c == 'Input.sentence_1':
						start_num = i
				gold_llen = len(line)
				continue
			#print len(content), content
			#print line_num, len(line)
			#sys.exit(0)
			assert start_num == 27
			assert len(line) == gold_llen-2
			#start_num += 2
			for j in range(10):
				#print type(line[start_num+10+j])
				data_map[line[start_num+10+j]] = [line[start_num+j].split(' '), line[field_map['Answer.named_entity_mask_'+str(j+1)]]]
				#ldata.append((line[start_num+j][1:-1], line[start_num+10+j][1:-1]))
	#print field_map
	print 'labeled data size:', len(data_map)
	#print data_map
	#for k, val in data_map.items():
	#	print k, ''.join(val[0]).encode("utf-8"), val[1]
	return (data_map, field_map) 


def unicode_csv_reader(utf8_data, dialect=csv.excel, **kwargs):
	csv_reader = csv.reader(utf8_data, dialect=dialect, **kwargs)
        for row in csv_reader:
	        yield [unicode(cell, 'utf-8') for cell in row]


def extract_content(text, label):
	token = ''
	pre_label = '0'
	token_array = []
	for i, l in enumerate(label):
		if l == '0' and pre_label != '0':
			token_array.append(token)
			token = ''
		if l == '1':
			if pre_label != '0':
				token_array.append(token)
				token = ''
			else:
				token += text[i]
		if l == '2':
			token += text[i]
		pre_label = l
	return token_array


def load_turker_lfile(filename, gold_idmap):
	field_map = {}
	worker_map = {}
	with open(filename, 'rb') as inf:
		reader = unicode_csv_reader(inf)
		line_num = 0
		#print gold_idmap
		for line in reader:
			line_num += 1
			if line_num == 1:
				for i, c in enumerate(line):
					if c.startswith('Answer'):
						field_map[c] = i
					if c == 'Input.tweets':
						start_num = i
					if c == 'WorkerId':
						field_map[c] = i
						worker_col = i
				gold_llen = len(line)
				continue
			#print len(content), content
			#print start_num
			#sys.exit(0)
			#print worker_col
			#print start_num
			assert worker_col == 15 
			assert start_num == 27
			assert len(line) == gold_llen-2
			original_content = line[start_num]
			original_content_map = ast.literal_eval(original_content)
			original_content_map = ast.literal_eval(original_content_map)
			for j in range(1, 11):
				#print line[start_num+10+j], type(line[start_num+10+j])
				tweet_id = str(original_content_map['tweet_id_'+str(j)])
				if tweet_id in gold_idmap:
					#print 'hit!'
					print 'turker number:', line[worker_col], 'item number:', (j+1)
					gold_text = gold_idmap[tweet_id][0]
					gold_label = gold_idmap[tweet_id][1]
					turker_label = line[field_map['Answer.named_entity_mask_'+str(j)]]
					print gold_label, ('(gold label)')
					print turker_label, ('(turker_label)')
					tarray = extract_content(gold_text, gold_label) 
					print ' '.join(tarray).encode('utf-8')
					tarray = extract_content(gold_text, turker_label) 
					print ' '.join(tarray).encode('utf-8')
					numerator = 0.0
					denominator = 0.0
					for i,c in enumerate(turker_label):
						if c != '0' or gold_label[i] != '0':
							denominator += 1
							if c == gold_label[i]:
								numerator += 1
					if line[worker_col] not in worker_map:
						worker_map[line[worker_col]] = [numerator, denominator, max(len(gold_label), len(turker_label))]
					else:
						worker_map[line[worker_col]][0] += numerator
						worker_map[line[worker_col]][1] += denominator 
						worker_map[line[worker_col]][2] += max(len(gold_label), len(turker_label))
		print worker_map
		for k in worker_map:
			worker_map[k] = (float(worker_map[k][0]) / (worker_map[k][1]+0.01), float(worker_map[k][0]+worker_map[k][2]-worker_map[k][1])/worker_map[k][2] )
		print worker_map
		return worker_map, field_map


def acc_rej(filename, worker_map):
	for k in worker_map:
		if worker_map[k][0] > 0.1 :
			worker_map[k] = 'acc'
		else:
			worker_map[k] = 'rej'
	with open(filename, 'rb') as inf, open('labeled_'+filename, 'wb') as outf:   #, encoding='utf-8'
		#outf.write(u'\ufeff'.encode('utf8'))
		reader = unicode_csv_reader(inf)
		#writer = csv.writer(outf, delimiter=',' , dialect='excel')
		outf = UnicodeWriter(outf)
		line_num = 0
		line_length = None
		#print gold_idmap
		'''Note: hacky version for the problematic file.
		   Changed line content to only contain 3 columns.
		'''
		for line in reader:
			line_num += 1
			if line_num == 1:
				#outf.write('AssignmentId,HITId,Approve,Reject\n')
				line_length = len(line)
				#outf.write(','.join(line)+'\n')
				outf.writerow(line)
				for i, c in enumerate(line):
					if c == 'WorkerId':
						worker_col = i
				continue
			else:
				AssignmentId = line[worker_col-1]
				worker = line[worker_col]
				arry_line = list(line) #[AssignmentId, line[0]] #list(line)
				if worker_map[worker] == 'acc':
					arry_line.append("x")
					arry_line.append("")
				else:
					arry_line.append("")
					arry_line.append("Too many mistakes.")
				#print len(arry_line), line_length
				assert len(arry_line) == line_length
				#outf.write(','.join(arry_line).encode('utf-8')+'\n')
				outf.writerow(arry_line)


def load_turker_lfile_oldFormat(filename, gold_idmap):
	field_map = {}
	worker_map = {}
	with open(filename, 'rb') as inf:
		reader = unicode_csv_reader(inf)
		line_num = 0
		for line in reader:
			line_num += 1
			if line_num == 1:
				for i, c in enumerate(line):
					if c.startswith('Answer'):
						field_map[c] = i
					if c == 'Input.sentence_1':
						start_num = i
					if c == 'WorkerId':
						field_map[c] = i
						worker_col = i
				gold_llen = len(line)
				continue
			assert worker_col == 15 
			assert start_num == 27
			assert len(line) == gold_llen-2
			for j in range(1, 11):
				tweet_id = line[start_num+9+j]
				if tweet_id in gold_idmap:
					#print 'hit!'
					print 'turker number:', line[worker_col], 'item number:', (j+1)
					gold_text = gold_idmap[tweet_id][0]
					gold_label = gold_idmap[tweet_id][1]
					turker_label = line[field_map['Answer.named_entity_mask_'+str(j)]]
					print gold_label, ('(gold label)')
					print turker_label, ('(turker_label)')
					tarray = extract_content(gold_text, gold_label) 
					print ' '.join(tarray).encode('utf-8')
					tarray = extract_content(gold_text, turker_label) 
					print ' '.join(tarray).encode('utf-8')
					numerator = 0.0
					denominator = 0.0
					for i,c in enumerate(turker_label):
						if c != '0' or gold_label[i] != '0':
							denominator += 1
							if c == gold_label[i]:
								numerator += 1
					if line[worker_col] not in worker_map:
						worker_map[line[worker_col]] = [numerator, denominator, max(len(gold_label), len(turker_label))]
					else:
						worker_map[line[worker_col]][0] += numerator
						worker_map[line[worker_col]][1] += denominator 
						worker_map[line[worker_col]][2] += max(len(gold_label), len(turker_label))
		print worker_map
		for k in worker_map:
			worker_map[k] = (float(worker_map[k][0]) / (worker_map[k][1]+0.01), float(worker_map[k][0]+worker_map[k][2]-worker_map[k][1])/worker_map[k][2] )
		print worker_map
		return worker_map, field_map


def compose_CONLL_NER_oldFormat(infile, worker_map, field_map, threshold, outfile):
	tweet_map = {}
	with open(infile, 'rb') as inf, cs.open(outfile+'.train', 'wb', encoding='utf-8') as trf, cs.open(outfile+'.dev', 'wb', encoding='utf-8') as df, cs.open(outfile+'.test', 'wb', encoding='utf-8') as tf:
		reader = unicode_csv_reader(inf)
		line_num = 0
		#print gold_idmap
		for line in reader:
			line_num += 1
			if line_num == 1:
				for i, c in enumerate(line):
					#if c.startswith('Input'):
					if c == 'Input.sentence_1':
						start_num = i
					if c == 'WorkerId':
						field_map[c] = i
						worker_col = i
				gold_llen = len(line)
				continue
			worker = line[worker_col]
			if worker_map[worker][0] < threshold:
				#print 'skipping worker:', worker
				continue
			assert len(line) == gold_llen-2
			for j in range(1, 11):
				#print line[start_num+10+j], type(line[start_num+10+j])
				tweet_id = line[start_num+9+j]
				tweet_text = line[start_num+j-1].split(' ')
				turker_label = line[field_map['Answer.named_entity_mask_'+str(j)]]
				#print len(tweet_text), tweet_text, '(tweet_text)'
				#print len(turker_label), turker_label, ('(turker_label)')
				try:
					assert len(tweet_text) == len(turker_label)
				except:
					print 'text length issue! tweet ', tweet_id, 'text length:', len(tweet_text), 'label_length:', len(turker_label)
					continue
				output = get_label_oldFormat(tweet_text, turker_label, field_map, line, j-1)
				if tweet_id not in tweet_map:
					tweet_map[tweet_id] = [output]
				else:
					tweet_map[tweet_id].append(output)
		count = 0
		ner_type_map = {}
		line_num = 0
		for key, value in tweet_map.items():
			if len(value) > 1:
				count += 1
			output = get_consensus(value, ner_type_map)
			line_num += 1
			if line_num % 7 == 6:
				df.write('tweet_id:'+tweet_id+'\n')
				for elem in output:
					df.write(elem+'\n')
				df.write('\n')
			elif line_num % 7 == 0:
				tf.write('tweet_id:'+tweet_id+'\n')
				for elem in output:
					tf.write(elem+'\n')
				tf.write('\n')
			else:
				trf.write('tweet_id:'+tweet_id+'\n')
				for elem in output:
					trf.write(elem+'\n')
				trf.write('\n')
		for k, val in ner_type_map.items():
			print k, val
		print 'dup:', count, 'all:', len(tweet_map)
				

def compose_CONLL_NER(infile, worker_map, field_map, threshold, outfile):
	tweet_map = {}
	#with open(infile, 'rb') as inf, cs.open(outfile, 'wb', encoding='utf-8') as outf:
	with open(infile, 'rb') as inf, cs.open(outfile+'.train', 'wb', encoding='utf-8') as trf, cs.open(outfile+'.dev', 'wb', encoding='utf-8') as df, cs.open(outfile+'.test', 'wb', encoding='utf-8') as tf:
		reader = unicode_csv_reader(inf)
		line_num = 0
		#print gold_idmap
		for line in reader:
			line_num += 1
			if line_num == 1:
				for i, c in enumerate(line):
					if c.startswith('Input'):
					#if c == 'Input.tweets':
						start_num = i
					if c == 'WorkerId':
						field_map[c] = i
						worker_col = i
				gold_llen = len(line)
				continue
			worker = line[worker_col]
			if worker_map[worker][0] < threshold:
				#print 'skipping worker:', worker
				continue
			assert len(line) == gold_llen-2
			original_content = line[start_num]
			original_content_map = ast.literal_eval(original_content)
			original_content_map = ast.literal_eval(original_content_map)
			for j in range(1, 11):
				#print line[start_num+10+j], type(line[start_num+10+j])
				tweet_id = str(original_content_map['tweet_id_'+str(j)])
				tweet_text = original_content_map['tweet_text_'+str(j)].split(' ')
				turker_label = line[field_map['Answer.named_entity_mask_'+str(j)]]
				#print len(tweet_text), tweet_text, '(tweet_text)'
				#print len(turker_label), turker_label, ('(turker_label)')
				try:
					assert len(tweet_text) == len(turker_label)
					output = get_label(tweet_text, turker_label, field_map, line, j-1)
					if tweet_id not in tweet_map:
						tweet_map[tweet_id] = [output]
					else:
						tweet_map[tweet_id].append(output)
				except:
					print 'text length issue! tweet ', tweet_id, 'text length:', len(tweet_text), 'label_length:', len(turker_label)
					continue
		count = 0
		ner_type_map = {}
		line_num = 0
		for key, value in tweet_map.items():
			if len(value) > 1:
				count += 1
			output = get_consensus(value, ner_type_map)
			line_num += 1
			if line_num % 7 == 6:
				df.write('tweet_id:'+tweet_id+'\n')
				for elem in output:
					df.write(elem+'\n')
				df.write('\n')
			elif line_num % 7 == 0:
				tf.write('tweet_id:'+tweet_id+'\n')
				for elem in output:
					tf.write(elem+'\n')
				tf.write('\n')
			else:
				trf.write('tweet_id:'+tweet_id+'\n')
				for elem in output:
					trf.write(elem+'\n')
				trf.write('\n')
			'''outf.write('tweet_id:'+tweet_id+'\n')
			for elem in output:
				outf.write(elem+'\n')
			outf.write('\n')
			'''
		for k, val in ner_type_map.items():
			print k, val
		print 'dup:', count, 'all:', len(tweet_map)
				

def get_consensus(annotations, ner_type_map):
	pre_label = ''
	pre_anno = ''
	output = []
	for elem in zip(*annotations): 
		result = get_majority(elem)
		if result[1] == 'O':
			output.append('\t'.join(result))
			pre_label = result[1]
		elif result[1].startswith('B'):
			stuff = result[1].split('-')
			if pre_label == 'B' and pre_anno == stuff[1]:
				output.append(result[0]+'\tI-'+pre_anno)
			else:
				output.append('\t'.join(result))
				if stuff[1] not in ner_type_map:
					ner_type_map[stuff[1]] = 0
				ner_type_map[stuff[1]] += 1
			pre_label = stuff[0]
			pre_anno = stuff[1]
		else:
			output.append(result[0]+'\tI-'+pre_anno)
			pre_label = result[1]
	return output



def get_majority(elems):
	columns = zip(*elems)
	assert len(columns) == 2
	output = []
	for answers in columns:
		c=Counter(answers)
		value, count = c.most_common()[0]
		output.append(value)
	return output

def get_label_oldFormat(tweet_text, labels, field_map, line, idx):
	output = []
	annotation = ''
	for i, l in enumerate(labels):
		if l == '0' :
			output.append((tweet_text[i],'O'))
		if l == '1':
			anno_key = get_key(idx, i)
			annotation = line[field_map[anno_key]]
			output.append((tweet_text[i],'B-'+annotation))
		if l == '2':
			output.append((tweet_text[i],'I-'+annotation))
	return output


def get_label(tweet_text, labels, field_map, line, idx):
	output = []
	annotation = ''
	for i, l in enumerate(labels):
		if l == '0' :
			output.append((tweet_text[i].decode('unicode-escape'),'O'))
		if l == '1':
			anno_key = get_key(idx, i)
			annotation = line[field_map[anno_key]]
			output.append((tweet_text[i].decode('unicode-escape'),'B-'+annotation))
		if l == '2':
			output.append((tweet_text[i].decode('unicode-escape'),'I-'+annotation))
	return output


def get_key(sent_idx, word_idx):
	return 'Answer.table'+str(sent_idx)+'sentence'+str(sent_idx)+'word'+str(word_idx)


class UTF8Recoder:
	"""
	Iterator that reads an encoded stream and reencodes the input to UTF-8
	"""
        def __init__(self, f, encoding):
	        self.reader = cs.getreader(encoding)(f)
	        def __iter__(self):
	        	return self

	        def next(self):
		        return self.reader.next().encode("utf-8")

class UnicodeWriter:
	"""
        A CSV writer which will write rows to CSV file "f",
	which is encoded in the given encoding.
        """

        def __init__(self, f, dialect='excel', encoding="utf-7", **kwds):
        # Redirect output to a queue
            self.queue = cStringIO.StringIO()
            self.writer = csv.writer(self.queue, dialect=dialect, **kwds)
	    self.stream = f
            self.encoder = cs.getincrementalencoder(encoding)()

	def writerow(self, row):
		self.writer.writerow([s.encode("utf-8") for s in row])
	        # Fetch UTF-8 output from the queue ...
	        data = self.queue.getvalue()
	        data = data.decode("utf-8")
	        # ... and reencode it into the target encoding
	        data = self.encoder.encode(data)
	        # write to the target stream
	        self.stream.write(data)
		# empty queue
		self.queue.truncate(0)

	def writerows(self, rows):
	        for row in rows:
			self.writerow(row)


''' Usage: python postProcess.py gold_file annotation_file
'''
if __name__ == '__main__':
	filename = sys.argv[1]
	anno_file = sys.argv[2]
	gold_data = load_gold_file(filename)
	#acc_rej(anno_file, worker_map)
	worker_map,field_map = load_turker_lfile_oldFormat(anno_file, gold_data[0]) 
	compose_CONLL_NER_oldFormat(anno_file, worker_map, field_map, 0.3, sys.argv[3])
	#worker_map,field_map = load_turker_lfile(anno_file, gold_data[0]) 
	#compose_CONLL_NER(anno_file, worker_map, field_map, 0.3, sys.argv[3])
