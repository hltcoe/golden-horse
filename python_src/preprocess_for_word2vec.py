#/user/local/bin/python
# -*- coding: utf-8 -*-

import sys
import os
import json
import jieba
import re
import gzip
import codecs as cs


def parse_file(file_dir, segment=False):
  punc = u'[\[\]\.,\-%()+:\'\";\\\/!\t\*?@#&1234567890><】~=.【…”“‘’（）——﹏『』① ② ③ ④ 「 」╮ ╯ ▽ ╰ ╭！→︶ ︿ ︶┮《》%   ]+'
  replace_symbol = u'[,]'
  total_line = -1
  prefix = '/export/projects/npeng/weiboNER_data/'
  weibo_count = 0
  if segment:
	  prefix += 'segmented_'
  with open(prefix+'weibo_w2v_train', 'w') as outf:
    for root, subFolders, files in os.walk(file_dir):
      if len(subFolders) == 0:
        print 'processing file ', root, subFolders
	for f in files:
	  if 'posts' in f:
	    try:
	      with gzip.open(root+'/'+f, 'rb') as inf:
		for line in inf :
			total_line += 1
			#if total_line % 10 != 0:
			#	continue
			#print 'processing!!!!'
			json_line = json.loads(line.decode('utf-8').strip())
			#if json_line["text"] is None :
			if json_line["text"] is None or json_line["id"] is None:
		                continue
			original = json_line['text'].strip()
			nonpunc = re.sub(punc, '', original)
			content = re.sub('\w+', '', nonpunc) #json_line['text'])
			if len(content.strip()) < 10:
				continue
			content = replace_emoji_characters(content)
			if segment:
				seg_list = jieba.cut(content)#, cut_all=True)
			else :
				seg_list = content
			outf.write(' '.join(seg_list).encode('utf-8')+'\n')
			weibo_count += 1
			#.write(content.encode('utf-8')+'\n')
			#outf.write(unicode(' '.join(new_word_list)).encode('utf-8') + '\n')
	    except:
	      print "There was an error opening tarfile. The file might be corrupt or missing."
  print "total weibo number =", weibo_count


def replace_emoji_characters(s, replacement_character=u' '):
    """
    Replace Emoji characters in a Unicode string with the specified
    character.

    Mechanical Turk will reject CSV input files containing Emoji
    characters with this error message:

      Your CSV file needs to be UTF-8 encoded and cannot contain
      characters with encodings larger than 3 bytes.
    """
    # The procedure for stripping Emoji characters is based on this StackOverflow post:
    #   http://stackoverflow.com/questions/12636489/python-convert-4-byte-char-to-avoid-mysql-error-incorrect-string-value

    if sys.maxunicode == 1114111:
        # Python was built with '--enable-unicode=ucs4'
        highpoints = re.compile(u'[\U00010000-\U0010ffff]')
    elif sys.maxunicode == 65535:
        # Python was built with '--enable-unicode=ucs2'
        highpoints = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
    else:
        raise UnicodeError("Unable to determine if Python was built using UCS-2 or UCS-4")

    return highpoints.sub(replacement_character, s)

# convert segmented document into char+position document
def char_position(infile, outfile, sample_freq=1):
	with cs.open(sys.argv[1], 'r', encoding='utf-8') as inf, cs.open(sys.argv[2], 'w', encoding='utf-8') as outf:
		line_num = 0
		for line in inf:
			line_num += 1
			if line_num % sample_freq != 0:
				continue
			new_arry = []
			elems = line.rstrip().split(' ')
			for wd in elems:
				for i, char in enumerate(wd):
					new_arry.append(char+str(i))
			outf.write(' '.join(new_arry)+' ')
		print "total line number =", line_num


def get_content(file):
	content = []
	file_content = []
	record = False
	for i, line in enumerate(file):
		if line.strip() == '<TEXT>':
			record = True
		if line.strip() == '</TEXT>':
			record = False
		if record and not line.strip().endswith('P>'):
			content.append(line.rstrip().decode('utf-8'))
		if line.strip() == '</P>':
			paragraph = ''.join(content)
			file_content.append(paragraph)
			content = []
	return file_content #''.join(content)


def process_gigaword_data(file_dir, segment=False):
  #punc = u'[\[\]\.,\-%()+:\'\";\\\/!\t\*?@#&1234567890><】~=.【…”“‘’（）——﹏『』① ② ③ ④ 「 」╮ ╯ ▽ ╰ ╭！→︶ ︿ ︶┮《》%   ]+'
  prefix = '/export/projects/npeng/weiboNER_data/'
  paragraph_count = 0
  if segment:
	  prefix += 'segmented_'
  with open(prefix+'gigaword_w2v_2nd_train', 'w') as outf:
    for root, subFolders, files in os.walk(file_dir):
      if len(subFolders) == 0:
        print 'processing files ', root, subFolders
	for f in files:
	    try:
	      with gzip.open(root+'/'+f, 'rb') as inf:
		  print 'processing file ', f
		  paragraphs = get_content(inf)
		  for line in paragraphs:
			#nonpunc = re.sub(punc, '', line)
			content = re.sub('\w+', '', line) #json_line['text'])
			if len(content.strip()) < 10:
				continue
			if segment:
				seg_list = jieba.cut(content)#, cut_all=True)
			else :
				seg_list = content
			outf.write(' '.join(seg_list).encode('utf-8')+'\n')
			paragraph_count += 1
			#.write(content.encode('utf-8')+'\n')
			#outf.write(unicode(' '.join(new_word_list)).encode('utf-8') + '\n')
	    except:
	      print "There was an error opening tarfile. The file might be corrupt or missing."
  print "total paragraph number =", paragraph_count 

if __name__ == '__main__':
	#segment = False
	#if sys.argv[2] == 'segment':
	#	segment = True
	#parse_file(sys.argv[1], segment)
	'''with cs.open(sys.argv[1], 'r', encoding='utf-8') as inf, cs.open(sys.argv[2], 'w', encoding='utf-8') as outf:
		freq = int(sys.argv[3])
		count = 0
		for line in inf:
			count += 1
			if count % freq == 0:
			    #outf.write(line.rstrip()+' ')
			    outf.write(' '.join(''.join(line.strip().split(' ')))+' ')
	'''
	char_position(sys.argv[1], sys.argv[2], int(sys.argv[3]))
	#process_gigaword_data(sys.argv[1], True)
