#/user/local/bin/python
# -*- coding: utf-8 -*-

import sys
import os
import json
import jieba
import re
import gzip
import codecs as cs
#reload(sys)
#sys.setdefaultencoding('utf-8')

def parse_file(file_dir, cn_dict, out_dir):
    punc = u'[\[\]\.,\-%()+:\'\";\\\/!\t\*?@#&1234567890><&#】~=.【…”“‘’（）——﹏『』① ② ③ ④ 「 」╮ ╯ ▽ ╰ ╭！→︶ ︿ ︶┮《》%   ]+'
    #for root, subFolders, files in os.walk(file_dir):
    	#print 'processing file ', root, subFolders, files
	#for f in files:
	#  if 'posts' in f:
    	#    print 'processing file ', root, subFolders, f
	#    try:
	#      with gzip.open(root+'/'+f, 'rb') as inf, gzip.open(out_dir+'/segmented_'+f, 'wb') as outf:
    with cs.open(file_dir, 'r', encoding='utf-8') as inf, cs.open(out_dir, 'w', encoding='utf-8') as outf:
		ocount = 0
		sent_arry = []
		id_arry = []
		for line in inf :
			json_line = json.loads(line.strip())  #encoding='utf-8'
			if json_line["text"] is None or json_line["id"] is None:
		                continue
			nonpunc = re.sub(punc, '', json_line['text'].strip())
			content = re.sub('\w+', '', nonpunc) #json_line['text'])
			if len(content.strip()) < 10:
				continue
			seg_list = jieba.cut(content)#, cut_all=True)
			has_new_word = False
			new_word_list = []
			for wd in seg_list:
				if wd not in cn_dict:
					new_word_list.append(wd)
					#print content
					#print wd
					#print "/ ".join(seg_list)
					#has_new_word = True
					#break
			if len(new_word_list) >=2 :
				#print content
				#print "/ ".join(new_word_list)
				#outf.write(line+'\n')
				#outf.write(content.encode('utf-8')+'\n')
				#sent_arry.append(content)
				sent_arry.append(replace_emoji_characters(' '.join(content)))
				id_arry.append(str(json_line["id"]))
				if len(sent_arry) == 10:
					outf.write(','.join(sent_arry)+','+','.join(id_arry)+'\n')
					sent_arry = []
					id_arry = []
				#outf.write(unicode(' '.join(new_word_list)).encode('utf-8') + '\n')
			else:
				ocount += 1	
		print '# sentences without new words:', ocount
	    #except:
	    #  print "There was an error opening tarfile. The file might be corrupt or missing."

def load_dict(dict_file):
	cn_dict = set()
	with cs.open(dict_file, 'r', encoding='utf-8') as df:
		for line in df:
			cn_dict.add(line.strip())
	print 'dictionary size:', len(cn_dict)
	#for elem in cn_dict:
	#	print elem
	return cn_dict

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


if __name__ == '__main__':
	cn_dict = load_dict(sys.argv[2])
	parse_file(sys.argv[1], cn_dict, sys.argv[3])
