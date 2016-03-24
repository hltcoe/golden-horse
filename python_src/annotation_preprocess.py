#/user/local/bin/python
# -*- coding: utf-8 -*-

import sys
import os
import json
import jieba
import re
import gzip
import codecs as cs
from concrete.util.file_io import *
from data_ingest import create_comm_from_sent

def convert_tweet_to_concrete(input_dir, out_dir):
    punc = u'[\[\]\.,\-%()+:\'\";\\\/!\t\*?@#&1234567890><】~=.【…”“‘’（）——﹏『』① ② ③ ④ 「 」╮ ╯ ▽ ╰ ╭！→︶ ︿ ︶┮《》%   ]+'
    replace_symbol = u'[,]'
    for root, subFolders, files in os.walk(input_dir):
        #print 'processing file ', root, subFolders, files
        #print "writing out to ", out_dir+re.sub('\/', '-', root)+'segmented'
        if len(subFolders) == 0:
            print 'processing file ', root, subFolders  #, files
            for f in files:
                if 'posts' in f:
                    try:
                        outfile_name = f[:-7]+'concrete.tar.gz'
                        writer = CommunicationWriterTGZ()
                        writer.open(os.path.join(out_dir, outfile_name))
                        print 'writing to file:', os.path.join(out_dir, outfile_name)
                        with gzip.open(root+'/'+f, 'rb') as inf:
                            for line in inf :
                                #print 'processing!!!!'
                                json_line = json.loads(line.decode('utf-8').strip())
                                if json_line["text"] is None or json_line["mid"] is None:
                                    continue
                                original = json_line['text'].strip()
                                nonpunc = re.sub(punc, '', original)
                                content = re.sub('\w+', '', nonpunc) #json_line['text'])
                                if len(content) < 10:
                                    continue
                                tag_arry = ['O'] * len(content)	
                                comm = create_comm_from_sent(json_line['mid'], replace_emoji_characters(re.sub(replace_symbol, u'，', original)), tag_arry) 
                                writer.write(comm,'weibo_'+json_line["mid"]+'.concrete')
                        writer.close()
                    except:
                        print "There was an error opening tarfile. The file might be corrupt or missing."


def parse_file(file_dir, cn_dict, out_dir):
  punc = u'[\[\]\.,\-%()+:\'\";\\\/!\t\*?@#&1234567890><】~=.【…”“‘’（）——﹏『』① ② ③ ④ 「 」╮ ╯ ▽ ╰ ╭！→︶ ︿ ︶┮《》%   ]+'
  replace_symbol = u'[,]'
  total_line = -1
  for root, subFolders, files in os.walk(file_dir):
    #print 'processing file ', root, subFolders, files
    #print "writing out to ", out_dir+re.sub('\/', '-', root)+'segmented'
    #continue
    if len(subFolders) == 0:
      print 'processing file ', root, subFolders
      with gzip.open(out_dir+'segmented'+re.sub('\/', '-', root)+'all.gz', 'wb+') as outf:
	for f in files:
	  if 'posts' in f:
	    try:
	      with gzip.open(root+'/'+f, 'rb') as inf:
		ocount = 0
		sent_arry = []
		id_arry = []
		for line in inf :
			total_line += 1
			if total_line % 10000 != 0:
				continue
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
				sent_arry.append(replace_emoji_characters(' '.join(re.sub(replace_symbol, u'，', original))))
				id_arry.append(str(json_line["id"]))
				if len(sent_arry) == 10:
					#if total_line % 1000 == 0:
					print "recording item", total_line
					outf.write(','.join(sent_arry).encode('utf-8')+','+','.join(id_arry)+'\n')
					#total_line += 1
					sent_arry = []
					id_arry = []
				#.write(content.encode('utf-8')+'\n')
				#outf.write(unicode(' '.join(new_word_list)).encode('utf-8') + '\n')
			else:
				ocount += 1	
		print '# sentences without new words:', ocount
	    except:
	      print "There was an error opening tarfile. The file might be corrupt or missing."

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
	#cn_dict = load_dict(sys.argv[2])
	#parse_file(sys.argv[1], cn_dict, sys.argv[3])
	convert_tweet_to_concrete(sys.argv[1], sys.argv[2])
