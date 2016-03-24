#/user/local/bin/python
# -*- coding: utf-8 -*-

import sys
import gzip
import codecs as cs

if __name__ == '__main__':
	infile = sys.argv[1]
	outfile = sys.argv[2]
	with gzip.open(infile, 'rb') as inf, cs.open(outfile, 'w', encoding='utf-8') as outf:
		for line in inf:
			newline = ' '.join(line.decode('utf-8'))
			outf.write(newline)
