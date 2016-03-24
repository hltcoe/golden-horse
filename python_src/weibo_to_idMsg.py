#!/usr/bin/python

import codecs as cs 
import sys

def convert_file(infile, outfile):
    with cs.open(infile, 'r', encoding='utf-8') as inf, cs.open(outfile, 'w', encoding='utf-8') as outf:
        line_count = 0
        for line in inf:
            line_count += 1
            #print line_count
            #print line
            if line_count == 1:
                continue
            else:
                elem = line.split(',')
                elem[-1] = elem[-1].rstrip()
                try:
                    assert len(elem) == 20
                except:
                    print len(elem)
                for i in range(10):
                    outf.write(elem[10+i]+'\t'+elem[i]+'\n')


if __name__ == '__main__':
    convert_file(sys.argv[1], sys.argv[2])
