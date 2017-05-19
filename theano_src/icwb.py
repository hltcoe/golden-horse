#!/usr/bin/python 
# -*- coding: utf-8 -*-

import codecs as cs

import re, sys
import numpy
import theano
import jieba
import random
from collections import defaultdict
from weiboNER_features import *

OOV = '_OOV_'
SEG = 'Segmentation'
feature_thresh = 0
name_len_thresh = 5
# Attribute templates. 
local_templates = (
    (('w', -2), ),
    (('w', -1), ),
    (('w',  0), ),
    (('w',  1), ),
    (('w',  2), ),
    (('w', -2), ('w',  -1)),
    (('w', -1), ('w',  0)),
    (('w',  0), ('w',  1)),
    (('w',  1), ('w',  2)),
    (('w',  -1), ('w',  1)),
)

def prepare_data(seqs, labels, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    # This part: suspeciously wrong.
    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, labels


def load_data(train_path=None, valid_path=None, test_path=None, representation='charpos', n_words=1000000, valid_portion=0.1, maxlen=None, sort_by_len=False):
    '''Loads the dataset

    :type train_path: String
    :param train_path: The path to the training dataset (here sighan05 segmentation task)
    :type valid_path: String
    :param valid_path: The path to the dev dataset (here sighan05 segmentation task)
    :type test_path: String
    :param test_path: The path to the test dataset (here sighan05 segmentation task)
    :type n_words: int
    :param n_words: The number of word to keep in the vocabulary.
        All extra words are set to unknow (1).
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.

    '''

    def sentence_segmentation(content_str):
        import re
        sent_end = u'[。？！，：；]+'
        sent_end_set = set([u'。', u'？', u'！', u'，', u'：', u'；'])
        wrapper = set([u'\"', u'”', u'』'])
        offsetInContent=0
        sstart = 0
        sent_array = []
        for sent_str in re.split(sent_end, content_str):
            if len(sent_str) == 0:
                continue
            if len(sent_str.strip()) == 1 and sent_str.strip() in wrapper:
                sent_array[-1] += sent_str.rstrip()
                #print sent_array[-1]
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
            if sent_str.strip() in sent_end_set:
                sent_array[-1] += sent_str
                continue
            sent_array.append(sent_str)
            #print 'sentence:', sent_str, 'start=', sstart, 'end=', send
        return sent_array


    def read_file(filename, representation, labeled=True):
        corpus_x = []
        corpus_y = []
        line_pointer = []
        urlStr = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        with cs.open(filename, 'r', encoding='utf-8') as inf:
            #residual = []
            line_count = 0
            #accumulate_count = 0
            for line in inf:
                line_count += 1
                line = line.strip()
                if len(line) == 0:
                    continue
                sentences = sentence_segmentation(line.strip())
                for sent in sentences:        
                    sent = stringQ2B(sent)
                    elems = sent.strip().split() #('  ') #line.split('  ')
                    #print elems
                    #print sent
                    if len(elems) < 1:
                        print elems
                        continue
                    x = []
                    y = []
                    if representation == 'charpos':
                        charpos = [char+str(i) for seg in jieba.cut(''.join(elems)) for i, char in enumerate(seg)]
                    pointer = 0
                    for wd in elems:
                        '''Special treatment for URL'''
                        wd = re.sub(urlStr, 'URL', wd)
                        if wd == 'URL':
                            x.append(wd)
                            y.append('S-word')
                            continue
                        '''EOSpecial treatment for URL'''
                        for i, char in enumerate(wd):
                            rep = char_transform(char)
                            # when rep == 'O', it's regular hanzi
                            if rep != 'O':
                                x.append(rep)
                            elif representation == 'charpos':
                                x.append(charpos[pointer])
                            else:
                                x.append(char) #pos[pointer])
                            pointer += 1
                            if not labeled:
                                y.append('N')
                                continue
                            if len(wd) == 1:
                                y.append('S-word')
                            elif i == 0:
                                y.append('B-word')
                            elif i == len(wd)-1 :
                                y.append('E-word')
                            else:
                                y.append('I-word')
                    assert len(x) == len(y)
                    if len(x) < 2:
                        continue
                    #print len(x), len(y)
                    corpus_x.append(x)
                    corpus_y.append(y)
                    line_pointer.append(line_count)
        features = extract_name_feature('resources/names.txt', corpus_x, name_len_thresh) #extract_conll_features(corpus_x, corpus_y)
        print 'read file', filename, len(features), len(corpus_x), len(corpus_y), len(line_pointer)
        return features, corpus_x, corpus_y#, line_pointer

    
    def extract_conll_features(corpus_x, corpus_y):
        conll_format_data = [[(x, y) for x, y in zip(line_x, line_y)] for line_x, line_y in zip(corpus_x, corpus_y)]
        fields_template = 'w y'
        features = [feature_extractor(readiter(sntc, fields_template.split(' ')), templates=local_templates) for sntc in conll_format_data]
        return features
    
    def extract_name_feature(name_file_name, corpus_x, length_thre):
        name_set, max_name_size = load_name_list(name_file_name)
        features = []
        for x in corpus_x:
            ftv = [{'F':dict()} for i in range(len(x))]
            #residual = 1
            x = [elem[:-1] for elem in x]
            #print 'sentence:', ''.join(x)
            assert len(ftv) == len(x)
            for i in range(len(x)):
                for j in range(1, max_name_size+1):
                    if i+j > len(x):
                        break
                    word = ''.join(x[i:min(i+j, len(x))])
                    #print i, j, word
                    if word in name_set:
                        #print 'name:', word
                        for pnt in xrange(j):
                            if pnt < name_len_thresh:
                                ftv[i+pnt]['F']['_InName_'+str(pnt)] = 1
                            ftv[i+pnt]['F']['_InName_'] = 1
                        #sent_features[i:i+j] = 1
                        #residual = j
                        break
                #if (j == max_name_size or i+j > len(x)) and word not in name_set:
                #    residual = 1
            features.append(ftv)
        assert len(features) == len(corpus_x)
        return features

    def load_name_list(file_name):
        name_list = []
        max_name_size = 0
        with cs.open(file_name, 'r', encoding='utf-8') as inf:
            for line in inf:
                elem = line.split()
                name_list.append(elem[0])
                if len(elem[0]) > max_name_size:
                    max_name_size = len(elem[0])
        return set(name_list), max_name_size

    print 'loading training data from', train_path, 'loading valid data from', valid_path, 'loading test data from', test_path
    train_set = read_file(train_path, representation, True) 
    labels2idx = {'S-word':0, 'B-word':1, 'E-word':2, 'I-word':3, 'N':4}

    if test_path != None:
        test_set = read_file(test_path, representation, True) 
    else:
        test_set = None
    words = [w for sent in train_set[1] for w in sent ]
    features = list(train_set[0])
    if valid_path != None:
        valid_set = read_file(valid_path, representation, True)  
        words_valid = [w for sent in valid_set[1] for w in sent ]
        words.extend(words_valid)
        features += valid_set[0]
    if test_set != None:
        words_test = [w for sent in test_set[1] for w in sent ]
        words.extend(words_test)
        features += test_set[0]
    ### compose words2idx on the fly. ###
    words2idx = {OOV: 0}
    for w in words:
        if w not in words2idx:
            words2idx[w] = len(words2idx)    
    print 'voc_size:', len(words2idx)
        
    feature_to_freq = defaultdict(int)
    for ftvs in features:
        for ftv in ftvs:
            for ft in ftv['F']:
                feature_to_freq[ft]+=1
    print 'feature to freq:'
    for k, v in feature_to_freq.items():
        print k, v
    
    features2idx = {OOV: 0}
    cur_idx = 1
    for ftvs in features:
        for ftv in ftvs:
            for ft in ftv['F']:
                ft = escape(ft)
                if (ft not in features2idx) and feature_to_freq[ft] > feature_thresh:
                    features2idx[ft] = cur_idx
                    cur_idx+=1
    print 'feature to idx:'
    for k, v in features2idx.items():
        print k,v
    if maxlen:
        new_train_set_f = []
        new_train_set_x = []
        new_train_set_y = []
        #new_train_set_z = []
        for f, x, y in zip(train_set[0], train_set[1], train_set[2]):#, train_set[3]):
            if len(x) < maxlen:
                new_train_set_f.append(f)
                new_train_set_x.append(x)
                new_train_set_y.append(y)
                #new_train_set_z.append(z)
        train_set = (new_train_set_f, new_train_set_x, new_train_set_y)#, new_train_set_z)
        del new_train_set_f, new_train_set_x, new_train_set_y#, new_train_set_z

    if valid_path == None:
        # split training set into validation set
        train_set_f, train_set_x, train_set_y = train_set  #, train_set_z
        print 'before splitting the training set', len(train_set_x)
        n_samples = len(train_set_x)
        #sidx = numpy.random.permutation(n_samples)
        n_train = int(numpy.round(n_samples * (1. - valid_portion)))
        valid_set_f = train_set_f[n_train:] #[train_set_f[s] for s in sidx[n_train:]]
        valid_set_x = train_set_x[n_train:] #[train_set_x[s] for s in sidx[n_train:]]
        valid_set_y = train_set_y[n_train:] #[train_set_y[s] for s in sidx[n_train:]]
        #valid_set_z = [train_set_z[s] for s in sidx[n_train:]]
        train_set_f = train_set_f[:n_train] #[train_set_f[s] for s in sidx[:n_train]]
        train_set_x = train_set_x[:n_train] #[train_set_x[s] for s in sidx[:n_train]]
        train_set_y = train_set_y[:n_train] #[train_set_y[s] for s in sidx[:n_train]]
        #train_set_z = [train_set_z[s] for s in sidx[:n_train]]

        print 'after splitting the training set', len(train_set_x), len(valid_set_x)
        train_set = (train_set_f, train_set_x, train_set_y)#, train_set_z)
        valid_set = (valid_set_f, valid_set_x, valid_set_y)#, valid_set_z)

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    valid_set_f, valid_set_x, valid_set_y = valid_set #, valid_set_z
    train_set_f, train_set_x, train_set_y = train_set #, train_set_z
    print 'before word to index, sizes:', len(train_set_f), len(train_set_x), len(valid_set_f), len(valid_set_x) 
    
    train_set_f = [[[features2idx[escape(ft)] for ft in ftv['F']] for ftv in ftvs] for ftvs in train_set_f]
    valid_set_f = [[[features2idx[escape(ft)] for ft in ftv['F']] for ftv in ftvs] for ftvs in valid_set_f]
    train_set_x = [[words2idx[w] for w in sent] for sent in train_set_x ]
    valid_set_x = [[words2idx[w] for w in sent]  for sent in valid_set_x]
    train_set_y = [[labels2idx[lb] for lb in sent] for sent in train_set_y ]
    valid_set_y = [[labels2idx[lb] for lb in sent] for sent in valid_set_y ]
    if test_path != None:
        test_set_f, test_set_x, test_set_y = test_set   #, test_set_z
        if not test_path.endswith('.gz'):
            test_set_f = [[[features2idx[escape(ft)] for ft in ftv['F']] for ftv in ftvs] for ftvs in test_set_f]

            test_set_x = [[words2idx.get(w, 0) for w in sent] for sent in test_set_x ]
    
            test_set_y = [[labels2idx[lb] for lb in sent] for sent in test_set_y ]
            test_set_x = remove_unk(test_set_x)
    
    print 'after word to index, sizes:', len(train_set_f), len(train_set_x), len(valid_set_f), len(valid_set_x), (len(test_set_f), len(test_set_x)) if test_path != None else 0
    train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        '''sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]
        '''
        sorted_index = len_argsort(valid_set_x)
        valid_set_f = [valid_set_f[i] for i in sorted_index]
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]
        #valid_set_z = [valid_set_z[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_f = [train_set_f[i] for i in sorted_index]
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]
        #train_set_z = [train_set_z[i] for i in sorted_index]

    train = [train_set_f, train_set_x, train_set_y]#, train_set_z)
    valid = [valid_set_f, valid_set_x, valid_set_y]#, valid_set_z)
    dics = {'features2idx':features2idx, 'labels2idx': labels2idx, 'words2idx': words2idx}
    if test_path != None: 
        test = [test_set_f, test_set_x, test_set_y]#, test_set_z)
    else:
        test = None
    return [train, valid, test, dics]


def convert_prediction(word_array, label_array, line_pointer):
    corpus = []
    line = []
    curr_line = 1
    for (x, y, z) in zip(word_array, label_array, line_pointer):
        if z > curr_line:
            curr_line = z
            line.append('  ') 
            corpus.append(line)
            line = []
        append_line(line, x, y)
    corpus.append(line)
    return corpus


def append_line(line, word_array, label_array):
    for (x, y) in zip(word_array, label_array):
        if y.startswith('B-') or y.startswith('S-'):
            line.append('  ')
            line.append(x)
        else:
            line.append(x)

def char_transform(uchar):
    punc = set(u'—（）／．《》『』，、。？；：！……“”‘’|,.;:\'\"!+-@#$%^&*()\\=~`></?{}[]')
    num = set([u'①', u'②', u'③', u'④', u'⑤', u'⑥', u'⑦', u'○',u'一',u'二',u'三',u'四',u'五',u'六',u'七',u'八',u'九',u'十',u'百',u'千',u'万',u'亿',u'两',u' ％',u'１', u'２', u'３', u'４', u'５', u'６', u'７', u'８', u'９', u'０'])
    date = set([u'日',u'月',u'年'])
    if uchar in punc:
        return 'P'
    elif is_number(uchar) or uchar in num:
        return 'N'
    elif uchar in date:
        return 'D'
    elif is_alphabet(uchar):
        return 'E'
    elif is_other(uchar):
        return 'S'
    else:
        return 'O'

def quick_convert(in_content):
    out_content = []
    for line in in_content:
        if line.strip() == '':
            out_content.append(line)
            continue
        elem = line.split('\t')
        debug_elem = line.split()
        try:
            assert len(debug_elem) == 2
        except:
            print 'abnomal line!', line
            print elem, debug_elem
        catgry = char_transform(elem[0])
        aabb = 'N'
        abab = 'N'
        #elem[0] = Q2B(elem[0])
        if len(out_content) > 0 and out_content[-1] != '' and elem[0] == out_content[-1][0]:
            aabb = 'Y'
        if len(out_content) > 1 and out_content[-2] != '' and elem[0] == out_content[-2][0]:
            abab = 'Y'
        out_content.append(elem[0]+' '+catgry+' '+aabb+' '+abab+' '+elem[1])
    return out_content

def is_chinese(uchar):  
    """判断一个unicode是否是汉字"""  
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':  
        return True  
    else:  
        return False  

def is_number(uchar):  
    """判断一个unicode是否是数字"""  
    if uchar >= u'\u0030' and uchar<=u'\u0039':  
        return True  
    else:  
        return False  

def is_alphabet(uchar):  
    """判断一个unicode是否是英文字母"""  
    if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):  
        return True  
    else:  
        return False 

def is_other(uchar):  
    """判断是否非汉字，数字和英文字符"""  
    if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):  
        return True  
    else:  
        return False  

def Q2B(uchar):  
    """全角转半角"""  
    inside_code=ord(uchar)  
    if inside_code==0x3000:  
        inside_code=0x0020  
    else:  
        inside_code-=0xfee0  
    if inside_code<0x0020 or inside_code>0x7e: #转完之后不是半角字符返回原来的字符  
        return uchar  
    return unichr(inside_code)  
                                                      
def stringQ2B(ustring):  
    """把字符串全角转半角"""  
    return "".join([Q2B(uchar) for uchar in ustring]) 


def idx2Content_conll(data, idx2word, idx2label):
    trf, trw, trl = data 
    assert len(trw) == len(trl)
    conll_content = []
    for wl, ll in zip(trw, trl):
        words = [idx2word[w] for w in wl]
        labels = [idx2label[l] for l in ll]
        assert len(words) == len(labels)
        for w, l in zip(words, labels):
            conll_content.append(w+'\t'+l)
        conll_content.append('')
    return conll_content

def quick_sample(filename):
    with cs.open(filename, 'r', encoding='utf-8') as inf, cs. open(filename+'.small', 'w', encoding='utf-8') as outf:
        corpus = []
        for line in inf:
            corpus.append(line)
        sample_idx = random.sample(corpus, 800)
        for idx in sample_idx:
            outf.write(idx)
        

if __name__ == '__main__':
    quick_sample(sys.argv[1])
    exit(0)
    train, valid, test, dics = load_data(train_path=sys.argv[1]) #, test_path=sys.argv[2])
    idx2label = dict((k, v) for v, k in dics['labels2idx'].iteritems())
    idx2word = dict((k, v) for v, k in dics['words2idx'].iteritems())
    # process training data
    train_conll = idx2Content_conll(train, idx2word, idx2label)    
    # process dev data
    valid_conll = idx2Content_conll(valid, idx2word, idx2label)
    # process test data
    #test_conll = idx2Content_conll(test, idx2word, idx2label)
    out_content = quick_convert(train_conll)
    with cs.open(sys.argv[3], 'w', encoding='utf-8') as outf:
        for line in out_content:
            outf.write(line+'\n')
    out_content = quick_convert(valid_conll)
    with cs.open(sys.argv[4], 'w', encoding='utf-8') as outf:
        for line in out_content:
            outf.write(line+'\n')
    '''out_content = quick_convert(test_conll)
    with cs.open(sys.argv[5], 'w', encoding='utf-8') as outf:
        for line in out_content:
            outf.write(line+'\n')
    '''
