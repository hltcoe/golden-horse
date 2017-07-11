import codecs as cs
import os
import sys
import tarfile
import cPickle as pickle
import time
import uuid
import random
#import jieba.posseg as pseg
#import jieba
from icwb import char_transform
from collections import defaultdict
from weiboNER_features import *

OOV = '_OOV_'
GOLD_TAG = 'GoldNER'
POS = 'POS'
SEG = 'Segmentation'
PRED_TAG = 'NER'
task = 'ner'
random.seed(0)
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

def create_dicts(train_fn, valid_fn, feature_thresh, test_fn, mode, anno):
    get_label = lambda fn, pos: [[e.split()[pos]#e.split('\t')[pos]
                                  for e 
                                  in line.strip().split('\n')]
                                  for line
                                  in cs.open(fn, encoding='utf-8').read().strip().split('\n\n')
                                  ]
                                  #if e.strip() != '']
    
    words = (get_label(train_fn, 0)
              + ([]
                 if valid_fn is None
                 else get_label(valid_fn, 0))
	          + ([]
		         if test_fn is None
		         else get_label(test_fn, 0)))
    labels = (get_label(train_fn, -1)
              + ([]
                 if valid_fn is None
                 else get_label(valid_fn, -1))
              + ([]
                 if test_fn is None
                 else get_label(test_fn, -1)))
    corpus_feat = []
    corpus_words = []
    for lwds, llbs in zip(words, labels):
        X = convdata_helper(lwds, llbs, mode, anno)
        features = apply_feature_templates(X)
        feats = []
        for fields in X:
            corpus_words.append(fields[1])
        for i, ftv in enumerate(features):
            feat = [escape(ft) for ft in ftv['F']]
            feats.append(feat)
        assert len(lwds) == len(llbs)
        assert len(lwds) == len(feats)
        corpus_feat.append(feats)
    feature_to_freq = defaultdict(int)
    for feats in corpus_feat:
        for ftv in feats:
            for ft in ftv:
                feature_to_freq[ft]+=1
    features_to_id = {OOV: 0}
    cur_idx = 1
    for feats in corpus_feat:
        for ftv in feats:
            for ft in ftv:
                if (ft not in features_to_id) and feature_to_freq[ft] > feature_thresh:
                    features_to_id[ft] = cur_idx
                    cur_idx+=1

    word_to_id = {}
    cur_idx = 0
    #print 'construct dict!!'
    for t in corpus_words:
        if not t in word_to_id:
            #print t, cur_idx
            word_to_id[t] = cur_idx
            cur_idx+=1
    
    label_to_id = {}
    cur_idx = 0
    for ilabels in labels:
        for t in ilabels:
            ## very hacky version to filter out all the nominal mentions
            #if t.endswith('NOM'):
            #    t = 'O'
            if not t in label_to_id:
                label_to_id[t] = cur_idx
                cur_idx+=1
    return features_to_id, word_to_id, label_to_id

def print_dict(dictname):
	for k, v in dictname.items():
		print k, v

def save_dicts(dict_file, dict_list):
    pickle.dump(dict_list, open(dict_file, 'wb'))

def load_dicts(dict_file):
    dict_feature, dict_lex, dict_y = pickle.load(open(dict_file, 'rb'))
    return dict_feature, dict_lex, dict_y

def write_predictions(output_dir, filename, predict_test):
    with cs.open(os.path.join(output_dir, os.path.basename(filename)+'.predictions'), 'w', encoding='utf-8') as outf:
        for line in predict_test:
            for tk in line:
                outf.write(tk+'\n')
            outf.write('\n')

# Note: if feature_thresh < 0, means load the feature, lex and y dict.
def loaddata(train_name, valid_name, test_name, feature_thresh=1, mode='char', anno=None, test_label=True):
    def f():
        print "construct dictionaries!"
        dict_feature, dict_lex, dict_y = create_dicts(train_name, valid_name, feature_thresh, test_name, mode, anno)
        train_set = get_data(train_name, dict_feature, dict_lex, dict_y, mode, anno)
        valid_set = get_data(valid_name, dict_feature, dict_lex, dict_y, mode, anno)
        test_set = get_data(test_name, dict_feature, dict_lex, dict_y, mode, anno, test_label)
        dic = {'words2idx':dict_lex, 'labels2idx':dict_y, 'features2idx':dict_feature}
        return [train_set, valid_set, test_set, dic]
    #return (train_feat, train_lex, train_y, valid_feat, valid_lex, valid_y, test_feat, test_lex, test_y, dict_feature, dict_lex, dict_y)
    return f()

def convdata_helper(chars, labels, repre, anno):
    X = []
    ## very hacky version to filter out all the nominal mentions
    #labels = [l if not l.endswith('NOM') else 'O' for l in labels]
    if repre == 'char' and anno == None:
        for c, l in zip(chars, labels):
            X.append((c,c,l))
    else:
        import jieba
        sent = ''.join(chars)
        token_tag_list = jieba.cut(sent) #pseg.cut(sent)
        count = 0
        #for token_str, pos_tag in token_tag_list:
        for token_str in token_tag_list:
            for i,char in enumerate(token_str):
                if repre == 'word':
                    r = token_str
                elif repre == 'charpos':
                    r = char+str(i)
                else:
                    raise ValueError('representation cannot take value %s!\n'%repre)
                if anno == None:
                    fields = (char, r, labels[count])
                else:
                    raise ValueError('representation cannot take value %s!\n'%repre)
                count += 1
                X.append(fields)
    #print X
    return X

def convdata(lines, dict_feature, dict_lex, dict_y, repre, anno, label=True):
    corpus_feat = []
    corpus_lex = []
    corpus_y = []
    print 'in convdata, num lines=%d, feature size=%d, voc size=%d, label size=%d' % (len(lines), len(dict_feature), len(dict_lex), len(dict_y))
    sentences = []
    for line in lines:
        chars = []
        labels = []
        for token in line.split('\n'):
            array = token.split()
            chars.append(array[0])
            if label:
                labels.append(array[-1])
            else:
                labels.append(None)
        X = convdata_helper(chars, labels, repre, anno)
        #print 'conv data:', ' '.join(['-'.join(fld) for fld in X])
        sentences.append(X)
    conll_feature_extract(sentences, dict_feature, dict_lex, dict_y, False, corpus_lex, corpus_y, corpus_feat)
    return [corpus_feat, corpus_lex, corpus_y]

def get_data(fn, dict_feature, dict_lex, dict_y, repre, anno, has_label=True):
    print 'get data from', fn
    with cs.open(fn, 'r', encoding='utf-8') as f:
        return convdata(f.read().strip().split('\n\n'),
                dict_feature,
                dict_lex, 
                dict_y,
                repre,
                anno,
                has_label
                )
	

def get_tokentaggings_of_type_v(tokenization, taggingType):
    return [tt for tt in tokenization.tokenTaggingList if (tt.taggingType and taggingType and tt.taggingType.lower() == taggingType.lower())]

def set_tokentaggings_of_type_v(tokenization, taggingType, prediction, toolname):
    timestamp = long(time.time()*1e6)
    tokens = tokenization.tokenList.tokenList
    new_pred = []
    start = 0
    for i, tk in enumerate(tokens):
        tg = ' '.join(prediction[start:start+len(tk.text)])
        #print tk.text, tg
        new_pred.append(TaggedToken(tokenIndex=i,tag=tg))
        start += len(tk.text)
    assert len(new_pred) == len(tokens)
    #print start, len(prediction)
    assert start == len(prediction)
    new_tokentagging = TokenTagging(
            taggingType=taggingType,
            taggedTokenList=new_pred,
            metadata=AnnotationMetadata(tool=toolname, timestamp=timestamp),
            uuid=generate_UUID())
    tokenization.tokenTaggingList.append(new_tokentagging)

def get_conll_style_tags(tokenization, golden_tags, mode='char', lang='CN'):
    X = []
    if tokenization.tokenList:
        for i, token in enumerate(tokenization.tokenList.tokenList):
            if lang == 'CN':
                tks = token.text #[token.text]
            else:
                tks = [token.text]
            tags = [None] * len(tks)
            if golden_tags != None:
                token_tag = golden_tags.taggedTokenList[i] #[unicode(token_tag_list[i]) for token_tag_list in golden_tags]
                tags = token_tag.tag.split(' ')
            assert len(tags) == len(tks)
            for pos, (tk, tg) in enumerate(zip(tks, tags)):
                # Hacky version
                tag = tg[:9]
                if mode == 'char':
                    fields = [tk, tk, tag]
                elif mode == 'word': 
                    fields = [tk, token.text, tag]
                elif mode == 'charpos':
                    fields = [tk, tk+str(pos), tag]
                else:
                    raise ValueError('representation cannot take mode %s!\n'%mode)
                X.append(fields)
    return X


def get_conll_style_annotated_tags(tokenization, annotations, golden_tags, mode='char', lang='CN'):
    X = []
    place_tags = set(['S', 'B', 'I', 'E', 'N', 'O'])
    if golden_tags:
        for (atype, tag) in annotations:
            assert len(tag.taggedTokenList) == len(golden_tags.taggedTokenList)
    if tokenization.tokenList:
        if lang == 'CN':
            tks = token.text 
        else:
            tks = [token.text]
        for i, token in enumerate(tokenization.tokenList.tokenList):
            tags = [None] * len(tks)
            if golden_tags != None:
                token_tag = golden_tags.taggedTokenList[i] #[unicode(token_tag_list[i]) for token_tag_list in golden_tags]
                tags = token_tag.tag.split(' ')
            assert len(tags) == len(tks)
            annos = [anno.taggedTokenList[i].tag.split(' ') for (atype,anno) in annotations]
            if len(annotations) == 1 and annotations[0][0] == POS:
                annos[0] = annos[0] * len(tags)
            annos.append(tags)
            for pos, anno in enumerate(zip(tks, *annos)):
                if len(tks) == 1:
                    converted_anno = [anno[0]] + [a if (len(a.split('-')) == 2 and a.split('-')[0] in place_tags) or a=='O' or a=='N' else 'S-'+a for a in anno[1:] ]
                elif pos == 0:
                    converted_anno = [anno[0]] + [a if (len(a.split('-')) == 2 and a.split('-')[0] in place_tags) or a=='O' or a=='N' else 'B-'+a for a in anno[1:] ]
                elif pos == len(tks)-1:
                    converted_anno = [anno[0]] + [a if (len(a.split('-')) == 2 and a.split('-')[0] in place_tags) or a=='O' or a=='N' else 'E-'+a for a in anno[1:]]
                else:
                    converted_anno = [anno[0]] + [a if (len(a.split('-')) == 2 and a.split('-')[0] in place_tags) or a=='O' or a=='N' else 'I-'+a for a in anno[1:]]
                rep = char_transform(converted_anno[0])
                if len(annotations) == 1:
                    converted_anno = [converted_anno[0], converted_anno[1], converted_anno[1], converted_anno[2]]
                    converted_anno[1] = converted_anno[1].split('-')[0] + '-word'
                if mode == 'word':
                    converted_anno[0] = tks
                elif mode == 'charpos':
                    converted_anno[0] = converted_anno[0] + str(pos)
                elif mode != 'char':
                    raise ValueError('representation cannot take mode %s!\n'%mode)
                # when rep == 'O', it's regular hanzi
                if rep != 'O':
                    converted_anno[0] = rep
                fields = list([tks[pos]] + converted_anno)
                X.append(fields)
    return X

def get_files(file_dir): 
    if os.path.isdir(file_dir):
	return [(read_communication_from_file(os.path.join(file_dir, fn)), fn) for fn in next(os.walk(file_dir))[2]]
    elif tarfile.is_tarfile(file_dir):
	return [(comm, filename) for (comm, filename) in CommunicationReader(file_dir)]
    elif os.path.isfile(file_dir):
        return [(read_communication_from_file(file_dir), None)]

def apply_feature_templates(sntc):
    if len(sntc[0]) == 3:
        fields_template = 'w r y'
    elif len(sntc[0]) == 4:
        fields_template = 'w r s y'
    elif len(sntc[0]) == 5:
        fields_template = 'w r s p y'
    if fields_template == 'w r y':
        features = feature_extractor(readiter(sntc, fields_template.split(' ')), templates=local_templates)
    else:
        features = feature_extractor(readiter(sntc, fields_template.split(' ')))
    return features

def conll_feature_extract(sentences_conll, features_to_id, word_to_id, label_to_id, no_label, corpus_lex, corpus_y, corpus_feat):
    get_label = lambda sentence, pos: [tok[pos] 
            for tok in sentence]
    upper_bound = len(word_to_id)-1
    '''print 'word to idx:'
    for k,v in word_to_id.iteritems():
        print k,v
    '''
    for sntc in sentences_conll:
        words = [word_to_id.get(wd, random.randint(0,upper_bound)) for wd in get_label(sntc, 1)]
        if len(words) == 0:
            print 'all oov!!!!'
            continue
        if no_label:
            labels = []
        else:
            labels = [label_to_id[lb] for lb in get_label(sntc, -1)]
        train_feat = []
        features = apply_feature_templates(sntc)
        for i, ftv in enumerate(features):
            #if raw_words[i] not in word_to_id:
            #	continue
            feat = [features_to_id[escape(ft)] for ft in ftv['F'] if escape(ft) in features_to_id]
            tft = [escape(ft) for ft in ftv['F'] if escape(ft) in features_to_id]
            if len(feat) == 0:
                tft.append(OOV)
                feat.append(features_to_id[OOV])
            train_feat.append(feat)
        if not no_label:
            assert len(words) == len(labels)
        assert len(words) == len(train_feat)
        #print len(words)
        corpus_lex.append(words)
        corpus_y.append(labels)
        corpus_feat.append(train_feat)
        '''for w, l, f in zip(twords, tlabels, ttrainft):
            print l+'\t'+ '\t'.join(f)
            print ''
        '''
    assert len(corpus_lex) == len(corpus_y)
    assert len(corpus_lex) == len(corpus_feat)


def create_feature_dict(features, feature_thresh):   
    feature_to_freq = defaultdict(int)
    for ftvs in features:
        for ftv in ftvs:
            for ft in ftv['F']:
                feature_to_freq[escape(ft)]+=1
    features_to_id = {OOV: 0}
    cur_idx = 1
    for ftvs in features:
        for ftv in ftvs:
            for ft in ftv['F']:
                ft = escape(ft)
                if (ft not in features_to_id) and feature_to_freq[ft] > feature_thresh:
                    features_to_id[ft] = cur_idx
                    cur_idx+=1
    return features_to_id

def create_lex_dict(words):
    word_to_id = {}
    cur_idx = 0
    for t in words:
	if not t in word_to_id:
            word_to_id[t] = cur_idx
            cur_idx+=1
    return word_to_id


def error_analysis(words, preds, golds, idx_to_word):
	print 'error analysis!!!'
	for w_1sent, p_1sent, g_1sent in zip(words, preds, golds):
	    for w, p, g in zip(w_1sent, p_1sent, g_1sent):
		#if p != g:
		print idx_to_word[w[0]], p, g
            print ''
	print 'end of error analysis!!!!'

# utilities for evaluation:
def eval_ner(pred, gold):
    print 'Evaluating...'
    eval_dict = {}    # value=[#match, #pred, #gold]
    for p_1sent, g_1sent in zip(pred, gold):
        in_correct_chunk = False
        last_pair = ['^', '$']
        for p, g in zip(p_1sent, g_1sent):
            tp = p.split('-')
            tg = g.split('-')
            if len(tp) == 2:
                if tp[1] not in eval_dict:
                    eval_dict[tp[1]] = [0]*3
                if tp[0] == 'B' or tp[0] == 'S':
                    eval_dict[tp[1]][1] += 1
            if len(tg) == 2:
                if tg[1] not in eval_dict:
                    eval_dict[tg[1]] = [0]*3 
                if tg[0] == 'B' or tg[0] == 'S':
                    eval_dict[tg[1]][2] += 1
        
            if p != g or len(tp) == 1:
                if in_correct_chunk and tp[0] != 'I' and tg[0] != 'I' and tp[0] != 'E' and tg[0] != 'E':
                    assert last_pair[0] == last_pair[1]
                    eval_dict[last_pair[0]][0] += 1
                in_correct_chunk = False
                last_pair = ['^', '$'] 
            else:
                if tg[0] == 'B' or tg[0] == 'S':
                    if in_correct_chunk:
                        assert (last_pair[0] == last_pair[1])
                        eval_dict[last_pair[0]][0] += 1
                    last_pair = [tp[-1], tg[-1]]
                if tg[0] == 'B':
                    in_correct_chunk = True
                if tg[0] == 'S':
                    eval_dict[last_pair[0]][0] += 1
                    in_correct_chunk = False
        if in_correct_chunk:
            assert last_pair[0] == last_pair[1]
            eval_dict[last_pair[0]][0] += 1
    agg_measure = [0.0]*3
    agg_counts = [0]*3
    for k, v in eval_dict.items():
        agg_counts = [sum(x) for x in zip(agg_counts, v)]
        prec = float(v[0])/v[1] if v[1] != 0 else 0.0 
        recall = float(v[0])/v[2] if v[2] != 0 else 0.0
        F1 = 2*prec*recall/(prec+recall) if prec != 0 and recall != 0 else 0.0
        agg_measure[0] += prec
        agg_measure[1] += recall
        agg_measure[2] += F1
        print k+':', v[0], '\t', v[1], '\t', v[2], '\t', prec, '\t', recall, '\t', F1
    agg_measure = [v/len(eval_dict) if len(eval_dict) != 0 else 0.0 for v in agg_measure]
    print 'Macro average:', '\t', agg_measure[0], '\t', agg_measure[1], '\t', agg_measure[2]
    prec = float(agg_counts[0])/agg_counts[1] if agg_counts[1] != 0 else 0.0
    recall = float(agg_counts[0])/agg_counts[2] if agg_counts[2] != 0 else 0.0
    F1 = 2*prec*recall/(prec+recall) if prec != 0 and recall != 0 else 0.0
    print 'Micro average:', agg_counts[0], '\t', agg_counts[1], '\t', agg_counts[2], '\t', prec, '\t', recall, '\t', F1 
    return {'p': prec, 'r': recall, 'f1': F1}



def quick_sample(fn):
    with cs.open(fn, 'r', encoding='utf-8') as f, cs.open(fn+'.small', 'w', encoding='utf-8') as outf:
        corpus = f.read().strip().split('\n\n')
        print type(corpus)
        print corpus[1]
        samples = random.sample(corpus, 800)
        for spl in samples:
            outf.write(spl+'\n\n')
        #for item in f.read().strip().split('\n\n'):
         #   print item



if __name__ == '__main__':
    train_set, valid_set, test_set, dicts = loaddata(sys.argv[1], sys.argv[2], sys.argv[3], feature_thresh=1, mode='charpos', anno=None)
    idx2word = dict((k, v) for v, k in dicts['words2idx'].iteritems())
    idx2label = dict((k, v) for v, k in dicts['labels2idx'].iteritems())
    idx2feature = dict((k, v) for v, k in dicts['features2idx'].iteritems())
    for f,l,y in zip (test_set[0], test_set[1], test_set[2]):
    #    print [' '.join([idx2feature[fi] for fi in i]) for i in f]
        print ' '.join([idx2word[li] for li in l])
        print ' '.join([idx2label[yi] for yi in y])
