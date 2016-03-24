import codecs as cs
import os
import sys
import tarfile
import cPickle as pickle
import time
import uuid
import random
from collections import defaultdict
from concrete.util.file_io import *
from concrete.util import generate_UUID
from concrete.inspect import *
from weiboNER_features import *
from concrete import (
	AnnotationMetadata,
	TextSpan,
	Token,
	Tokenization,
	TokenizationKind,
	TokenList,
	TokenTagging,
	TaggedToken,
	EntityMentionSet,
	EntityMention,
	EntitySet,
	Entity,
	TokenRefSequence
	)


OOV = '_OOV_'
GOLD_TAG = 'GoldNER'
PRED_TAG = 'NER'
fields_template = 'w r y'  #['w', 'r', 'y']
task = 'ner'
random.seed(0)

def create_dicts(train_fn, valid_fn, feature_thresh, test_fn=None):
    get_label = lambda fn, pos: [e.split()[pos] #e.split('\t')[pos]
                                  for e
                                  in cs.open(fn, encoding='utf-8').read().strip().split('\n')
                                  if e.strip() != '']
    get_features = lambda fn, start: [e.split()[start:]
                                  for e
                                  in cs.open(fn, encoding='utf-8').read().strip().split('\n')
                                  if e.strip() != ''] #e.strip() != '']
    
    words = (get_label(train_fn, 0)
              + ([]
                 if valid_fn is None
                 else get_label(valid_fn, 0))
	      + ([]
		 if test_fn is None
		 else get_label(test_fn, 0)))
    labels = (get_label(train_fn, 1)
              + ([]
                 if valid_fn is None
                 else get_label(valid_fn, 1))
              + ([]
                 if test_fn is None
                 else get_label(test_fn, 1)))
    features = (get_features(train_fn, 2)
              + ([]
                 if valid_fn is None
                 else get_features(valid_fn, 2))
              + ([]
                 if test_fn is None
                 else get_features(test_fn, 2)))
    feature_to_freq = defaultdict(int)
    for ftv in features:
	for ft in ftv:
            feature_to_freq[ft]+=1
    features_to_id = {OOV: 0}
    cur_idx = 1
    for ftv in features:
	for ft in ftv:
            if (ft not in features_to_id) and feature_to_freq[ft] > feature_thresh:
                features_to_id[ft] = cur_idx
                cur_idx+=1

    word_to_id = {}
    cur_idx = 0
    for t in words:
	if not t in word_to_id:
            word_to_id[t] = cur_idx
            cur_idx+=1
    
    label_to_id = {}
    cur_idx = 0
    for t in labels:
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


# Note: if feature_thresh < 0, means load the feature, lex and y dict.
def loaddata(train_name, valid_name, test_name, feature_thresh=1, dict_file=None):
    def f():
	if feature_thresh >= 0:
	    print "construct dictionaries!"
	    dict_feature, dict_lex, dict_y = create_dicts(train_name, valid_name, feature_thresh, test_fn=test_name)
            #save_dicts(dict_file, (dict_feature, dict_lex, dict_y)) 
        else:
            print "load dictionaries!"
	    dict_feature, dict_lex, dict_y = load_dicts(dict_file)
	#for k, v in dict_feature.items():
	#	print k, v
	#exit(0)
	#print_dict(dict_y)
	train_feat, train_lex, train_y = get_data(train_name, dict_feature, dict_lex, dict_y)
        valid_feat, valid_lex, valid_y = get_data(valid_name, dict_feature, dict_lex, dict_y)
        test_feat, test_lex, test_y = get_data(test_name, dict_feature, dict_lex, dict_y)
        return (train_feat, train_lex, train_y, valid_feat, valid_lex, valid_y, test_feat, test_lex, test_y, dict_feature, dict_lex, dict_y)
    return f()


def convdata(lines, dict_feature, dict_lex, dict_y):
    train_feat = []
    train_lex = []
    train_y = []
    print 'in convdata, num lines=%d, voc size=%d, label size=%d' % (len(lines), len(dict_feature), len(dict_y))
    for line in lines:
	vec_feat = []
        vec_lex = []
        vec_y = []
        for token in line.split('\n'):
	    array = token.split('\t')
	    #lex_id = dict_feature[array[0]] if array[0] in dict_feature else dict_feature[OOV]
	    #print array
	    features = [dict_feature[t] for t in array[2:] if t in dict_feature]
	    if len(features) == 0:
		    features.append(dict_feature[OOV])
	    vec_feat.append(features)
	    x_id = dict_lex[array[0]]
	    vec_lex.append(x_id)
            y_id = dict_y[array[1]]
            vec_y.append(y_id)
	assert len(vec_feat) == len(vec_y)
        assert len(vec_lex) == len(vec_y)
	'''sys.stderr.write("in convdata, line=\n")
	sys.stderr.write(str( vec_lex)+'\n')
	sys.stderr.write(str( vec_y)+'\n')	
	'''
	train_feat.append(vec_feat)
	train_lex.append(vec_lex)
        train_y.append(vec_y)
    return train_feat, train_lex, train_y

def get_data(fn, dict_feature, dict_lex, dict_y):
    with cs.open(fn, 'r', encoding='utf-8') as f:
        return convdata(f.read().strip().split('\n\t\n'),
                        dict_feature,
			dict_lex, 
                        dict_y
                        )

	
def read_concrete(comm, no_label=False, mode='char'):
    #comm = read_communication_from_file(concrete_file)
    tokenizations = get_tokenizations(comm)
    sentences = []
    for tokenization in tokenizations:
	if no_label:
	    X = get_conll_style_tags(tokenization, None, mode)
            sentences.append(X)
	else:    
    	    gold_tags = get_tokentaggings_of_type_v(tokenization, GOLD_TAG)
	    X = get_conll_style_tags(tokenization, gold_tags[0], mode)
            sentences.append(X)
    return sentences

def get_tokentaggings_of_type_v(tokenization, taggingType):
    return [tt for tt in tokenization.tokenTaggingList if (tt.taggingType and tt.taggingType.lower() == taggingType.lower())]
    
def get_conll_style_tags(tokenization, token_tags, mode='char'):
    X = []
    if tokenization.tokenList:
	for i, token in enumerate(tokenization.tokenList.tokenList):
	    tags = [None] * len(token.text)
	    if token_tags != None:
	        token_tag = token_tags.taggedTokenList[i] #[unicode(token_tag_list[i]) for token_tag_list in token_tags]
	        tags = token_tag.tag.split(' ')
	    assert len(tags) == len(token.text)
	    for pos, (tk, tg) in enumerate(zip(token.text, tags)):
		if tg != None and tg.endswith('PRO'):
		    tg = 'O'
		if mode == 'char':
	    	    fields = (tk, tk, tg)
	        elif mode == 'word': 
		    fields = (tk, token.text, tg)
	        elif mode == 'charpos':
		    fields = (tk, tk+str(pos), tg)
		else:
	            raise ValueError('representation cannot take mode %s!\n'%mode)
	        #print fields
	    #fields.extend(token_tags)
	    	X.append(fields)
    return X


def load_data_concrete(train_dir, dev_dir, test_dir, eval_test=False, feature_thresh=1, mode='char'):
    train_files = train_dir 
    valid_files = dev_dir  
    test_files  = test_dir
    dict_feature, dict_lex, dict_y = create_dicts_concrete(train_files, valid_files, feature_thresh, test_fn=test_files, test_label=eval_test, mode=mode)
    train_feat, train_lex, train_y = get_data_concrete(train_files, dict_feature, dict_lex, dict_y, mode=mode)
    valid_feat, valid_lex, valid_y = get_data_concrete(valid_files, dict_feature, dict_lex, dict_y, mode=mode)
    test_feat, test_lex, test_y = get_data_concrete(test_files , dict_feature, dict_lex, dict_y, no_label=(not eval_test), mode=mode)
    return (train_feat, train_lex, train_y, valid_feat, valid_lex, valid_y, test_feat, test_lex, test_y, dict_feature, dict_lex, dict_y)

def get_files(file_dir): 
    #print file_dir
    if os.path.isdir(file_dir):
	return [(read_communication_from_file(os.path.join(file_dir, fn)), fn) for fn in next(os.walk(file_dir))[2]]
    elif tarfile.is_tarfile(file_dir):
	#for (comm, filename) in CommunicationReader(file_dir):
	#	print filename
	return [(comm, filename) for (comm, filename) in CommunicationReader(file_dir)]

def get_data_concrete(file_dir, features_to_id, word_to_id, label_to_id, no_label=False, mode='char'):
    get_label = lambda sentences, pos: [tok[pos] 
                                  for sentence in sentences 
				  for tok in sentence
                                  ]
    corpus_lex = []
    corpus_y = []
    corpus_feat = []
    upper_bound = len(word_to_id)-1
    for (file, fname) in get_files(file_dir):
	sentences = read_concrete(file, no_label, mode)
	#if len(sentences) == 0:
	#	print 'file', fname, 'do not have sentences in it!'
	#	continue
	#raw_words = get_label(sentences, 1)
	words = [word_to_id.get(wd, random.randint(0,upper_bound)) for wd in get_label(sentences, 1)]
	#print words
	if len(words) == 0:
		print fname, 'all oov!!!!'
		continue
	if no_label:
		labels = []
	else:
		labels = [label_to_id[lb] for lb in get_label(sentences, 2)]
	features = []
	train_feat = []
	for sentence in sentences:
	    features.extend(feature_extractor(readiter(sentence, fields_template.split(' '))) )
	for i, ftv in enumerate(features):
		#if raw_words[i] not in word_to_id:
		#	continue
		feat = [features_to_id[ft] for ft in ftv['F'] if ft in features_to_id]
		if len(feat) == 0:
			feat.append(features_to_id[OOV])
		train_feat.append(feat)
	if not no_label:
		assert len(words) == len(labels)
	assert len(words) == len(train_feat)
	corpus_lex.append(words)
	corpus_y.append(labels)
	corpus_feat.append(train_feat)
    assert len(corpus_y) == len(corpus_lex)
    print 'in get_data_concrete, num lines=%d, voc size=%d, label size=%d' % (len(corpus_y), len(features_to_id), len(label_to_id))
    return corpus_feat, corpus_lex, corpus_y
    # Change the crf_ner code to handle no label case

def create_dicts_concrete(train_files, valid_files, feature_thresh, test_fn=None, test_label=False, mode='char'):  
    get_label = lambda sentences, pos: [tok[pos] 
                                  for sentence in sentences 
				  for tok in sentence
                                  ]
    get_features = lambda sentences: [feature_extractor(X)
                                  for X in sentences
                                  ]
    words = []
    labels = []
    features = []
    all_labeled_files = get_files(train_files) + ([] if valid_files is None else get_files(valid_files))
    all_unlabeled_files = [] if test_fn is None else get_files(test_fn)
    if test_label:
	    all_labeled_files += all_unlabeled_files
	    all_unlabeled_files = []
    for (file, fn) in all_labeled_files:
	#print fn
	sentences = read_concrete(file, False, mode)
	words += get_label(sentences, 1)
	labels += get_label(sentences, 2)
	features.extend(feature_extractor(readiter(sentence, ['w', 'y'])) for sentence in sentences)
    for (file, _) in all_unlabeled_files:
	sentences = read_concrete(file, True, mode)
	words += get_label(sentences, 1)
	#labels += get_label(sentences, 2)
	features.extend(feature_extractor(readiter(sentence, ['w', 'y'])) for sentence in sentences)
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

    word_to_id = {}
    cur_idx = 0
    for t in words:
	if not t in word_to_id:
            word_to_id[t] = cur_idx
            cur_idx+=1
    
    label_to_id = {}
    cur_idx = 0
    for t in labels:
        if not t in label_to_id:
            label_to_id[t] = cur_idx
            cur_idx+=1
    #print 'feature size:', len(features_to_id), 'lexical size:', len(word_to_id), 'label size:', len(label_to_id)
    '''print 'feature map:'
    for k,v in features_to_id.items():
	    print k,v
    print 'word map:'
    for k,v in word_to_id.items():
	    print k,v
    for k,v in label_to_id.items():
	    print k,v
    '''
    return features_to_id, word_to_id, label_to_id

def write_data_concrete(input_tar, output_dir, predictions):
    #print output_dir, input_tar
    #print os.path.join(output_dir, os.path.basename(input_tar))
    writer = CommunicationWriterTGZ()
    writer.open(os.path.join(output_dir, 'golden_horse_'+os.path.basename(input_tar)))
    for i, (comm, fn) in enumerate(get_files(input_tar)):
	prediction = predictions[i]
	update_concrete_file_write(comm, fn, writer, prediction)
    writer.close()

def update_concrete_file_write(comm, filename, writer, prediction):
    toolname = 'Violet_NER_annotator'
    timestamp = int(time.time())
    #comm = read_communication_from_file(outfile)
    mention_list = []
    for section in comm.sectionList:
	for sentence in section.sentenceList:
	    start = 0
	    pred_ner_tags = []
	    tknzation = sentence.tokenization
	    in_NE = False
	    ne_type = ''
	    tokenization_id = None
	    token_idx_list = []
	    ne_text = []
	    for i, tk in enumerate(tknzation.tokenList.tokenList):
		pred_tags = ' '.join(prediction[start:start+len(tk.text)])
		if in_NE:
			#print 'in NE,', prediction[start:start+len(tk.text)]
			for i, tag in enumerate(prediction[start:start+len(tk.text)]):
				if tag != 'I-' + ne_type:
					if i != 0:
						token_idx_list.append(i)
						ne_text.append(tk.text)
					entity_tokens = TokenRefSequence(tokenizationId=tokenization_id, tokenIndexList=token_idx_list)
					e_type, p_type = ne_type.split('.') if '.' in ne_type else (ne_type, 'NAM')
					#print token_idx_list, ne_text, e_type, p_type
					e_mention = EntityMention(uuid=generate_UUID(), tokens=entity_tokens, entityType=e_type, phraseType=p_type, text=''.join(ne_text))
					mention_list.append(e_mention)
					tokenization_id = None
					token_idx_list = []
					ne_text = []
					ne_type = ''
					in_NE = False
					break
		if not in_NE and 'B-' in pred_tags:
			#print 'not in NE,', prediction[start:start+len(tk.text)]
			in_NE = True
			for tag in prediction[start:start+len(tk.text)]:
				#print tag
				if tag.startswith('B-'):
					ne_type = tag.split('-')[1]
					tokenization_id = tknzation.uuid
					token_idx_list.append(i)
					ne_text.append(tk.text)
					break
			#print token_idx_list, ne_text
			if prediction[start+len(tk.text)-1] != 'I-'+ne_type:
				entity_tokens = TokenRefSequence(tokenizationId=tokenization_id, tokenIndexList=token_idx_list)
				e_type, p_type = ne_type.split('.') if '.' in ne_type else (ne_type, 'NAM')
				e_mention = EntityMention(uuid=generate_UUID(), tokens=entity_tokens,entityType=e_type,phraseType=p_type,text=''.join(ne_text))
				mention_list.append(e_mention)
				tokenization_id = None
				token_idx_list = []
				ne_text = []
				ne_type = ''
				in_NE = False
		start += len(tk.text)
		pred_ner_tags.append(TaggedToken(tokenIndex=i, tag=pred_tags))
	    pner_tokentagging = TokenTagging(
		    taggingType=PRED_TAG,
		    taggedTokenList=pred_ner_tags,
		    metadata=AnnotationMetadata(tool=toolname, timestamp=timestamp),
		    uuid=generate_UUID())
	    tknzation.tokenTaggingList.append(pner_tokentagging)
    entity_list = [Entity(uuid=generate_UUID(),type=mention.entityType,canonicalName=mention.text,mentionIdList=[mention.uuid]) for mention in mention_list]
    entity_mention_set = EntityMentionSet(uuid=generate_UUID(),metadata=AnnotationMetadata(tool=toolname, timestamp=timestamp),mentionList=mention_list)
    entity_set = EntitySet(uuid=generate_UUID(),metadata=AnnotationMetadata(tool=toolname, timestamp=timestamp),entityList=entity_list,mentionSetId=entity_mention_set.uuid)
    comm.entityMentionSetList = [entity_mention_set]
    comm.entitySetList = [entity_set]
    #print filename
    writer.write(comm, filename)
    #write_communication_to_file(comm, outfile)
    #return comm

def error_analysis(words, preds, golds, idx_to_word):
	print 'error analysis!!!'
	for w_1sent, p_1sent, g_1sent in zip(words, preds, golds):
	    for w, p, g in zip(w_1sent, p_1sent, g_1sent):
		if p != g:
			print idx_to_word[w], p, g
	print 'end of error analysis!!!!'

# utilities for evaluation:
def eval_ner(pred, gold):
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
			if tp[0] == 'B':
				eval_dict[tp[1]][1] += 1
		if len(tg) == 2:
			if tg[1] not in eval_dict:
				eval_dict[tg[1]] = [0]*3 
			if tg[0] == 'B':
				eval_dict[tg[1]][2] += 1
		# hit a row outside of a contineous NE chunk
		# either p == g == 'O' or p != g
		# in this case, it does not enter another NE chunk, change in_correct_chunk to false and change last_pair to default different value. 
		# check whether previous block could be a correctly predicted NE chunk.
		if p != g or len(tp) == 1:
			if in_correct_chunk and tp[0] != 'I' and tg[0] != 'I':
				assert last_pair[0] == last_pair[1]
				eval_dict[last_pair[0]][0] += 1
			in_correct_chunk = False
			last_pair = ['^', '$'] 
		# p == g and p == g != 'O'
		else:
			# start a new chunk
			if tg[0] == 'B':
				if in_correct_chunk:
					assert (last_pair[0] == last_pair[1])
					eval_dict[last_pair[0]][0] += 1
				in_correct_chunk = True
				last_pair = [tp[-1], tg[-1]]
	    if in_correct_chunk:
		    assert last_pair[0] == last_pair[1]
		    eval_dict[last_pair[0]][0] += 1
	agg_measure = [0.0]*3
	agg_counts = [0]*3
	#print 'evaluate dict:', eval_dict
	#print 'violet\'s evaluation: #match\t #pred\t #gold\t precision\t recall\t F1'
	for k, v in eval_dict.items():
		agg_counts = [sum(x) for x in zip(agg_counts, v)]
		prec = float(v[0])/v[1] if v[1] != 0 else 0.0 
		recall = float(v[0])/v[2] if v[2] != 0 else 0.0
		F1 = 2*prec*recall/(prec+recall) if prec != 0 and recall != 0 else 0.0
		agg_measure[0] += prec
		agg_measure[1] += recall
		agg_measure[2] += F1
		print k+':', v[0], '\t', v[1], '\t', v[2], '\t', prec, '\t', recall, '\t', F1
	agg_measure = [v/len(eval_dict) for v in agg_measure]
	print 'Macro average:', '\t', agg_measure[0], '\t', agg_measure[1], '\t', agg_measure[2]
	prec = float(agg_counts[0])/agg_counts[1] if agg_counts[1] != 0 else 0.0
	recall = float(agg_counts[0])/agg_counts[2] if agg_counts[2] != 0 else 0.0
	F1 = 2*prec*recall/(prec+recall) if prec != 0 and recall != 0 else 0.0
	print 'Micro average:', agg_counts[0], '\t', agg_counts[1], '\t', agg_counts[2], '\t', prec, '\t', recall, '\t', F1 
	return {'p': prec, 'r': recall, 'f1': F1} 


if __name__ == '__main__':
    #read_concrete(sys.argv[1])
    #create_dicts_concrete(sys.argv[1], sys.argv[1], feature_thresh=0, test_fn=sys.argv[1])
    load_data_concrete(sys.argv[1], sys.argv[2], sys.argv[3], mode='word')
