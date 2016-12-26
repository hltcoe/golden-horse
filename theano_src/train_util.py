#!/usr/bin/env python

"""
The implementation of the shared model training code.
"""

import sys
import datetime
import time
import codecs
import theano
import theano.tensor as T
import numpy as np
import random
import warnings
import cPickle as pickle
from neural_lib_crf import ArrayInit

def np_floatX(data):
    return np.asarray(data, dtype=config.floatX)

def dict_from_argparse(apobj):
    return dict(apobj._get_kwargs()) 

def shuffle(lol, seed):
    '''
    lol :: list of list as input
    shuffle inplace each list in the same order by ensuring that we use the same state for every run of shuffle.
    '''
    random.seed(seed)
    state = random.getstate()
    for l in lol:
        random.setstate(state)
        random.shuffle(l)

def read_matrix_from_pkl(fn, dic):
    '''
    Assume that the file contains words in first column,
    and embeddings in the rest and that dic maps words to indices.
    '''
    voc, data = pickle.load(codecs.open(fn))
    dim = len(data[0])
    # NOTE: The norm of onesided_uniform rv is sqrt(n)/sqrt(3)
    # Since the expected value of X^2 = 1/3 where X ~ U[0, 1]
    # => sum(X_i^2) = dim/3
    # => norm       = sqrt(dim/3)
    # => norm/dim   = sqrt(1/3dim)
    multiplier = np.sqrt(1.0/(3*dim))
    M = ArrayInit(ArrayInit.onesided_uniform, multiplier=1.0/dim).initialize(len(dic), dim)
    not_in_dict = 0
    for i, e in enumerate(data):
        r = np.array([float(_e) for _e in e])
	if voc[i] in dic:
            idx = dic[voc[i]]
	    M[idx] = (r/np.linalg.norm(r)) * multiplier
        else:
	    not_in_dict += 1
    print 'load embedding! %d words, %d not in the dictionary. Dictionary size: %d' %(len(voc), not_in_dict, len(dic))
    return M

def read_matrix_from_file(fn, dic, ecd='utf-8'):
    multiplier = np.sqrt(1.0/3)
    not_in_dict = 0
    with codecs.open(fn, encoding=ecd, errors='ignore') as inf:
        row, column = inf.readline().rstrip().split()
        dim = int(column)
        #print row, column
        idx_map = dict()
        line_count = 0
        M = ArrayInit(ArrayInit.onesided_uniform, multiplier=1.0/dim).initialize(len(dic), dim)
        for line in inf:
            elems = line.rstrip().split(' ')
            if elems[0] in dic:
                idx = dic[elems[0]]
                vec_elem = elems[1:]
                r = np.array([float(_e) for _e in vec_elem])
                M[idx] = (r/np.linalg.norm(r)) * multiplier 
                idx_map[idx] = line_count
            else:
                not_in_dict += 1
            line_count += 1
        print 'load embedding! %s words, %d not in the dictionary. Dictionary size: %d' %(row, not_in_dict, len(dic))
        #print M.shape, len(idx_map)
        return M, idx_map 

def read_idxmap_from_file(fn):
	idx_map = dict()
	with open(fn, 'r') as inf:
		for line in inf:
			elems = line.strip().split()
			idx_map[elems[0]] = elems[1]


def write_matrix_to_file(fn, matrix, idx_map):
    with open(fn, 'w') as outf:  #, open(fn+'.idx', 'w') as idxf:
	dim = matrix.shape
	outf.write(str(len(idx_map)) + ' ' + str(dim[1]) + '\n')
	for i, row in enumerate(matrix):
	    if i in idx_map:
		outf.write(str(idx_map[i]))
		for el in row:
			outf.write(' ' + str(el))
		outf.write('\n')

def conv_emb(lex_arry, M_emb, win_size):
    lexv = []
    for x in lex_arry:
	embs = _conv_x(x, M_emb, win_size)
	lexv.append(embs)
    return lexv

def conv_data(feature_arry, lex_arry, label_arry, win_size, feat_size):
    fv = []
    lexv = []
    labv = []
    for i, (f, x, y) in enumerate(zip(feature_arry, lex_arry, label_arry)):
	words = x #_conv_x(x, M_emb, win_size)
        features = _conv_feat(f, feat_size)
        labels = _conv_y(y)
	fv.append(features)
	lexv.append(words)
	labv.append(labels)
    return fv, lexv, labv

def _conv_feat(x, feat_size):
    lengths = [len(elem) for elem in x]
    max_len = max(lengths)
    features = np.ndarray((len(x), max_len)).astype('int32')
    for i, feat in enumerate(x):
	fpadded = feat + [feat_size] * (max_len-len(feat))
	features[i] = fpadded
    #print words.shape
    return features 

def _contextwin(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence

    l :: array containing the word indexes

    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)

    lpadded = win // 2 * [-1] + l + win // 2 * [-1]
    out = [lpadded[i:(i + win)] for i in range(len(l))]

    assert len(out) == len(l)
    return out

def _conv_x(x, M_emb, window_size):
    x = M_emb[x]
    emb_dim = M_emb.shape[1]
    for line in x:
	assert len(line) == emb_dim
    cwords = _contextwin(x, window_size)
    words = np.ndarray((len(x), emb_dim*window_size)).astype('float32')
    for i, win in enumerate(cwords):
        words[i] = win[0]
    return words

def _conv_y(y):
    labels = np.ndarray(len(y)).astype('int32')
    for i, label in enumerate(y):
        labels[i] = label
    return labels

def _make_temp_storage(name, tparams):
    return [theano.shared(p.get_value() * 0., name= (name%k)) for k, p in tparams]

def sgd(lr, tparams, grads, x_f, x_w, y, cost, build_cost_f=True):
    # First we make shared variables from everything in tparams
    true_grad = _make_temp_storage('%s_grad', tparams)
    f_cost = theano.function([x_f, x_w, y],
                             (cost if build_cost_f else []),
                             updates=[(tg, g) for tg, g in zip(true_grad, grads)],
                             on_unused_input='warn',
			     name='f_cost')
    f_update = theano.function([lr],
                               [],
                               updates=[(p, p - lr * g)
                                        for (k, p), g
                                        in zip(tparams, true_grad)],
                               on_unused_input='warn',
                               name='f_update')
    return f_cost, f_update

def build_optimizer(lr, tparams, grads, x_f, x_w, y, y_pred, cost, optimizer, build_cost_f=True):
    '''
    lr: Learning Rate.
    tparams: Usually a StackConfig object which behaves like a dictionary.
    grads, x, y, y_pred, cost: Outputs of MetaStackMaker, theano variables or None.
      : x      = absolute_input_tv
      : y      = gold_y
      : y_pred = pred_y
      : cost   = cost_or_gradarr
      : grads  = grads
    optimizer: Either the sgd or adadelta function.

    In my case I need to interface between the inside-outside code
    and the theano functions.
    '''
    filtered_tparams = [(k,p)
                        for k, p
                        in tparams.iteritems()
                        if hasattr(p, 'is_regularizable')]
    f_cost, f_update = optimizer(lr, filtered_tparams, grads, x_f, x_w, y, cost,
                                 build_cost_f=build_cost_f)
    f_classify = theano.function(inputs=[x_f, x_w],
                                 outputs=y_pred,
                                 on_unused_input='warn',
                                 name='f_classify')
    return f_cost, f_update, f_classify


def create_circuit(_args, StackConfig, build_cost_f=True):
    cargs = dict_from_argparse(_args)
    cargs = StackConfig(cargs)
    x_f, x_w, y, y_pred, cost, grads, f_debug = _args.circuit(cargs)
    if _args.verbose == 2:
      print "\n", "Printing Configuration after StackConfig:"
      for k in cargs:
	  if type(cargs[k]) == dict:
	      print k, 'dict size:', len(cargs[k])
          else:
	      print k, cargs[k]
    lr = T.scalar('lr')
    f_cost, f_update, f_classify = build_optimizer(lr, cargs, grads, x_f, x_w, y, y_pred, cost, _args.optimizer,
                                                   build_cost_f=build_cost_f)
    return (f_cost, f_update, f_classify, f_debug, cargs)

def save_parameters(path, params):
    pickle.dump(params, open(path, 'w'))

def load_params(path, params):
    pp = pickle.load(open(path, 'r'))
    for kk, vv in pp.iteritems():
        print 'updating parameter:', kk
        params[kk]=pp[kk]
    return params

