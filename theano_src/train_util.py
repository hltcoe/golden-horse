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
from theano import config
from sighan_ner import eval_ner
from neural_lib import ArrayInit

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

def convert_id_to_word(corpus, idx2label):
    return [[idx2label[word] for word in sentence]
            for sentence
            in corpus]

def add_arg(name, default=None, **kwarg):
    assert hasattr(add_arg, 'arg_parser'), "You must register an arg_parser with add_arg before calling it"
    import ast
    if 'action' in kwarg:
        add_arg.arg_parser.add_argument(name, default=default, **kwarg)
    elif type(default)==bool:
        add_arg.arg_parser.add_argument(name, default=default, type=ast.literal_eval, **kwarg)  
    else:
        add_arg.arg_parser.add_argument(name, default=default, type=type(default), **kwarg)
    return

def add_arg_to_L(L, name, default=None, **kwarg):
    L.append(name[2:])
    add_arg(name, default, **kwarg)
    return

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
        r = np.array([np_floatX(_e) for _e in e])
        if voc[i] in dic:
            idx = dic[voc[i]]
            M[idx] = (r/np.linalg.norm(r)) * multiplier
        else:
            not_in_dict += 1
    print 'load embedding! %d words, %d not in the dictionary. Dictionary size: %d' %(len(voc), not_in_dict, len(dic))
    return M

def batch_run_func(corpora, func, *parameters ):
    converted_corpora = []
    for corpus in corpora:
        converted_corpora.append(func(corpus, *parameters))
    return converted_corpora

def read_matrix_from_file(fn, dic, ecd='utf-8'):
    not_in_dict = 0
    with codecs.open(fn, encoding=ecd, errors='ignore') as inf:
        row, column = inf.readline().rstrip().split()
        dim = int(column)
        multiplier = np.sqrt(1.0/3)
        #print row, column
        idx_map = dict()
        line_count = 0
        M = ArrayInit(ArrayInit.onesided_uniform, multiplier=1.0/dim).initialize(len(dic)+2, dim)
        for line in inf:
            elems = line.rstrip().split(' ')
            if elems[0] in dic:
                idx = dic[elems[0]]
                vec_elem = elems[1:] #.split(',')
                r = np.array([np_floatX(_e) for _e in vec_elem])
                M[idx] = (r/np.linalg.norm(r)) * multiplier 
                idx_map[idx] = line_count
            else:
                not_in_dict += 1
            line_count += 1
    print 'load embedding! %s words, %d not in the dictionary. Dictionary size: %d' %(row, not_in_dict, len(dic))
    print 'embedding matrix shape:', M.shape, 'word map size:', len(idx_map)
    return M, idx_map 

def read_matrix_and_idmap_from_file(fn, dic=dict(), ecd='utf-8'):
    idx_map = dic.copy()
    with codecs.open(fn, encoding=ecd, errors='ignore') as inf:
        row, column = inf.readline().rstrip().split(' ')
        dim = int(column)
        multiplier = 1 #np.sqrt(1.0/3) #(2.0/dim)
        #print row, column
        # note: here +3 include OOV, BOS and EOS
        content = [] 
        not_in_dict = 0
        for line in inf:
            elems = line.rstrip().split(' ')
            content.append(elems)
            if elems[0] not in idx_map:
                idx_map[elems[0]] = len(idx_map) 
                not_in_dict += 1
        print 'load embedding! %s words, %d not in the dictionary. Dictionary size: %d' %(row, not_in_dict, len(dic))
        assert len(idx_map) == len(dic) + not_in_dict
        M = ArrayInit(ArrayInit.onesided_uniform, multiplier=1.0/dim).initialize(len(idx_map), dim) #*2)
        for elems in content:
            if elems[0] in idx_map:
                idx = idx_map[elems[0]]
                vec_elem = elems[1:] #.split(',')
                #print len(vec_elem), elems
                r = np.array([np_floatX(_e) for _e in vec_elem])
                M[idx,:dim] = (r/np.linalg.norm(r)) * multiplier 
            else:
                print 'error in load embedding matrix!!' 
    print 'embedding matrix shape:', M.shape, 'word map size:', len(idx_map)
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
        #assert matrix.shape[0] == len(idx_map)
        outf.write(str(len(idx_map)) + ' ' + str(dim[1]) + '\n')
        for i, row in enumerate(matrix):
            if i in idx_map:
                #print idx_map[i]
                outf.write(str(idx_map[i]))
                for el in row:
                    outf.write(' ' + str(el))
                outf.write('\n')
        #for k, v in idx_map.items():
        #	idxf.write(str(k)+' '+str(v)+'\n')

def conv_emb(lex_arry, M_emb, win_size):
    lexv = []
    for x in lex_arry:
        embs = _conv_x(x, M_emb, win_size)
        lexv.append(embs)
    return lexv

def conv_data(corpus, win_l, win_r, feat_size, label_size):
    fv = []
    lexv = []
    labv = []
    temp_f, temp_lex, temp_y = corpus
    for i, (f,x,y) in enumerate(zip(temp_f, temp_lex, temp_y)):
        words = conv_x(x, win_l, win_r)
        features = _conv_feat(f, feat_size)
        labels = _conv_y(y, label_size)
        #print len(words), len(labels)
        assert len(words) == len(labels)
        fv.append(features)
        lexv.append(words)
        labv.append(labels)
    return [fv, lexv, labv]

def _conv_feat(x, feat_size):
    lengths = [len(elem) for elem in x]
    max_len = max(lengths)
    features = np.ndarray((len(x), max_len)).astype('int32')
    #features = np.ndarray((len(x)+2, max_len)).astype('int32')
    #features[0] = [feat_size] + [feat_size+2] * (max_len-1)
    #features[-1] = [feat_size + 1] + [feat_size+2] * (max_len-1)
    for i, feat in enumerate(x):
        fpadded = feat + [feat_size] * (max_len-len(feat))
        features[i] = fpadded
        #fpadded = feat + [feat_size+2] * (max_len-len(feat))
        #features[i+1] = fpadded
    #print words.shape
    return features 

def _contextwin(l, win_l, win_r):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence

    l :: array containing the word indexes

    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert win_l <= 0
    assert win_r >= 0
    l = list(l)
    
    win_size = win_r - win_l + 1
    lpadded = -win_l * [-2] + l + win_r * [-1]
    out = [lpadded[i:(i + win_size)] for i in range(len(l))]

    assert len(out) == len(l)
    return out

def _conv_x(x, M_emb, window_size):
    x = M_emb[x]
    emb_dim = M_emb.shape[1]
    for line in x:
        assert len(line) == emb_dim
    cwords = _contextwin(x, -window_size // 2, window_size // 2)
    words = np.ndarray((len(x), emb_dim*window_size)).astype('float32')
    for i, win in enumerate(cwords):
        #print emb_dim, window_size, len(win[0])
        #print win
        #assert len(win[0]) == emb_dim*window_size
        words[i] = win[0]
    return words

def _conv_y(y, labelsize):
    labels = np.ndarray(len(y)).astype('int32')
    #labels = np.ndarray(len(y)+2).astype('int32')
    #labels[0] = labelsize-2
    #labels[-1] = labelsize-1
    for i, label in enumerate(y):
        labels[i] = label
        #labels[i+1] = label
    assert len(labels) == len(y) #+ 2
    return labels

def conv_x(x, window_l, window_r):
    #print 'in conv_x, window_l=', window_l, 'window_r=', window_r
    x = list(x)
    #x = [vocsize] + x + [vocsize + 1]
    cwords = _contextwin(x, window_l, window_r)
    words = np.ndarray((len(x), window_r-window_l+1)).astype('int32')
    for i, win in enumerate(cwords):
        words[i] = win
    return words

def _make_temp_storage(name, tparams):
    return [theano.shared(p.get_value() * 0., name= (name%(p.name))) for p in tparams]

def sgd(lr, tparams, grads, cost, prefix, *input_params): #x_f, x_w, y, 
    # First we make shared variables from everything in tparams
    true_grad = _make_temp_storage('%s_grad', tparams)
    #print 'in sgd, input:', input_params
    #print true_grad
    assert len(true_grad) == len(grads)
    print 'prefix=', prefix, 'input params:', input_params
    f_cost = theano.function(list(input_params),
            cost,
            updates=[(tg, g) for tg, g in zip(true_grad, grads)],
            on_unused_input='warn',
            name=prefix+'_sgd_f_cost')
    f_update = theano.function([lr],
            [],
            updates=[(p, p - lr * g)
                for p, g
                in zip(tparams, true_grad)],
            on_unused_input='warn',
            name=prefix+'_sgd_f_update')
    return f_cost, f_update


def adadelta(lr, tparams, grads, cost, *input_params): # x_f, x_w, y,
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
    Initial learning rate
    tpramas: Theano SharedVariable
    Model parameters
    grads: Theano variable
    Gradients of cost w.r.t to parameres
    x: Theano variable
    Model inputs
    mask: Theano variable
    Sequence mask
    y: Theano variable
    Targets
    cost: Theano variable
    Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
    Rate Method*, arXiv:1212.5701.
    """
    zipped_grads = [theano.shared(p.get_value() * np_floatX(0.),
        name='%s_grad' % k)
        for k, p in tparams] #.iteritems()]
    running_up2 = [theano.shared(p.get_value() * np_floatX(0.),
        name='%s_rup2' % k)
        for k, p in tparams] #.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * np_floatX(0.),
        name='%s_rgrad2' % k)
        for k, p in tparams] #.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
            for rg2, g in zip(running_grads2, grads)]

    f_cost = theano.function(input_params,
            cost,
            updates=zgup+rg2up, #[(tg, g) for tg, g in zip(true_grad, grads)],
            on_unused_input='warn',
            name='adadelta_f_cost')
    
    #f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
    #        name='adadelta_f_grad_shared')

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
            for zg, ru2, rg2 in zip(zipped_grads,
                running_up2,
                running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
            for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for (_,p), ud in zip(tparams, updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
            on_unused_input='ignore',
            name='adadelta_f_update')

    return f_cost, f_update


#def rmsprop(lr, tparams, grads, x, mask, y, cost):
def rmsprop(lr, tparams, grads, cost, *input_params): # x_f, x_w, y,
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
    Initial learning rate
    tpramas: Theano SharedVariable
    Model parameters
    grads: Theano variable
    Gradients of cost w.r.t to parameres
    x: Theano variable
    Model inputs
    mask: Theano variable
    Sequence mask
    y: Theano variable
    Targets
    cost: Theano variable
    Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
    lecture 6a,
    http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """
    zipped_grads = [theano.shared(p.get_value() * np_floatX(0.),
        name='%s_grad' % k)
        for k, p in tparams] #.iteritems()]
    running_grads = [theano.shared(p.get_value() * np_floatX(0.),
        name='%s_rgrad' % k)
        for k, p in tparams] #.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * np_floatX(0.),
        name='%s_rgrad2' % k)
        for k, p in tparams] #.iteritems()]
    
    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
            for rg2, g in zip(running_grads2, grads)]

    f_cost = theano.function(input_params,
            cost,
            updates=zgup+rg2up+rgup, #[(tg, g) for tg, g in zip(true_grad, grads)],
            on_unused_input='warn',
            name='rmsprop_f_cost')
    
    #f_grad_shared = theano.function([x, mask, y], cost,
    #        updates=zgup + rgup + rg2up,
    #        name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * np_floatX(0.),
        name='%s_updir' % k)
        for k, p in tparams] #.iteritems()]

    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / T.sqrt(rg2 - rg ** 2 + 1e-4))
            for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                running_grads2)]
    param_up = [(p, p + udn[1])
            for (_, p), udn in zip(tparams, updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
            on_unused_input='ignore',
            name='rmsprop_f_update')

    return f_cost, f_update


def build_optimizer(lr, grads, cost, regularizable_params, prefix, optimizer, *input_params):
    '''
    lr: Learning Rate.
    grads, cost, regularizable_params: gradient, cost and regularizable parameters.
    prefix: build opt for what data
    optimizer: Either the sgd or adadelta function.

    In my case I need to interface between the inside-outside code
    and the theano functions.
    '''
    f_cost, f_update = optimizer(lr, regularizable_params, grads, cost, prefix, *input_params)
    return f_cost, f_update


def create_circuit(_args, StackConfig):
    cargs = dict_from_argparse(_args)
    print type(cargs)
    print_args(cargs)
    cargs = StackConfig(cargs)
    print type(cargs)
    x_f, x_w, y, y_pred, cost, grads, regularizable_params = _args.circuit(cargs)
    if _args.verbose == 2:
        print "\n", "Printing Configuration after StackConfig:"
        print_args(cargs)
    lr = T.scalar('lr')
    # need to refactor: all the build_optimizer and optimizers
    f_cost, f_update = build_optimizer(lr, grads, cost, regularizable_params, 'global', _args.optimizer, 
            x_f, x_w, y)
    f_classify = theano.function(inputs=[x_f, x_w],
            outputs=y_pred,
            on_unused_input='warn',
            name='f_classify')
    return (f_cost, f_update, f_classify, cargs)


def build_obj_grad_and_classifier(_args, cargs, f_arr, x_arr, y_arr, pred_y_arr, cost, grads, global_regularizable_params, costs_arr, grads_arr, regularizable_param_arr):
    if _args.verbose == 2:
        print "\n", "Printing Configuration after StackConfig:"
        print_args(cargs)
    lr = T.scalar('lr')
    input_params = f_arr + x_arr + y_arr
    print 'in create_multi_task_circuit, input size:', len(input_params), type(input_params)
    f_cost, f_update = build_optimizer(lr, grads, cost, global_regularizable_params, 'joint', _args.optimizer, *input_params)
    f_costs_and_updates = []
    f_classifies = []
    for i in range(len(costs_arr)):
        print i
        f_costs_and_updates.append(build_optimizer(lr, grads_arr[i], costs_arr[i], regularizable_param_arr[i], 'task'+str(i), _args.optimizer, f_arr[i], x_arr[i], y_arr[i]))
        f_classifies.append(theano.function(inputs=[f_arr[i], x_arr[i]],
            outputs=pred_y_arr[i],
            on_unused_input='warn',
            name='f_classify_'+str(i)))
    return (f_cost, f_update, f_costs_and_updates, f_classifies, cargs)


def create_joint_circuit(_args, StackConfig):
    cargs = dict_from_argparse(_args)
    print_args(cargs)
    cargs = StackConfig(cargs)
    f_arr, x_arr, y_arr, pred_y_arr, costs_arr, cost, grads_arr, grads, regularizable_param_arr, global_regularizable_params = _args.circuit(cargs, _args.cws_joint_weight)
    print 'in create_joint_circuit, global regularizable_params:', [p.name for p in global_regularizable_params]
    return build_obj_grad_and_classifier(_args, cargs, f_arr, x_arr, y_arr, pred_y_arr, cost, grads, global_regularizable_params, costs_arr, grads_arr, regularizable_param_arr)


def print_args(args):
    for k in args:
        if type(args[k]) == dict or type(args[k]) == list or type(args[k]) == tuple:
            print k, 'container size:', len(args[k])
        else:
            print k, args[k]

def save_parameters(path, params):
    #savable_params={k : v.get_value() for k, v in params.items() }
    pickle.dump(params, open(path, 'w'))

def load_params(path, params):
    pp = pickle.load(open(path, 'r'))
    for kk, vv in pp.iteritems():
        #print "loading parameters:", kk, "type:", type(vv)
    #if kk not in params:
    #    warnings.warn('%s is not in the archive' % kk)
        print 'updating parameter:', kk
        params[kk]=pp[kk]
    return params

