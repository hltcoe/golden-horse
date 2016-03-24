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
		elems = line.rstrip().split()
		if elems[0] in dic:
			idx = dic[elems[0]]
			vec_elem = elems[1].split(',')
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
	#print emb_dim, window_size, len(win[0])
	#print win
	#assert len(win[0]) == emb_dim*window_size
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



#------------------------Belows are for theano implementation of neural lanugage model, did not carry it through.---------------------

def get_datetime():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def get_elapsed_time_min(start_seconds, end_seconds):
    return (end_seconds - start_seconds) / 60

def get_train_sample_num(i, batch_size, num_train_batches, num_train_samples):
    ngram_start_id = i * batch_size
    ngram_end_id = (i + 1) * batch_size - 1 if i < (num_train_batches - 1) else num_train_samples - 1
    return ngram_end_id - ngram_start_id + 1

def open_train_files(train_file, lang, vocab_size):
    train_file = train_file + '.' + \
        str(vocab_size) + '.id.' + lang
    train_f = codecs.open(train_file, 'r', 'utf-8')
    return train_f

def print_cml_args(args):
    argsDict = vars(args)
    print "------------- PROGRAM PARAMETERS -------------"
    for a in argsDict:
        print "%s: %s" % (a, str(argsDict[a]))
    print "------------- ------------------ -------------"
    
class TrainModel(object):
    def __init__(self):
        self.vocab_loaded = False
        self.model_param_loaded = False
        self.train_param_loaded = False
        self.valid_set_loaded = False
        self.test_set_loaded = False

    def loadVocab(self, lang, vocab_size, vocab_file):
        self.lang = lang
        self.vocab_size = vocab_size
        self.vocab_file = vocab_file
        self.vocab_loaded = True

    def loadModelParams(self):
        """
        Subclass should override this function.
        """
        raise NotImplementedError("Override this function.")

    def loadTrainParams(self, train_file, batch_size, default_learning_rate, opt, valid_freq, finetune_epoch, 
        chunk_size=10000, vocab_size=1000, n_epochs=5):
        self.train_file = train_file
        self.batch_size = batch_size
        self.default_learning_rate = default_learning_rate
        self.opt = opt
        self.valid_freq = valid_freq
        self.finetune_epoch = finetune_epoch
        self.chunk_size = chunk_size
        self.vocab_size = vocab_size
        self.n_epochs = n_epochs
        self.train_param_loaded = True

    def loadValidSet(self, valid_data_package):
        """
        Subclass should override this function.
        """
        raise NotImplementedError("Override this function.")

    def loadTestSet(self, test_data_package):
        """
        Subclass should override this function.
        """
        raise NotImplementedError("Override this function.")

    def validate(self, model, num_ngrams, batch_size, num_batches):
        """
        Subclass should override this function.
        """
        raise NotImplementedError("Override this function.")

    def test(self, model, num_ngrams, batch_size, num_batches):
        """
        Subclass should override this function.
        """
        raise NotImplementedError("Override this function.")

    # Return False if there is no more data
    def loadBatchData(self, isInitialLoad=False):
        """
        Subclass should override this function.
        """
        raise NotImplementedError("Override this function.")

    def buildModels(self):
        """
        Subclass should override this function.
        """
        raise NotImplementedError("Override this function.")

    # Open training files, called once in train function
    def openFiles(self):
        self.train_f = open_train_files(self.train_file, self.lang, self.vocab_size)

    # Close training files, called once in train function
    def closeFiles(self):
        self.train_f.close()

    def trainOnBatch(self, train_model, i, batch_size, num_train_batches, num_train_samples, learning_rate):
        """
        Subclass should override this function.
        """
        raise NotImplementedError("Override this function.")

    def displayFirstNExamples(self, n):
        """
        Subclass should override this function.
        """
        raise NotImplementedError("Override this function.")        

    def train(self):
        if not (self.vocab_loaded and self.model_param_loaded and self.train_param_loaded and self.valid_set_loaded):
            sys.stderr.write("TrainModel not initialized with enough parameters!\n")
            sys.exit(-1)

        # SGD HYPERPARAMETERS
        batch_size = self.batch_size
        learning_rate = self.default_learning_rate
        valid_freq = self.valid_freq
        finetune_epoch = self.finetune_epoch
        n_epochs = self.n_epochs
        model = self.model
        
        # Open files and load initial batch of data
        print "Getting initial data ..." 
        self.openFiles()
        self.loadBatchData(True)
        print "Display example input data ..."
        self.displayFirstNExamples(10)
        print ""
        
        ##### MODELING BEG #############################################
        train_model, valid_model, test_model = self.buildModels()
        ##### MODELING END #############################################

        ##### TRAINING BEG #############################################
        iter = 0
        epoch = 1
        start_iter = 0
        train_batches_epoch = 0
        train_costs = []
        best_valid_perp = float('inf')
        bestvalid_test_perp = 0
        prev_valid_loss = 0 # for learning rate halving
        epoch_valid_perps = [] # for convergence check
        seq_epoch_num = 5 # think model converges if perp not change in seq_epoch_num epochs
        # finetuning
        # the fraction of an epoch that we halve our learning rate
        finetune_fraction = 1
        assert finetune_epoch >= 1
        # printing info bookkeeping
        train_sample_per_epoch = 0
        train_sample_cnt = 0
        start_time_seconds = time.time()
        ##########################
        # epoch:
        # num_train_batch:
        ##########################
        while (epoch <= n_epochs):
            print "-------- EPOCH %d BEGINS ----------" % (epoch)
            while(True):
                num_train_samples = model.getTrainSetXSize()
                num_train_batches = (num_train_samples - 1) / batch_size + 1
		print 'number of train samples: %d, number of train batches: %d' % (num_train_samples,  num_train_batches)
		if epoch == 1:
                    train_batches_epoch += num_train_batches
                    train_sample_per_epoch += num_train_samples

                # train
                for i in xrange(num_train_batches):
                    outputs = self.trainOnBatch(train_model, i, batch_size, num_train_batches, num_train_samples, learning_rate)
                    train_sample_cnt += get_train_sample_num(i, batch_size, num_train_batches, num_train_samples)
                    train_costs.append(outputs[0])

                    # # check for nan/inf and print out debug infos
                    if np.isnan(outputs[0]) or np.isinf(outputs[0]):
                        sys.stderr.write('---> Epoch %d, iter %d: nan or inf, bad ... Stop training and exit\n' % (epoch, iter))
                        sys.exit(1)

                    iter += 1

                    # finetuning
                    if iter % (train_batches_epoch * finetune_fraction) == 0:
                        (valid_loss, valid_perp) = self.validate(valid_model, model.num_valid_ngrams, batch_size, model.num_valid_batches)
                        # if likelihood loss is worse than the previous epoch, the learning rate is multiplied by 0.5.
                        if epoch > finetune_epoch and valid_loss > prev_valid_loss:
                            learning_rate /= 2
                            print '---> Epoch %d, iter %d, halving learning rate to: %f' % (epoch, iter, learning_rate)
                        prev_valid_loss = valid_loss

                    if iter % valid_freq == 0:
                        train_loss = np.mean(train_costs)
                        (valid_loss, valid_perp) = self.validate(valid_model, model.num_valid_ngrams, batch_size, model.num_valid_batches)
                        (test_loss, test_perp) = self.test(test_model, model.num_test_ngrams, batch_size, model.num_test_batches)
                        elapsed_time_min = get_elapsed_time_min(start_time_seconds, time.time())
                        print 'iter: %d \t train_loss = %.4f \t valid_loss = %.4f \t test_loss = %.4f \t valid_perp = %.2f \t test_perp = %.2f \t time = %d min \t #samples: %d \t speed: %.0f samples/sec' % (iter, train_loss, valid_loss, test_loss, \
                            valid_perp, test_perp, elapsed_time_min, train_sample_cnt, train_sample_cnt/elapsed_time_min/60)
                        if valid_perp < best_valid_perp:
                            best_valid_perp = valid_perp
                            bestvalid_test_perp = test_perp

                # read more data
                if self.loadBatchData() ==  False:
                    break

            # end an epoch
            print "-------- EPOCH %d FINISHES --------" % (epoch)
            if iter > 1000:
                print "-------- (best valid perp = %.2f \t test_perp = %.2f) -----" % (best_valid_perp, bestvalid_test_perp)
                # Convergence check
                epoch_valid_perps.append(best_valid_perp)
                l = len(epoch_valid_perps)
                if l >= seq_epoch_num:
                    converged = True
                    for ii in range(seq_epoch_num):
                        if epoch_valid_perps[l-ii-1] != best_valid_perp:
                            converged = False
                else:
                    converged = False
                if converged:
                    print "-------- Converged --------"
                    break
            else:
                print "-------- Cannot give best valid perp (have not reached 1000 iterations) -----"
            if epoch == 1:
                print "-------- (%d samples/epoch) --------" % (train_sample_per_epoch)
            epoch = epoch + 1

            self.closeFiles()
            self.openFiles()
            self.loadBatchData()
        ##### TRAINING END #############################################
        print "--------"
        print "-------- Final Result: best valid perp = %.2f \t test perp = %.2f" % (best_valid_perp, bestvalid_test_perp)
        print "--------"
        self.closeFiles()

