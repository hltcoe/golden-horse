#!/usr/bin/python


import argparse, os, random, subprocess, sys, time
import theano, numpy
import theano.tensor as T
import codecs as cs
from neural_lib_crf import StackConfig, ArrayInit, plainOrderOneCRF 
from train_util import dict_from_argparse, shuffle, conv_data, read_matrix_from_pkl, read_matrix_from_file, write_matrix_to_file, create_circuit, save_parameters, load_params, conv_emb # _conv_feat, _conv_x, _conv_y,
from train_util import sgd 
 

def convert_id_to_word(corpus, idx2label):
    return [[idx2label[word] for word in sentence]
            for sentence
            in corpus]

def add_arg(name, default=None, **kwarg):
    assert hasattr(add_arg, 'arg_parser'), "You must register an arg_parser with add_arg before calling it"
    if 'action' in kwarg:
        add_arg.arg_parser.add_argument(name, default=default, **kwarg)
    else:
        add_arg.arg_parser.add_argument(name, default=default, type=type(default), **kwarg)
    return

def add_arg_to_L(L, name, default=None, **kwarg):
    L.append(name[2:])
    add_arg(name, default, **kwarg)
    return

def predict(features, words, idx2label, idx2word, _args, f_classify, groundtruth=None):
    predictions = convert_id_to_word(
        [f_classify(f, w) for f, w in zip(features, words)],  #_conv_x(sentence, _args.win, _args.vocsize))
        idx2label
    )
    if groundtruth == None:
	    return None, predictions
    results = eval_ner(predictions, groundtruth)  #conlleval(predictions, groundtruth, 
              #          folder + '/current.valid.txt', folder)
    #error_analysis(words, predictions, groundtruth, idx2word)
    return results, predictions

def train(train_feat, train_lex, train_y, _args, f_cost, f_update, f_debug, epoch_id, learning_rate):
    # This function is called from the main method. and it is primarily responsible for updating the
    #parameters. Because of the way that create_circuit works that creates f_cost, f_update etc. this function
    #needs to be flexible and can't be put in a lib.
    #Look at lstm_dependency_parsing_simplification.py for more pointers.
    def train_crf(features, words, labels, learning_rate, f_cost, f_update, f_debug):
        ' Since function is called only for side effects, it is likely useless anywhere else'
        if labels.shape[0] < 2:
            return 0.0
        iter_cost = f_cost(features, words, labels)
        #_, gold_y, pred_y = f_debug(words, labels)
	f_update(learning_rate)
	return iter_cost

    shuffle([train_feat, train_lex, train_y], _args.seed)
    tic = time.time()
    aggregate_cost = 0.0
    for i, (x_f, x_w, y) in enumerate(zip(train_feat, train_lex, train_y)):
        
        try:
            aggregate_cost += train_crf(x_f, x_w, y, learning_rate, f_cost, f_update, f_debug)
        except IndexError:
            import pdb; pdb.set_trace()
        if _args.verbose == 2 and i % 10 == 0:
            print '[learning] epoch %i >> %2.2f%%' % (epoch_id, (i + 1) * 100. / _args.nsentences),
            print 'completed in %.2f (sec) <<\r' % (time.time() - tic),
            sys.stdout.flush()
    if _args.verbose == 2:
	    print '>> Epoch completed in %.2f (sec) <<' % (time.time() - tic), 'training cost: %.2f' % (aggregate_cost)
    #print 'training, current learning rate:', learning_rate
    return

def write_prediction(filename, output_dir, lex_test, pred_test):
    with cs.open(os.path.join(os.path.basename(output_dir), filename), 'w', encoding='utf-8') as outf:
        for sent_w, sent_l in zip(lex_test, pred_test):
            assert len(sent_w) == len(sent_l)
            for w, l in zip(sent_w, sent_l):
                outf.write(w+'\t'+l+'\n')
            outf.write('\n')

def main(_args):
    if _args.only_test:
        cargs = {}
        print "loading parameters!"
        load_params(_args.save_model_param, cargs)
        test_feat, test_lex_orig, test_y = get_data(_args.test_data, cargs['feature2idx'], cargs['word2idx'], cargs['label2idx'], cargs['emb_type'], anno=None, has_label=_args.eval_test)
        test_feat, test_lex, test_y = conv_data(test_feat, test_lex_orig, test_y, cargs['win'], cargs['vocsize'])
        idx2label = dict((k, v) for v, k in cargs['label2idx'].iteritems())
        idx2word = dict((k, v) for v, k in cargs['word2idx'].iteritems())
        groundtruth_test = None
        if _args.eval_test:
            groundtruth_test = convert_id_to_word(test_y, idx2label)
        original_text = convert_id_to_word(test_lex_orig, idx2word)
        f_classify = cargs['f_classify']
        res_test, pred_test = predict(test_feat, test_lex, idx2label, idx2word, _args, f_classify, groundtruth_test)
        write_prediction(_args.test_data+'.prediction', _args.output_dir, original_text, pred_test)
        exit(0)
    
    print "loading data from:", _args.training_data, _args.valid_data, _args.test_data
    train_set, valid_set, test_set, dicts = loaddata(_args.training_data, _args.valid_data, _args.test_data, feature_thresh=_args.ner_feature_thresh, mode=_args.emb_type, test_label=_args.eval_test) 
    train_feat, train_lex_orig, train_y = train_set 
    valid_feat, valid_lex_orig, valid_y = valid_set 
    test_feat, test_lex_orig, test_y = test_set 
    feature2idx = dicts['features2idx'] 
    word2idx = dicts['words2idx'] 
    label2idx = dicts['labels2idx']
    #idx2feature = dict((k, v) for v, k in feature2idx.iteritems())
    _args.label2idx = label2idx
    _args.word2idx = word2idx
    _args.feature2idx = feature2idx
    nclasses = len(label2idx)
    nsentences = len(train_lex_orig)
    numpy.random.seed(_args.seed)
    random.seed(_args.seed)
    _args.y_dim = nclasses
    _args.vocsize = len(feature2idx) #ufnum #vocsize
    _args.in_dim = _args.vocsize #+ 2
    if _args.circuit == 'plainOrderOneCRF':
        _args.emission_trans_out_dim = nclasses
    _args.nsentences = nsentences
    # eval all training and topology related parameters
    for a in TOPO_PARAM + TRAIN_PARAM:
        try:
	    _args.__dict__[a] = eval(_args.__dict__[a])
        except:
            pass
    # This way we can inject code from command line.
    if _args.use_emb == 'true':
    	M_emb, idx_map = read_matrix_from_file(_args.emb_file, word2idx) 
    	emb_var = theano.shared(M_emb, name='emb_matrix')
    	_args.emb_matrix = emb_var 
        _args.emb_dim = len(M_emb[0])
        print 'embeding size:', _args.emb_dim
    	if _args.fine_tuning == 'true':
        	print 'fine tuning!!!!!'
        	_args.emb_matrix.is_regularizable = True
    train_feat, train_lex, train_y = conv_data(train_feat, train_lex_orig, train_y, _args.win, _args.vocsize)
    valid_feat, valid_lex, valid_y = conv_data(valid_feat, valid_lex_orig, valid_y, _args.win, _args.vocsize)
    test_feat, test_lex, test_y = conv_data(test_feat, test_lex_orig, test_y, _args.win, _args.vocsize)
    best_f1 = -numpy.inf
    param = dict(clr = _args.lr, ce = 0, be = 0) # Create Circuit
    (f_cost, f_update, f_classify, f_debug, cargs) = create_circuit(_args, StackConfig)
    #params_to_save = {k:v for k,v in cargs.items() if (hasattr(v, 'is_regularizable') and v.is_regularizable and k.startswith('tparam'))}
    #print _args
    idx2label = dict((k, v) for v, k in _args.label2idx.iteritems())
    idx2word = dict((k, v) for v, k in _args.word2idx.iteritems())
    groundtruth_valid = convert_id_to_word(valid_y, idx2label)
    groundtruth_test = None
    if _args.eval_test:
    	groundtruth_test = convert_id_to_word(test_y, idx2label)
    epoch_id = -1
    while epoch_id+1 < _args.nepochs:
        epoch_id += 1
	train(train_feat, train_lex, train_y, _args, f_cost, f_update, f_debug, epoch_id, param['clr'])
        # Train and Evaluate
	if epoch_id % _args.neval_epochs == 0:
            groundtruth_train = convert_id_to_word(train_y, idx2label)
	    #print 'evaluate train!!!'
	    res_train, pred_train = predict(train_feat, train_lex, idx2label, idx2word, _args, f_classify, groundtruth_train)
	    #print 'evaluate valid!!!'
	    res_valid, pred_valid = predict(valid_feat, valid_lex, idx2label, idx2word, _args, f_classify, groundtruth_valid)
	    res_test, pred_test = predict(test_feat, test_lex, idx2label, idx2word, _args, f_classify, groundtruth_test)
	    print('TEST: epoch', epoch_id,
	          'train F1'   , res_train['f1'],
                  'valid F1'   , res_valid['f1'],
                  #'test F1'   , res_test['f1']
		  )
	    if _args.eval_test:
		print 'test F1'   , res_test['f1']
            # If this update created a 'new best' model then save it.
            if res_valid['f1'] > best_f1:
                best_f1 = res_valid['f1']
                param['be'] = epoch_id
                param['last_decay'] = epoch_id
                param['vf1'] = (res_valid['f1'])  #res_train['f1'], , res_test['f1']
                param['vp'] = (res_valid['p'])  #res_train['p'], , res_test['p']
                param['vr'] = (res_valid['r'])  #res_train['r'], , res_test['r']
		if _args.eval_test:
		    param['tf1'] = (res_test['f1'])
           	    param['tp'] = (res_test['p'])
           	    param['tr'] = (res_test['r'])
		print "saving parameters!"
		cargs['f_classify'] = f_classify
		save_parameters(_args.save_model_param, cargs)
	        '''
		print "loading parameters!"
		load_params(_args.save_model_param, cargs)
	        res_test, pred_test = predict(test_feat, test_lex, idx2label, idx2word, _args, f_classify, groundtruth_test)
	        print 'test F1:', res_test['f1']
		'''
	    else:
                pass
        # decay learning rate if no improvement in 10 epochs
        if _args.decay and (epoch_id - param['last_decay']) >= _args.decay_epochs: #and (epoch_id - param['be']) % _args.decay_epochs == 0:
            print 'learning rate decay at epoch', epoch_id
            param['last_decay'] = epoch_id
            param['clr'] *= 0.5
        # If learning rate goes down to minimum then break.
        if param['clr'] < _args.minimum_lr:
            print "\nLearning rate became too small, breaking out of training"
            break

    print('BEST RESULT: epoch', param['be'],
          'valid F1', param['vf1'], param['vp'], param['vr'],
          #'best test F1', param['tf1'], param['tp'], param['tr'] 
          )
    if _args.eval_test:
	print 'best test F1', param['tf1'], param['tp'], param['tr']


if __name__ == "__main__":
    #######################################################################################
    ## PARSE ARGUMENTS, BUILD CIRCUIT, TRAIN, TEST
    #######################################################################################
    _arg_parser = argparse.ArgumentParser(description='CRF')
    add_arg.arg_parser = _arg_parser
    TOPO_PARAM = []
    TRAIN_PARAM = []
    ## File IO
    add_arg('--only_test'       , False, help='Only do the test')
    add_arg('--save_model_param'       , 'best-parameters', help='The best model will be saved there')
    add_arg('--training_data'       , 'train-weiboner.crfsuite.txt', help='training file name')
    add_arg('--valid_data'       , 'dev-weiboner.crfsuite.txt', help='develop file name')
    add_arg('--test_data'       , 'test-weiboner.crfsuite.txt', help='test file name')
    add_arg('--output_dir'       , '/export/projects/npeng/weiboNER_data/', help='the output dir that stores the prediction')
    add_arg('--eval_test'       , False, help='Whether evaluate the test data: test data may not have annotations.')
    add_arg('--emb_type'       , 'char', help='The embedding type, choose from (char, word, charpos)')
    add_arg('--emb_file'       , '/export/projects/npeng/weiboNER_data/weibo_char_vectors', help='The initial embedding file name')
    add_arg('--ner_feature_thresh'     , 0    , help="The minimum count (upto and including) OOV threshold for NER")
    ## Training
    add_arg_to_L(TRAIN_PARAM, '--lr'           , 0.05)
    add_arg_to_L(TRAIN_PARAM, '--use_emb'           , 'true')
    add_arg_to_L(TRAIN_PARAM, '--fine_tuning'           , 'true')
    add_arg_to_L(TRAIN_PARAM, '--nepochs'      , 200)
    add_arg_to_L(TRAIN_PARAM, '--neval_epochs'      , 5)
    add_arg_to_L(TRAIN_PARAM, '--optimizer'    , 'sgd')
    add_arg_to_L(TRAIN_PARAM, '--seed'         , 1) 
    add_arg_to_L(TRAIN_PARAM, '--decay'        , True,  action='store_true')
    add_arg_to_L(TRAIN_PARAM, '--decay_epochs' , 10)
    add_arg_to_L(TRAIN_PARAM, '--minimum_lr'   , 1e-5)
    ## Topology
    add_arg_to_L(TOPO_PARAM, '--circuit',                      'plainOrderOneCRF',
                 help="the conbination of different models")
    add_arg_to_L(TOPO_PARAM, '--in_dim',                       -1)
    add_arg_to_L(TOPO_PARAM, '--emission_trans_out_dim',-1)
    add_arg_to_L(TOPO_PARAM, '--L2Reg_reg_weight',             0.0)
    add_arg_to_L(TOPO_PARAM, '--win',                          1)
    ## DEBUG
    add_arg('--verbose'      , 2)
    args = _arg_parser.parse_args()
    import functools
    from sighan_ner import loaddata, get_data, eval_ner, error_analysis #, conlleval
    main(args)
