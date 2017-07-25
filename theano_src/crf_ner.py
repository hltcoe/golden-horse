#!/usr/bin/python

import argparse
import random
import sys
import time

import theano, numpy
from neural_lib import StackConfig, ArrayInit
from neural_architectures import plainOrderOneCRF
from sighan_ner import loaddata, get_data, eval_ner, error_analysis, write_predictions
from train_util import dict_from_argparse, shuffle, conv_data, convert_id_to_word, add_arg, add_arg_to_L, read_matrix_from_file, write_matrix_to_file, create_circuit, save_parameters, load_params, batch_run_func 
from train_util import sgd, adadelta, rmsprop  

POS = 'POS'
SEG = 'Segmentation'

def predict(feature_test, lex_test, _args, groundtruth_test):
    ''' On the test set predict the labels using f_classify.
    Compare those labels against groundtruth.

    It returns a dictionary 'results' that contains
    f1 : F1 or Accuracy
    p : Precision
    r : Recall
    '''
    predictions_test= convert_id_to_word(
        [_args.f_classify(feature, sentence) if len(sentence) > 1 else [0] for feature, sentence in zip(feature_test, lex_test)],
        _args.idx2label
    )
    print len(predictions_test), len(groundtruth_test)
    results = None
    if len(groundtruth_test[0]) != 0:
        results = eval_ner(predictions_test, groundtruth_test) #, words_test, folder + '/current.test.txt', folder)
    '''converted_corpus = convert_prediction(words_test, predictions_test, line_pointer)
    for line in converted_corpus:
        print ''.join(line).lstrip()
    '''
    if type(results) is float:
        tmp = results
        results = {}
        results['f1'] = tmp
        results['p'] = tmp
        results['r'] = tmp
    return results, predictions_test


def train_seq(train_lex, train_f, train_y, _args, f_cost, f_update, epoch_id, learning_rate):
    ''' This function is called from the main method. and it is primarily responsible for updating the
    parameters. Because of the way that create_circuit works that creates f_cost, f_update etc. this function
    needs to be flexible and can't be put in a lib.
    Look at lstm_dependency_parsing_simplification.py for more pointers.
    '''
    def train_crf(features, words, labels, learning_rate, f_cost, f_update):
        ' Since function is called only for side effects, it is likely useless anywhere else'
        if labels.shape[0] < 2:
            return 0.0
        iter_cost = f_cost(features, words, labels)
        f_update(learning_rate)
        return iter_cost
    
    def train_lstm(features, words, labels, learning_rate, f_cost, f_update, _args):
        ' Since function is called only for side effects, it is likely useless anywhere else'
        if labels.shape[0] < 2:
            return
        # add a dummy x_f
        iter_cost = f_cost(features, words, labels)
        f_update(learning_rate)
        return iter_cost

    ## main body of train_seq
    if train_f == None:
        shuffle([train_lex, train_y], _args.seed)
    else:
        shuffle([train_lex, train_f, train_y], _args.seed)
    tic = time.time()
    aggregate_cost = 0.0
    for i, (features, words, labels) in enumerate(zip(train_f, train_lex, train_y)):
        if len(words) < 2:
            continue
        assert len(words) == len(labels) #+ 2
        if _args.model == 'lstm': #train_f == None:
            aggregate_cost += train_lstm(features, words, labels, learning_rate, f_cost, f_update, _args)
        elif _args.model == 'crf':
            aggregate_cost += train_crf(features, words, labels, learning_rate, f_cost, f_update)
        else:
            raise NotImplementedError
        if _args.verbose == 2 and i % 10 == 0:
            print '[learning] epoch %i >> %2.2f%%' % (epoch_id, (i + 1) * 100. / _args.nsentences),
            print 'completed in %.2f (sec) <<\r' % (time.time() - tic),
            sys.stdout.flush()
    if _args.verbose == 2:
        print '>> Epoch completed in %.2f (sec) <<' % (time.time() - tic), 'training cost: %.2f' % (aggregate_cost)


class Tester:
    def __init__(self, _args):
        cargs = {}
        print "loading parameters!"
        load_params(_args.save_model_param, cargs)

        self.cargs = cargs

    def run(self, _args):
        cargs = self.cargs

        test_feat, test_lex, test_y = get_data(_args.test_data, cargs['feature2idx'], cargs['word2idx'], cargs['label2idx'], cargs['emb_type'], anno=None, has_label=_args.eval_test)
        test_feat, test_lex, test_y = conv_data((test_feat, test_lex, test_y), cargs['win_l'], cargs['win_r'], cargs['featsize'], cargs['y_dim'])
        _args.idx2label = dict((k, v) for v, k in cargs['label2idx'].iteritems())
        _args.idx2word = dict((k, v) for v, k in cargs['word2idx'].iteritems())
        _args.f_classify = cargs['f_classify']
        if _args.eval_test:
            groundtruth_test = convert_id_to_word(test_y, _args.idx2label)

        else:
            groundtruth_test = [[] for elem in test_y]
        res_test, pred_test = predict(test_feat, test_lex, _args, groundtruth_test)
        return (res_test, pred_test)


def main(_args):
    if _args.only_test:
        tester = Tester(_args)
        res_test, pred_test = tester.run(_args)
        write_predictions(_args.output_dir, _args.test_data, pred_test)
        exit(0)

    print "loading data from:", _args.training_data, _args.valid_data, _args.test_data
    train_set, valid_set, test_set, dic = loaddata(_args.training_data, _args.valid_data, _args.test_data, feature_thresh=_args.ner_feature_thresh, mode=_args.emb_type, test_label=_args.eval_test) #, anno=SEG)  
    _args.label2idx = dic['labels2idx'] 
    _args.word2idx = dic['words2idx']
    _args.feature2idx = dic['features2idx']
    _args.win_l = -(_args.win//2)
    _args.win_r = _args.win//2
    train_set, valid_set, test_set = batch_run_func((train_set, valid_set, test_set), conv_data, _args.win_l, _args.win_r, len(_args.feature2idx), len(_args.label2idx))
    _args.wemb1_win = _args.win
    print _args.label2idx
    nclasses = len(_args.label2idx)
    nsentences = len(train_set[1])
    numpy.random.seed(_args.seed)
    random.seed(_args.seed)
    _args.y_dim = nclasses
    _args.vocsize = len(_args.word2idx) #ufnum #vocsize
    _args.featsize = len(_args.feature2idx) #+ 1  #!!!! Important: maybe should + 1
    _args.feature_size = _args.featsize + 1 #3
    _args.voc_size = _args.vocsize #+ 2
    if _args.circuit == 'plainOrderOneCRF':
        _args.emission_trans_out_dim = nclasses
        _args.emb_output_transform_out_dim = nclasses 
        _args.model = 'crf'
        print 'emission_trans_out_dim:', _args.emission_trans_out_dim
    else:
        raise NotImplementedError
    _args.nsentences = nsentences
    # eval all training and topology related parameters
    for a in TOPO_PARAM + TRAIN_PARAM:
        try:
            _args.__dict__[a] = eval(_args.__dict__[a])
        except:
            pass
    # This way we can inject code from command line.
    if _args.use_emb and _args.emb_init != 'RANDOM':
        M_emb, idx_map = read_matrix_from_file(_args.emb_file, _args.word2idx) 
        emb_var = theano.shared(M_emb, name='emb_matrix')
        _args.emb_matrix = emb_var 
        '''print 'printing ner embedding matrix:'
        for row in emb_var.get_value():
            for num in row:
                print num,
            print ''
        '''
        _args.emb_dim = len(M_emb[0])
        print 'embeding size:', _args.emb_dim
        _args.emb_matrix.is_regularizable = False
        if _args.fine_tuning :
            print 'fine tuning!!!!!'
            _args.emb_matrix.is_regularizable = True
    best_f1 = -numpy.inf
    param = dict(clr = _args.lr, ce = 0, be = 0) # Create Circuit
    (_args.f_cost, _args.f_update, _args.f_classify, #_args.f_debug, 
            cargs) = create_circuit(_args, StackConfig)
    #params_to_save = {k:v for k,v in cargs.items() if (hasattr(v, 'is_regularizable') and v.is_regularizable and k.startswith('tparam'))}
    #print _args
    _args.idx2label = dict((k, v) for v, k in _args.label2idx.iteritems())
    _args.idx2word = dict((k, v) for v, k in _args.word2idx.iteritems())
    groundtruth_valid = convert_id_to_word(valid_set[2], _args.idx2label)
    groundtruth_test = None
    if _args.eval_test:
        groundtruth_test = convert_id_to_word(test_set[2], _args.idx2label)
    epoch_id = -1
    while epoch_id+1 < _args.nepochs:
        epoch_id += 1
        #print 'train_f', train_set[0]
        train_seq(train_set[1], train_set[0], train_set[2], _args, _args.f_cost, _args.f_update, epoch_id, param['clr'])
        # Train and Evaluate
        if epoch_id % _args.neval_epochs == 0:
            groundtruth_train = convert_id_to_word(train_set[2], _args.idx2label)
            #print 'evaluate train!!!'
            res_train, pred_train = predict(train_set[0], train_set[1], _args, groundtruth_train)
            #print 'evaluate valid!!!'
            res_valid, pred_valid = predict(valid_set[0], valid_set[1], _args, groundtruth_valid)
            print('TEST: epoch', epoch_id,
                'train F1'   , res_train['f1'],
                'valid F1'   , res_valid['f1'],
                #'test F1'   , res_test['f1']
                )
            if _args.eval_test:
                res_test, pred_test = predict(test_set[0], test_set[1], _args, groundtruth_test)
                print 'test F1'   , res_test['f1']
            # If this update created a 'new best' model then save it.
            if res_valid['f1'] > best_f1:
                best_f1 = res_valid['f1']
                param['be'] = epoch_id
                param['vf1'] = (res_valid['f1'])  #res_train['f1'], , res_test['f1']
                param['vp'] = (res_valid['p'])  #res_train['p'], , res_test['p']
                param['vr'] = (res_valid['r'])  #res_train['r'], , res_test['r']
                if _args.eval_test:
                    param['tf1'] = (res_test['f1'])
                    param['tp'] = (res_test['p'])
                    param['tr'] = (res_test['r'])
                print "saving parameters!"
                cargs['f_classify'] = _args.f_classify
                save_parameters(_args.save_model_param, cargs)
                #error_analysis(valid_set[1], pred_valid, groundtruth_valid, _args.idx2word)
        else:
                pass
        # decay learning rate if no improvement in 10 epochs
        if _args.decay and (epoch_id - param['be']) >= _args.decay_epochs and (epoch_id - param['be']) % _args.decay_epochs == 0:
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


def create_arg_parser(args=None):
    _arg_parser = argparse.ArgumentParser(description='CRF')
    add_arg.arg_parser = _arg_parser
    ## File IO
    add_arg('--only_test'       , False, help='Only do the test')
    add_arg('--save_model_param'       , 'best-parameters', help='The best model will be saved there')
    add_arg('--training_data'       , 'train-weiboner.crfsuite.txt', help='training file name')
    add_arg('--valid_data'       , 'dev-weiboner.crfsuite.txt', help='develop file name')
    add_arg('--test_data'       , 'test-weiboner.crfsuite.txt', help='test file name')
    add_arg('--output_dir'       , '/export/projects/npeng/weiboNER_data/', help='the output dir that stores the prediction')
    add_arg('--eval_test'       , True, help='Whether evaluate the test data: test data may not have annotations.')
    add_arg('--emb_type'       , 'char', help='The embedding type, choose from (char, word, charpos)')
    add_arg('--emb_file'       , '/export/projects/npeng/weiboNER_data/weibo_char_vectors', help='The initial embedding file name')
    add_arg('--emb_init'       , 'RANDOM', help='The initial embedding type for cws')
    add_arg('--ner_feature_thresh'     , 0    , help="The minimum count (upto and including) OOV threshold for NER")
    ## Training
    add_arg_to_L(TRAIN_PARAM, '--use_features'      , True)
    add_arg_to_L(TRAIN_PARAM, '--lr'           , 0.05)
    add_arg_to_L(TRAIN_PARAM, '--use_emb'           , True)
    add_arg_to_L(TRAIN_PARAM, '--fine_tuning'           , True)
    add_arg_to_L(TRAIN_PARAM, '--nepochs'      , 200)
    add_arg_to_L(TRAIN_PARAM, '--neval_epochs'      , 5)
    add_arg_to_L(TRAIN_PARAM, '--optimizer'    , 'sgd')
    add_arg_to_L(TRAIN_PARAM, '--seed'         , 1) 
    add_arg_to_L(TRAIN_PARAM, '--decay'        , True,  help='whether learning rate decay')
    add_arg_to_L(TRAIN_PARAM, '--decay_epochs' , 10)
    add_arg_to_L(TRAIN_PARAM, '--minimum_lr'   , 1e-5)
    ## Topology
    add_arg_to_L(TOPO_PARAM, '--circuit',                      'plainOrderOneCRF',
            help="the conbination of different models")
    add_arg_to_L(TOPO_PARAM, '--emb_output_transform_out_dim',500)
    add_arg_to_L(TOPO_PARAM, '--wemb1_out_dim',                100)
    add_arg_to_L(TOPO_PARAM, '--in_dim',                       -1)
    add_arg_to_L(TOPO_PARAM, '--emission_trans_out_dim',-1)
    add_arg_to_L(TOPO_PARAM, '--L2Reg_reg_weight',             0.0)
    add_arg_to_L(TOPO_PARAM, '--win',                          1)
    ## DEBUG
    add_arg('--verbose'      , 2)

    return _arg_parser


if __name__ == "__main__":
    TOPO_PARAM = []
    TRAIN_PARAM = []
    _arg_parser = create_arg_parser()
    args = _arg_parser.parse_args()
    #print type(args)
    #print args
    main(args)
