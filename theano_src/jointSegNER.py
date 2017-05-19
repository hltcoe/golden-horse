#!/usr/bin/python


import argparse, os, random, subprocess, sys, time
import theano
import theano.tensor as T
from copy import copy
from icwb import convert_prediction
from neural_lib import StackConfig, ArrayInit
from neural_architectures import plainOrderOneCRF, LSTMOrderOneCRF, LSTMwithFeatureOrderOneCRF, jointSegmentationNER, SerializableLambda
from train_util import dict_from_argparse, shuffle, conv_data, conv_x, _conv_y, _conv_feat, create_joint_circuit, save_parameters, convert_id_to_word, add_arg, add_arg_to_L
from train_util import sgd, adadelta, rmsprop, read_matrix_from_file, read_matrix_and_idmap_from_file
from sighan_ner import create_feature_dict, create_lex_dict, eval_ner, loaddata 
import numpy

SEG = 'Segmentation'

def predict(feature_test, lex_test, f_classify, idx2label, groundtruth_test):
    ''' On the test set predict the labels using f_classify.
    Compare those labels against groundtruth.

    It returns a dictionary 'results' that contains
    f1 : F1 or Accuracy
    p : Precision
    r : Recall
    '''
    predictions_test= convert_id_to_word(
        [f_classify(feature, sentence) if len(sentence) > 1 else [0] for feature, sentence in zip(feature_test, lex_test)],
        idx2label
    )
        #[f_classify(_conv_feat(feature, _args.featsize), conv_x(sentence, _args.win_l, _args.win_r, _args.vocsize)) if len(sentence) > 1 else [0] for feature, sentence in zip(feature_test, lex_test)],
    print len(predictions_test), len(groundtruth_test)
    results = None
    if len(groundtruth_test[0]) != 0:
        results = eval_ner(predictions_test, groundtruth_test) #, words_test, folder + '/current.test.txt', folder)
    if type(results) is float:
        tmp = results
        results = {}
        results['f1'] = tmp
        results['p'] = tmp
        results['r'] = tmp
    return results, predictions_test


def train_joint(_args, f_cost, f_update, epoch_id, learning_rate, num_tasks, nsentences, words_arr, feat_arr, label_arr):
    ''' This function is called from the main method. and it is primarily responsible for updating the parameters.'''
    def train_one_instance(learning_rate, f_cost, f_update, *inputs):
        ' Since function is called only for side effects, it is likely useless anywhere else'
        iter_cost = f_cost(*inputs)
        f_update(learning_rate)
        return iter_cost

    #shuffle([tl1, tf1, ty1], _args.seed)
    for i in range(num_tasks):
        shuffle([words_arr[i], feat_arr[i], label_arr[i]], _args.seed)
    tic = time.time()
    aggregate_cost = 0.0
    input_params = feat_arr + words_arr + label_arr
    for i, one_input in enumerate(zip(*input_params)):
        aggregate_cost += train_one_instance(learning_rate, f_cost, f_update, *one_input)
        if _args.verbose == 2 and i % 10 == 0:
            print '[learning] epoch %i >> %2.2f%%' % (epoch_id, (i + 1) * 100. / nsentences),
            print 'completed in %.2f (sec) <<\r' % (time.time() - tic),
            sys.stdout.flush()
    if _args.verbose == 2:
        print '>> Epoch completed in %.2f (sec) <<' % (time.time() - tic), 'training cost: %.2f' % (aggregate_cost)


def train_alternative(_args, f_costs_and_updates, epoch_id, learning_rate_arr, nsentences_arr, words_arr, feat_arr, label_arr):
    num_tasks = len(f_costs_and_updates)
    print 'num_tasks:', num_tasks
    for i in range(num_tasks):
        f_cost, f_update = f_costs_and_updates[i]
        print 'num_sentences:', nsentences_arr[i]
        train_single(words_arr[i], feat_arr[i], label_arr[i], _args, f_cost, f_update, epoch_id, learning_rate_arr[i], nsentences_arr[i])


def train_single(tw, tf, ty, _args, f_cost, f_update, epoch_id, learning_rate, nsentences):
    def train_one_instance(f, w, l, learning_rate, f_cost, f_update):
        ' Since function is called only for side effects, it is likely useless anywhere else'
        iter_cost = f_cost(f, w, l)
        f_update(learning_rate)
        return iter_cost

    shuffle([tw, tf, ty], _args.seed)
    if nsentences != len(tw):
        tw = tw[:nsentences]
        tf = tf[:nsentences]
        ty = ty[:nsentences]
    tic = time.time()
    aggregate_cost = 0.0
    for i, (f, x, y) in enumerate(zip(tf, tw, ty)):
        assert len(x) >= 2
        assert len(x) == len(y) #+ 2
        aggregate_cost += train_one_instance(f, x, y, learning_rate, f_cost, f_update)
        if _args.verbose == 2 and i % 10 == 0:
            print '[learning] epoch %i >> %2.2f%%' % (epoch_id, (i + 1) * 100. / nsentences),
            print 'completed in %.2f (sec) <<\r' % (time.time() - tic),
            sys.stdout.flush()
    if _args.verbose == 2:
        print '>> Epoch completed in %.2f (sec) <<' % (time.time() - tic), 'training cost: %.2f' % (aggregate_cost)


def prepare_params(_args):
    numpy.random.seed(_args.seed)
    random.seed(_args.seed)
    # Do not load data if we want to debug topology
    _args.voc_size = len(_args.global_word_map)
    _args.feature_size_1 = len(_args.cws_dicts['features2idx'])
    _args.feature_size_2 = len(_args.ner_dicts['features2idx'])
    _args.m1_hidden_output_trans_out_dim = len(_args.cws_idx2label)
    _args.m2_hidden_output_trans_out_dim = len(_args.ner_idx2label)
    _args.m1_feature_emission_trans_out_dim = len(_args.cws_idx2label)
    _args.m2_feature_emission_trans_out_dim = len(_args.ner_idx2label)
    _args.nsentences = len(_args.ner_train_set[1]) * _args.sample_coef
    _args.nsentences1 = _args.nsentences #len(_args.cws_train_set[1])
    _args.nsentences2 = len(_args.ner_train_set[1])
    _args.m1_lstm_go_backwards = False
    _args.wemb1_win = _args.cws_win_r - _args.cws_win_l + 1
    _args.m1_wemb1_win = _args.ner_win_r - _args.ner_win_l + 1
    _args.circuit = 'jointSegmentationNER'
    eval_args(_args)
    print 'Chkpt1: cws label size:', len(_args.cws_idx2label), 'ner label size:', len(_args.ner_idx2label), 'cws training data size:', len(_args.cws_train_set[0]), 'ner training data size:', len(_args.ner_train_set[0]), _args.cws_train_set[1][0], 'vocabulary size:', _args.voc_size, _args.cws_train_set[-1][0], _args.ner_train_set[1][0], _args.ner_train_set[-1][0]

def eval_args(_args):
    for a in _args.__dict__:  #TOPO_PARAM + TRAIN_PARAM:
        try:
            _args.__dict__[a] = eval(_args.__dict__[a])
        except:
            pass
      
def run_training(_args, param, mode='joint'):
    print 'in run training!! Mode:', mode
    if mode == 'joint' or mode == 'alternative' or mode == 'cws':
        train_f1, train_lex1, train_y1 = _args.cws_train_set
    if mode == 'joint' or mode == 'alternative' or mode == 'ner':
        train_f2, train_lex2, train_y2 = _args.ner_train_set
    if mode == 'joint' or mode == 'alternative':
        words_arr = (train_lex1, train_lex2)
        feat_arr = (train_f1, train_f2)
        label_arr = (train_y1, train_y2)
        if mode == 'joint':
            train_joint(_args, _args.f_cost, _args.f_update, param['epoch_id'], param['clr'], 2, _args.nsentences, words_arr, feat_arr, label_arr)
        elif mode == 'alternative':
            train_alternative(_args, _args.f_costs_and_updates, param['epoch_id'], (param['clr1'], param['clr2']), (_args.nsentences1, _args.nsentences2), words_arr, feat_arr, label_arr)
    elif mode == 'cws':
        f_cost, f_update = _args.f_costs_and_updates[0]
        train_single(train_lex1, train_f1, train_y1, _args, f_cost, f_update, param['epoch_id'], param['clr1'], _args.nsentences2*10)
    elif mode == 'ner':
        f_cost, f_update = _args.f_costs_and_updates[1]
        train_single(train_lex2, train_f2, train_y2, _args, f_cost, f_update, param['epoch_id'], param['clr2'], _args.nsentences2)
    else:
        raise NotImplementedError 

def run_model(_args, mode='joint'):
    print 'start running model!! Mode =', mode
    cws_best_f1 = -numpy.inf
    ner_best_f1 = -numpy.inf
    param = dict(clr = _args.lr, clr1 = _args.cws_lr, clr2 = _args.ner_lr, be1 = 0, be2 = 0, epoch_id = -1)
    if mode == 'ner':
        for i in range(_args.nepochs):
            run_training(_args, param, 'cws')
    while param['epoch_id']+1 < _args.nepochs:
        param['epoch_id'] += 1
        run_training(_args, param, mode)
        if mode == 'joint' or mode == 'alternative' or mode == 'cws':
            train_f1, train_lex1, train_y1 = _args.cws_train_set
            valid_f1, valid_lex1, valid_y1 = _args.cws_valid_set
            test_f1, test_lex1, test_y1 = _args.cws_test_set
            groundtruth_valid1 = convert_id_to_word(valid_y1, _args.cws_idx2label) 
            groundtruth_train1 = convert_id_to_word(train_y1, _args.cws_idx2label)
            groundtruth_test1 = convert_id_to_word(test_y1, _args.cws_idx2label)
        if mode == 'joint' or mode == 'alternative' or mode == 'ner':
            train_f2, train_lex2, train_y2 = _args.ner_train_set
            valid_f2, valid_lex2, valid_y2 = _args.ner_valid_set
            test_f2, test_lex2, test_y2 = _args.ner_test_set
            groundtruth_valid2 = convert_id_to_word(valid_y2, _args.ner_idx2label) 
            groundtruth_train2 = convert_id_to_word(train_y2, _args.ner_idx2label)
            groundtruth_test2 = convert_id_to_word(test_y2, _args.ner_idx2label)
        if mode == 'joint' or mode == 'alternative' or mode == 'cws':
            res_train1, _ = predict(train_f1, train_lex1, _args.f_classifies[0], _args.cws_idx2label,  groundtruth_train1)
            res_valid1, _ = predict(valid_f1, valid_lex1, _args.f_classifies[0], _args.cws_idx2label, groundtruth_valid1)
            if res_valid1['f1'] > cws_best_f1: 
                cws_best_f1 = res_valid1['f1']
                param['be1']  = param['epoch_id']
                param['last_decay1'] = param['be1']
                param['vf11'] = res_valid1['f1']
                param['vp1']  = res_valid1['p']
                param['vr1']  = res_valid1['r']
        if mode == 'joint' or mode == 'alternative' or mode == 'ner':
            res_train2, _ = predict(train_f2, train_lex2, _args.f_classifies[1], _args.ner_idx2label,  groundtruth_train2)
            res_valid2, _ = predict(valid_f2, valid_lex2, _args.f_classifies[1], _args.ner_idx2label, groundtruth_valid2)
            # If this update created a 'new best' model then save it.
            if res_valid2['f1'] > ner_best_f1:
                ner_best_f1 = res_valid2['f1']
                param['be2']  = param['epoch_id']
                param['last_decay2'] = param['be2']
                param['vf12'] = res_valid2['f1']
                param['vp2']  = res_valid2['p']
                param['vr2']  = res_valid2['r']
                param['ner_best_classifier'] = _args.f_classifies[1]
        if _args.verbose:
            print('TEST: epoch', param['epoch_id'],)
            if mode == 'joint' or mode == 'alternative' or mode == 'cws':
                print('SEGMENTATION train F1'   , res_train1['f1'],
                  'valid F1'   , res_valid1['f1'],)
            if mode == 'joint' or mode == 'alternative' or mode == 'ner':
                print('NER train F1'   , res_train2['f1'],
                  'valid F1'   , res_valid2['f1'])
            res_test2, predictions_test2 = predict(test_f2, test_lex2, _args.f_classifies[1], _args.ner_idx2label, groundtruth_test2) 
        if (mode == 'joint' or mode == 'alternative' or mode == 'ner') and (param['be2'] == param['epoch_id']) and _args.ner_eval_test:
            # get the prediction, convert and write to concrete.
            param['tf12'] = res_test2['f1']
            param['tp2']  = res_test2['p']
            param['tr2']  = res_test2['r']
            print '\nEpoch:%d'%param['epoch_id'], 'Test metric', ' '.join([(k + '=' + str(param[k])) for k in ['tf12', 'tp2', 'tr2']]), '\n'
        
        if (mode == 'joint' or mode == 'alternative' or mode == 'ner') and _args.decay and (param['epoch_id'] - param['last_decay2']) >= args.ner_decay_epochs:
            print 'NER learning rate decay at epoch', param['epoch_id'], '! Previous best epoch number:', param['be2']
            param['last_decay2'] = param['epoch_id']
            param['clr2'] *= 0.5
            param['clr'] *= 0.5
        # If learning rate goes down to minimum then break.
            if param['clr2'] < args.minimum_lr or param['clr'] < args.minimum_lr:
                print "\nNER Learning rate became too small, breaking out of training"
                break
        if (mode == 'joint' or mode == 'alternative' or mode == 'cws') and _args.decay and (param['epoch_id'] - param['last_decay1']) >= args.cws_decay_epochs:
            print 'SEGMENTATION learning rate decay at epoch', param['epoch_id'], '! Previous best epoch number:', param['be1']
            param['last_decay1'] = param['epoch_id']
            param['clr1'] *= 0.5
            param['clr'] *= 0.5
        # If learning rate goes down to minimum then break.
            if param['clr1'] < args.minimum_lr or param['clr'] < args.minimum_lr:
                print "\nNER Learning rate became too small, breaking out of training"
                break
    print('NER BEST RESULT: epoch', param['be2'],
          'valid F1', param['vf12'],
          'best test F1', param['tf12'],
          #'SEGMENTATION BEST RESULT: epoch', param['be1'],
          #'best valid F1', param['vf11'],
          )


def combine_word_dicts(dict1, dict2):
    print 'the size of the two dictionaries are:', len(dict1), len(dict2)
    combine_dict = dict1.copy()
    for k, v in dict2.items():
        if k not in combine_dict:
            combine_dict[k] = len(combine_dict)
    print 'the size of the combined dictionary is:', len(combine_dict)
    return combine_dict 

### map the idx of the second dict to the first dict ###
def get_index_map(dict1, dict2):
    print 'in get index map!'
    assert len(dict1) >= len(dict2)
    idx_map = [0 for i in range(len(dict2)+2)]
    for k, v in dict2.items():
        try:
            assert k in dict1
        except:
            print 'key', k, 'not in the global dict!!!!!!!!!!!!!!'
        idx_map[v] = dict1.get(k, 0)
        #print k, v, idx_map[v]
    ### Speical design for the BOS and EOS symbol ###
    idx_map[-1] = len(dict1) - 1
    idx_map[-2] = len(dict1) - 2
    return idx_map

### convert the old word idx to the new one ###
def convert_word_idx(corpus_word, idx2word_old, word2idx_new):
    print 'in convert word index!'
    assert len(idx2word_old) <= len(word2idx_new)
    new_corpus = [[word2idx_new[idx2word_old[idx]] for idx in line] for line in corpus_word]
    return new_corpus

def batch_run_func(corpora, func, *parameters ):
    converted_corpora = []
    for corpus in corpora:
        converted_corpora.append(func(corpus, *parameters))
    return converted_corpora

def main(_args):
    _args.rng = numpy.random.RandomState(_args.seed)
    from icwb import load_data 
    cws_loaddata = load_data
    ner_loaddata = loaddata #load_data_concrete
    ### load cws and ner data ### 
    _args.cws_train_set, _args.cws_valid_set, _args.cws_test_set, _args.cws_dicts = cws_loaddata(_args.cws_train_path, _args.cws_valid_path, _args.cws_test_path)
    ## map the dictionaries
    _args.cws_idx2word = dict((k, v) for v, k in _args.cws_dicts['words2idx'].iteritems())
    _args.cws_idx2label = dict((k, v) for v, k in _args.cws_dicts['labels2idx'].iteritems())
    ## load ner data
    _args.ner_train_set, _args.ner_valid_set, _args.ner_test_set, _args.ner_dicts = ner_loaddata(_args.ner_train_path, _args.ner_valid_path, _args.ner_test_path, feature_thresh=_args.ner_feature_thresh)  
    # map the dictionaries
    _args.ner_idx2word = dict((k, v) for v, k in _args.ner_dicts['words2idx'].iteritems())
    _args.ner_idx2label = dict((k, v) for v, k in _args.ner_dicts['labels2idx'].iteritems())
    ### Combine two word dicts and re-mapping the words
    _args.global_word_map = combine_word_dicts(_args.cws_dicts['words2idx'], _args.ner_dicts['words2idx']) 
    if _args.emb_init != 'RANDOM':
        print 'started loading embeddings from file', _args.emb_file
        M_emb, _args.global_word_map = read_matrix_and_idmap_from_file(_args.emb_file, _args.global_word_map)
        print 'global map size:', len(M_emb)
        ## load pretrained embeddings
        _args.emb_matrix = theano.shared(M_emb, name='emb_matrix')
        _args.emb_dim = len(M_emb[0])
        _args.m1_wemb1_out_dim = _args.emb_dim
        if _args.ner_fine_tuning :
            print 'fine tuning!!!!!'
            _args.emb_matrix.is_regularizable = True
    ## re-map words in the cws dataset
    _args.cws_train_set[1], _args.cws_valid_set[1], _args.cws_test_set[1] = batch_run_func((_args.cws_train_set[1], _args.cws_valid_set[1], _args.cws_test_set[1]), convert_word_idx, _args.cws_idx2word, _args.global_word_map)
    ## convert word, feature and label for array to numpy array
    _args.cws_train_set, _args.cws_valid_set, _args.cws_test_set = batch_run_func((_args.cws_train_set, _args.cws_valid_set, _args.cws_test_set), conv_data, _args.ner_win_l, _args.ner_win_r, len(_args.cws_dicts['features2idx']), len(_args.cws_dicts['labels2idx']))
    ## re-map words in the ner dataset
    _args.ner_train_set[1], _args.ner_valid_set[1], _args.ner_test_set[1] = batch_run_func((_args.ner_train_set[1], _args.ner_valid_set[1], _args.ner_test_set[1]), convert_word_idx, _args.ner_idx2word, _args.global_word_map)
    ## convert word, feature and label for array to numpy array
    _args.ner_train_set, _args.ner_valid_set, _args.ner_test_set = batch_run_func((_args.ner_train_set, _args.ner_valid_set, _args.ner_test_set), conv_data, _args.ner_win_l, _args.ner_win_r, len(_args.ner_dicts['features2idx']), len(_args.ner_dicts['labels2idx']))
    ### Parameters initialization, circuit compiliance and run training.
    prepare_params(_args)
    _args.f_cost, _args.f_update, _args.f_costs_and_updates, _args.f_classifies, cargs = create_joint_circuit(_args, StackConfig)
    print "Finished Compiling"
    run_model(_args, _args.train_mode)


def create_arg_parser(args=None):
    _arg_parser = argparse.ArgumentParser(description='LSTM')
    add_arg.arg_parser = _arg_parser
    ## File IO
    add_arg('--cws_train_path'        , '.')
    add_arg('--cws_valid_path'        , '.')
    add_arg('--cws_test_path'         , '.')
    add_arg('--ner_train_path'        , '.')
    add_arg('--ner_valid_path'        , '.')
    add_arg('--ner_test_path'         , '.')
    add_arg('--cws_use_features'         , False)
    add_arg('--ner_use_features'        , True)
    add_arg('--cws_circuit'         , 'LSTMOrderOneCRF')
    add_arg('--ner_circuit'        , 'plainOrderOneCRF')
    #add_arg('--cws_emb_init'       , 'RANDOM', help='The initial embedding type for cws')
    #add_arg('--ner_emb_init'       , 'FILE', help='The initial embedding type for ner')
    add_arg('--emb_init'       , 'RANDOM', help='The initial embedding type for cws')
    add_arg('--emb_file'       , '', help='The initial embedding file name')
    add_arg('--m1_wemb1_dropout_rate'    , 0.2, help='Dropout rate for the input embedding layer')
    add_arg('--use_emb'   , True, help='cws always use embeddings. so this always true. Just need to set it.')
    add_arg('--cws_use_emb'   , True, help='cws always use embeddings. so this always true. Just need to set it.')
    add_arg('--ner_use_emb'   , True)
    add_arg('--cws_fine_tuning'   , True)
    add_arg('--ner_fine_tuning'   , True)
    add_arg('--ner_eval_test', True, help='Whether evaluate the test data: test data may not have annotations.')
    add_arg('--ner_feature_thresh'     , 0)
    ## Task
    add_arg('--ner_oovthresh'     , 0    , help="The minimum count (upto and including) OOV threshold for NER") # Maybe 1 ?
    add_arg('--chunking_oovthresh', 0)
    add_arg('--pos_oovthresh'     , 2)
    ## Training
    add_arg_to_L(TRAIN_PARAM, '--train_mode'    , 'alternative' , help='possible train mode including joint, alternative, cws and ner')
    add_arg_to_L(TRAIN_PARAM, '--lr'           , 0.01)
    add_arg_to_L(TRAIN_PARAM, '--cws_lr'           , 0.01)
    add_arg_to_L(TRAIN_PARAM, '--ner_lr'           , 0.05)
    add_arg_to_L(TRAIN_PARAM, '--sample_coef'           , 10)
    add_arg_to_L(TRAIN_PARAM, '--nepochs'      , 30)
    add_arg_to_L(TRAIN_PARAM, '--ner_nepochs'      , 30)
    add_arg_to_L(TRAIN_PARAM, '--cws_nepochs'      , 10)
    add_arg_to_L(TRAIN_PARAM, '--cws_joint_weight' , 0.1)
    add_arg_to_L(TRAIN_PARAM, '--optimizer'    , 'sgd', help='sgd or adadelta')
    add_arg_to_L(TRAIN_PARAM, '--seed'         , 1) #int(random.getrandbits(10)))
    add_arg_to_L(TRAIN_PARAM, '--decay'        , True,  help='whether learning rate decay')
    add_arg_to_L(TRAIN_PARAM, '--cws_decay_epochs' , 5)
    add_arg_to_L(TRAIN_PARAM, '--ner_decay_epochs' , 10)
    add_arg_to_L(TRAIN_PARAM, '--minimum_lr'   , 1e-5)
    add_arg_to_L(TRAIN_PARAM, '--lower_case_input',     0)
    add_arg_to_L(TRAIN_PARAM, '--digit_to_zero'   ,     1)
    ## Topology
    add_arg_to_L(TOPO_PARAM, '--emission_trans_out_dim',    -1)
    add_arg_to_L(TOPO_PARAM, '--crf_viterbi',   False)
    add_arg_to_L(TOPO_PARAM, '--m1_wemb1_out_dim',                100)
    add_arg_to_L(TOPO_PARAM, '--m1_lstm_out_dim',                150)
    #add_arg_to_L(TOPO_PARAM, '--emb_output_transform_out_dim',500)
    #add_arg_to_L(TOPO_PARAM, '--lstm_activation_activation_fn',RELU_FN)
    add_arg_to_L(TOPO_PARAM, '--L2Reg_reg_weight',             0.0)
    add_arg_to_L(TOPO_PARAM, '--cws_win_l',                          0)
    add_arg_to_L(TOPO_PARAM, '--ner_win_l',                          0)
    add_arg_to_L(TOPO_PARAM, '--cws_win_r',                          2)
    add_arg_to_L(TOPO_PARAM, '--ner_win_r',                          0)
    ## DEBUG
    add_arg('--verbose'      , 2)
    add_arg('--debugtopo'    , False)

    return _arg_parser
    
    
if __name__ == "__main__":
    #######################################################################################
    ## PARSE ARGUMENTS, BUILD CIRCUIT, TRAIN, TEST
    #######################################################################################
    TOPO_PARAM = []
    TRAIN_PARAM = []
    RELU_FN = SerializableLambda('lambda x: x + theano.tensor.abs_(x)')
    SIGMOID_FN = SerializableLambda('lambda x: theano.tensor.nnet.sigmoid(x)')
    TANH_FN = SerializableLambda('lambda x: 2*theano.tensor.nnet.sigmoid(2*x) - 1')
    _arg_parser = create_arg_parser()
    args = _arg_parser.parse_args()
    main(args)
