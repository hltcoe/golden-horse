import theano.tensor as T
#from neural_lib_crf import *
from neural_lib import *
import numpy as np
from train_util import print_args

# Note: need to refactor many places to return regularizable params  lists for the optimization.

def calculate_params_needed(chips):
    l = []
    for c, _ in chips:
        l += c.needed_key()
    return l

# Automatically create(initialize) layers and hook them together
def stackLayers(chips, current_chip, params, feature_size=0):
    instantiated_chips = []
    print 'stack layers!!!'
    for e in chips:
        previous_chip = current_chip
        if e[1].endswith('feature_emission_trans'):
            current_chip = e[0](e[1], params).prepend(previous_chip, feature_size)
        else:
            current_chip = e[0](e[1], params).prepend(previous_chip)
        instantiated_chips.append((current_chip, e[1]))
        print 'current chip:', e[1], "In_dim:", current_chip.in_dim, "Out_dim:", current_chip.out_dim
        print 'needed keys:'
        for e in current_chip.needed_key():
            print (e, params[e])
    return instantiated_chips 

# Compute the initialized layers by feed in the inputs.
def computeLayers(instantiated_chips, current_chip, params, feature_input=None):
    print 'compute layers!!!'
    regularizable_params = []
    for e in instantiated_chips:
        previous_chip = current_chip
        current_chip = e[0]
        print 'current chip:', e[1], "In_dim:", current_chip.in_dim, "Out_dim:", current_chip.out_dim
        if e[1].endswith('feature_emission_trans'):
            internal_params = current_chip.compute(previous_chip.output_tv, feature_input)
        else:
            internal_params = current_chip.compute(previous_chip.output_tv)
        assert current_chip.output_tv is not None
        for k in internal_params:
            print 'internal_params:', k.name
            #assert k.is_regularizable
            if k.is_regularizable:
                params[k.name] = k
                regularizable_params.append(k)
    return regularizable_params

def metaStackMaker(chips, params):
    feature_input = T.imatrix('feature_input')
    emb_input = T.imatrix('emb_input')
    current_chip = Start(params['voc_size'], emb_input) #(feature_input, emb_input)) 

    print '\n', 'Building Stack now', '\n', 'Start: ', params['voc_size'], 'out_tv dim:', current_chip.output_tv.ndim
    feature_size = params['feature_size']
    instantiated_chips = stackLayers(chips, current_chip, params, feature_size)
    regularizable_params = computeLayers(instantiated_chips, current_chip, params, feature_input)
    current_chip = instantiated_chips[-1][0]
    pred_y = current_chip.output_tv
    gold_y = (current_chip.gold_y
            if hasattr(current_chip, 'gold_y')
            else None)
    print 'gold_y:', gold_y
    # Show all parameters that would be needed in this system
    params_needed = calculate_params_needed(instantiated_chips)
    print "Parameters Needed", params_needed
    for k in params_needed:
        assert k in params, k
        print k, params[k]
    assert hasattr(current_chip, 'score')
    cost = current_chip.score #/ params['nsentences'] 
    grads = T.grad(cost,
            wrt=regularizable_params)
            #[params[k] for k in params if (hasattr(params[k], 'is_regularizable') and params[k].is_regularizable)])
    print 'Regularizable parameters:'
    for k, v in params.items():
        if hasattr(v, 'is_regularizable'):
            print k, v, v.is_regularizable
    return (feature_input, emb_input, gold_y, pred_y, cost, grads, regularizable_params) #, f_debug)


def SegNERStackMaker(module1, module2, params, Lambda):
    feature_inputs = [T.imatrix('feature_input_'+str(i)) for i in range(2)]
    emb_inputs = [T.imatrix('emb_input_'+str(i)) for i in range(2)]
    current_chip = Start(params['voc_size'], emb_inputs[0])  

    print '\n', 'Building Stack now', '\n', 'Start: ', params['voc_size'], 'out_tv dim:', current_chip.output_tv.ndim
    feature_size = params['feature_size_1']
    instantiated_chips = stackLayers(module1, current_chip, params, feature_size)
    regularizable_params = computeLayers(instantiated_chips, current_chip, params, feature_inputs[0])
    current_chip = instantiated_chips[-1][0]
    pred_ys = []
    gold_ys = []
    costs_arr = []
    grads_arr = []
    regularizable_param_arr = []
    global_regularizable_params = []
    pred_ys.append(current_chip.output_tv)
    gold_ys.append(current_chip.gold_y)
    assert hasattr(current_chip, 'score')
    #print type(Lambda), type(current_chip.score), current_chip.score
    Lambda = np.float32(Lambda)
    cost = Lambda * current_chip.score 
    #print type(Lambda), type(cost), cost
    grads = T.grad(cost,
            wrt=regularizable_params)
    costs_arr.append(cost)
    grads_arr.append(grads)
    regularizable_param_arr.append(regularizable_params)
    global_regularizable_params.extend(regularizable_params)
    print 'SEGMENTATION module regularizable parameters:'
    for k, v in params.items():
        if hasattr(v, 'is_regularizable'):
            print k, v, v.is_regularizable
    # Show all parameters that would be needed in this system
    params_needed = ['voc_size', 'feature_size_1']
    params_needed += calculate_params_needed(instantiated_chips)
    print "Module 1 Parameters Needed", params_needed
    for k in params_needed:
        assert k in params, k
        print k, params[k]
    # Computation for module 2
    regularizable_params = []
    internal_params = instantiated_chips[0][0].compute(emb_inputs[1])
    regularizable_params.extend(internal_params)
    emb_tv = instantiated_chips[0][0].output_tv
    internal_params = instantiated_chips[1][0].compute(emb_tv)
    regularizable_params.extend(internal_params)
    lstm_htv = instantiated_chips[1][0].output_tv
    win_size=instantiated_chips[0][0].params[instantiated_chips[0][0].kn('win')] if hasattr(instantiated_chips[0][0], 'kn') and instantiated_chips[0][0].kn('win') in params else 1
    current_chip = Start(instantiated_chips[0][0].out_dim*win_size+instantiated_chips[1][0].out_dim, T.concatenate([emb_tv, lstm_htv], axis=1) )
    #current_chip = Start(instantiated_chips[0][0].out_dim+instantiated_chips[1][0].out_dim, T.concatenate([emb_tv, lstm_htv], axis=1) )
    #current_chip = Start(instantiated_chips[0][0].out_dim, emb_tv)
    #current_chip = Start(instantiated_chips[1][0].out_dim, lstm_htv)
    feature_size = params['feature_size_2']
    instantiated_chips = stackLayers(module2, current_chip, params, feature_size)
    internal_params = computeLayers(instantiated_chips, current_chip, params, feature_inputs[1])
    regularizable_params.extend(internal_params)
    current_chip = instantiated_chips[-1][0]
    pred_ys.append(current_chip.output_tv)
    gold_ys.append(current_chip.gold_y) 
    assert hasattr(current_chip, 'score')
    cost = current_chip.score #/ params['nsentences'] 
    grads = T.grad(cost,
            wrt=regularizable_params)
    costs_arr.append(cost)
    grads_arr.append(grads)
    regularizable_param_arr.append(regularizable_params)
    global_regularizable_params.extend(regularizable_params)
    print 'NER module regularizable parameters:'
    for v in regularizable_params:
        assert hasattr(v, 'is_regularizable')
        print v, v.is_regularizable
    cost = sum(costs_arr)
    global_regularizable_params = list(set(global_regularizable_params))
    grads = T.grad(cost,
            wrt=global_regularizable_params)
    print 'The joint model regularizable parameters:'
    for k, v in params.items():
        if hasattr(v, 'is_regularizable'):
            print k, v, v.is_regularizable
    return (feature_inputs, emb_inputs, gold_ys, pred_ys, costs_arr, cost, grads_arr, grads, regularizable_param_arr, global_regularizable_params)


def plainOrderOneCRF(params): # simplecrf1
    chips = [
            (Embedding          ,'wemb1'),
            (BiasedLinear       ,'emb_output_transform'),
            (ComputeFeature,      'feature_emission_trans'),
            (OrderOneCrf,   'crf'),
            (L2Reg,             'L2Reg'),
            ]
    return metaStackMaker(chips, params)


def LSTMOrderOneCRF(params):
    chips = [
            (Embedding          ,'wemb1'),
            (LSTM               ,'lstm'),
            (BiasedLinear       ,'emb_output_transform'),
            (ComputeFeature,      'feature_emission_trans'),
            (OrderOneCrf        ,'crf'),
            (L2Reg              ,'L2Reg'),
            ]
    return metaStackMaker(chips, params)


def BiLSTMOrderOneCRF(params):
    chips = [
            (Embedding          ,'wemb1'),
            (BiLSTM               ,'lstm'),
            (BiasedLinear       ,'emb_output_transform'),
            (OrderOneCrf        ,'crf'),
            (L2Reg              ,'L2Reg'),
            ]
    return metaStackMaker(chips, params)


def LSTMwithFeatureOrderOneCRF(params):
    chips = [
            (Embedding          ,'wemb1'),
            (LSTM               ,'lstm'),
            (BiasedLinear       ,'emb_output_transform'),
            (ComputeFeature     ,'feature_emission_trans'),
            (OrderOneCrf        ,'crf'),
            (L2Reg              ,'L2Reg'),
            ]
    return metaStackMaker(chips, params)

def jointSegmentationNER(params, Lambda=0.1):
    module1 = [
              (Embedding          ,'m1_wemb1'),
              (LSTM               ,'m1_lstm'),
              (BiasedLinear       ,'m1_hidden_output_trans'),
              #(ComputeFeature     ,'m1_feature_emission_trans'),
              (OrderOneCrf        ,'m1_crf_decode'),
              #(L2Reg,             'L2Reg'),
              ] 
    module2 = [
              (BiasedLinear       ,'m2_hidden_output_trans'),
              (ComputeFeature,      'm2_feature_emission_trans'),
              (OrderOneCrf,   'm2_crf_decode'),
              #(L2Reg,             'L2Reg'),
              ]
    return SegNERStackMaker(module1, module2, params, Lambda)

