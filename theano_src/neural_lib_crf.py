import os
import theano.tensor as T
import theano
import time
from theano import config
import numpy as np
import collections
#theano.config.compute_test_value = 'off'
#theano.config.profile=True
#theano.config.profile_memory=True

np.random.seed(1)
def make_name(*params):
    """
    Join the params as string using '_'
    and also add a unique id, since every node in a theano
    graph should have a unique name
    """
    if not hasattr(make_name, "uid"):
        make_name.uid = 0
    make_name.uid += 1
    tmp = "_".join(params)
    return "_".join(['tparam', tmp, str(make_name.uid)])


def reverse(tensor):
    rev, _ = theano.scan(lambda itm: itm,
                         sequences=tensor,
                         go_backwards=True,
                         strict=True,
                         name='reverse_rand%d'%np.random.randint(1000))
    return rev


def read_matrix_from_file(fn, dic):
    '''
    Assume that the file contains words in first column,
    and embeddings in the rest and that dic maps words to indices.
    '''
    _data = open(fn).read().strip().split('\n')
    _data = [e.strip().split() for e in _data]
    dim = len(_data[0]) - 1
    data = {}
    # NOTE: The norm of onesided_uniform rv is sqrt(n)/sqrt(3)
    # Since the expected value of X^2 = 1/3 where X ~ U[0, 1]
    # => sum(X_i^2) = dim/3
    # => norm       = sqrt(dim/3)
    # => norm/dim   = sqrt(1/3dim)
    multiplier = np.sqrt(1.0/(3*dim))
    for e in _data:
        r = np.array([float(_e) for _e in e[1:]])
        data[e[0]] = (r/np.linalg.norm(r)) * multiplier
    M = ArrayInit(ArrayInit.onesided_uniform, multiplier=1.0/dim).initialize(len(data), dim)
    for word, idx in dic.iteritems():
        if word in data:
            M[idx] = data[word]
    return M


class ArrayInit(object):
    normal = 'normal'
    onesided_uniform = 'onesided_uniform'
    twosided_uniform = 'twosided_uniform'
    ortho = 'ortho'
    zero = 'zero'
    unit = 'unit'
    ones = 'ones'
    fromfile = 'fromfile'
    def __init__(self, option,
                 multiplier=0.01,
                 matrix=None,
                 word2idx=None):
        self.option = option
        self.multiplier = multiplier
        self.matrix_filename = None
        self.matrix = self._matrix_reader(matrix, word2idx)
        if self.matrix is not None:
            self.multiplier = 1
        return

    def _matrix_reader(self, matrix, word2idx):
        if type(matrix) is str:
            self.matrix_filename = matrix
            assert os.path.exists(matrix), "File %s not found"%matrix
            matrix = read_matrix_from_file(matrix, word2idx)
            return matrix
        else:
            return None

    def initialize(self, *xy, **kwargs):
        if self.option == ArrayInit.normal:
            M = np.random.randn(*xy)
        elif self.option == ArrayInit.onesided_uniform:
            M = np.random.rand(*xy)
        elif self.option == ArrayInit.twosided_uniform:
            M = np.random.uniform(-1.0, 1.0, xy)
        elif self.option == ArrayInit.ortho:
            f = lambda dim: np.linalg.svd(np.random.randn(dim, dim))[0]
            if int(xy[1]/xy[0]) < 1 and xy[1]%xy[0] != 0:
                raise ValueError(str(xy))
            M = np.concatenate(tuple(f(xy[0]) for _ in range(int(xy[1]/xy[0]))),
                               axis=1)
            assert M.shape == xy
        elif self.option == ArrayInit.zero:
            M = np.zeros(xy)
        elif self.option in [ArrayInit.unit, ArrayInit.ones]:
            M = np.ones(xy)
        elif self.option == ArrayInit.fromfile:
            assert isinstance(self.matrix, np.ndarray)
            M = self.matrix
        else:
            raise NotImplementedError
        self.multiplier = (kwargs['multiplier']
                           if ('multiplier' in kwargs
                               and kwargs['multiplier'] is not None)
                           else self.multiplier)
        return (M*self.multiplier).astype(config.floatX)

    def __repr__(self):
        mults = ', multiplier=%s'%((('%.3f'%self.multiplier)
                                     if type(self.multiplier) is float
                                     else str(self.multiplier)))
        mats = ((', matrix="%s"'%self.matrix_filename)
                if self.matrix_filename is not None
                else '')
        return "ArrayInit(ArrayInit.%s%s%s)"%(self.option, mults, mats)


class StackConfig(collections.MutableMapping):
    """A dictionary like object that would automatically recognize
    keys that end with the following pattern and return appropriate keys.
    _out_dim  :
    _initializer :
    The actions to take are stored in a list for easy composition.
    """
    actions = [
        (lambda key: key.endswith('_out_dim')       , lambda x: x),
        (lambda key: key.endswith('_T_initializer') , ArrayInit(ArrayInit.onesided_uniform)),
        (lambda key: key.endswith('_W_initializer') , ArrayInit(ArrayInit.twosided_uniform, multiplier=1)),
        (lambda key: key.endswith('_N_initializer') , ArrayInit(ArrayInit.normal)),
        (lambda key: key.endswith('_viterbi')       , False),
        (lambda key: key.endswith('_reg_weight')    , 0),
    ]
    def __init__(self, dictionary):
        self.store = collections.OrderedDict()
        self.store.update(dictionary)

    def __getitem__(self, key):
        if key in self.store:
            return self.store[key]
        for (predicate, retval) in self.actions:
            if predicate(key):
                return retval
        raise KeyError(key)

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def reset(self):
        for k in self.store:
            if k.startswith('tparam_'):
                del self.store[k]
        return


class Chip(object):
    """ A Chip object requires name and a param dictionary
    that contains param[name+'_'+out_dim] (This can be a function that depends on the input_dim)
    Other than that it must also contain appropriate initializers for all the parameters.

    The params dictionary is updated to contain 'tparam_<param name>_uid'
    """
    def __init__(self, name, params=None):
        """ I set the output dimension of every node in the parameters.
            The input dimension would be set when prepend is called.
            (Since prepend method receives the previous chip)
        """
        self.name = name
        if params is not None:
            self.out_dim = params[self.kn('out_dim')]
            self.params = params
        return

    def prepend(self, previous_chip):
        """ Note that my input_dim of self = output_dim of previous_chip
        Also we keep track of absolute_input (the first input) to the layer
        """
        self.in_dim = previous_chip.out_dim * previous_chip.params[previous_chip.kn('win')] if (hasattr(previous_chip, 'kn') and previous_chip.kn('win') in previous_chip.params) else previous_chip.out_dim
        if hasattr(self.out_dim, '__call__'):
            self.out_dim = self.out_dim(self.in_dim)
        # Let the assert take care of everything
        internal_params = self.construct(previous_chip.output_tv)
        assert self.output_tv is not None
        #self.output_tv.name = make_name(self.name, 'output_tv')
        for k in internal_params:
            self.params[k.name] = k
        return self

    def construct(self, input_tv):
        """ Note that input_tv = previous_chip.output_tv
        This method returns a dictionary of internal weight params
        and This method sets self.output_tv
        """
        raise NotImplementedError

    def regularizable_variables(self):
        """ If a value stored in the dictionary has the attribute
        is_regularizable then that value is regularizable
        """
        return [k for k in self.params
                if hasattr(self.params[k], 'is_regularizable')
                and self.params[k].is_regularizable]

    def kn(self, thing):
        """ knm stands for key name
        """
        if len(thing) == 1: # It is probably ['U', 'W', 'b', 'T', 'N'] or some such Matrix
            keyname_suffix = '_initializer'
        else:
            keyname_suffix = ''
        return self.name + '_' + thing + keyname_suffix

    def needed_key(self):
        return self._needed_key_impl()

    def _needed_key_impl(self, *things):
        return [self.kn(e) for e in ['out_dim'] + list(things)]

    def _declare_mat(self, name, *dim, **kwargs):
        multiplier = (kwargs['multiplier']
                      if 'multiplier' in kwargs
                      else None)
        var = theano.shared(
            self.params[self.kn(name)].initialize(*dim, multiplier=multiplier),
            name=make_name(self.name, name)
        )
        if 'is_regularizable' not in kwargs:
            var.is_regularizable = True # Default
        else:
            var.is_regularizable = False
        return var

class Start(object):
    """ A start object which has all the necessary attributes that
    any chip object that would call it would need.
    """
    def __init__(self, out_dim, output_tv):
        self.out_dim = out_dim
        self.output_tv = output_tv


class ComputeFeature(Chip):
    """ This chip converts features to transition scores.
    This requires a W_initializer
    """ 
    def construct(self, input_tv):
        #assert 'emb_dim' in self.params
        T_ = self._declare_mat('W', self.in_dim, self.out_dim, multiplier=1.0/self.out_dim)
        Pad = np.zeros((1, self.out_dim))
        FW = T.concatenate([T_, Pad], axis=0)
        if 'emb_dim' in self.params: #self.params.use_emb:
            EW = self._declare_mat('W', self.params['emb_dim'], self.out_dim, multiplier=1.0/self.out_dim)
        feat_tv, emb_tv = input_tv
        n_timesteps = feat_tv.shape[0]
        n_features = feat_tv.shape[1]
        chain_potential = self._declare_mat('T', self.out_dim, self.out_dim,  multiplier=1.0/self.out_dim)
        feat_emission = FW[feat_tv.flatten()].reshape([n_timesteps, n_features, self.out_dim], ndim=3).sum(axis=1)  #transsition
        emission_potential = feat_emission 	
        if 'emb_dim' in self.params: #self.params.use_emb:
            emb_new = self.params['emb_matrix'][emb_tv]
            emb_emission = T.dot(emb_new, EW) 
            emission_potential += emb_emission
        self.output_tv = (emission_potential, chain_potential)
        if 'emb_dim' in self.params: #self.params.use_emb:
            return (T_, EW, chain_potential)
        else:
            return (T_, chain_potential)


    def needed_key(self):
        return []


class Activation(Chip):
    """ This requires a _activation_fn parameter
    """
    def construct(self, input_tv):
        self.output_tv = self.params[self.kn('activation_fn')](input_tv)
        return tuple()

    def needed_key(self):
        return self._needed_key_impl('activation_fn')


class ScorableChip(Chip):
    """ A ScorableChip is defined by its ability of producing a score of an (input, output) tuple.
    This type of Chip would usually be the last in a stack, though it's not necessary, for example
    Conjunctive features can use output classes to produce scores even if they are second last.

    Usually though the supervision does not penetrate down.
    """
    def prepend(self, previous_chip):
        retval = super(ScorableChip, self).prepend(previous_chip)
        assert hasattr(self, 'score')
        assert hasattr(self, 'gold_y')
        return retval


class OrderOneCrf(ScorableChip):
    """ Find the highest scoring path in a 3D-tensor [T]
    Value of T(i,j,k)  = transition(state k | state j) + emission(output i | state k)
    let [I, J, K] be dimensions of the tensor
    N be length of sentence,
    and S be the number of states
    I = N
    J = S 
    K = S
    T[0, -2, :] contains the score of starting from <BOS> and
         then transitioning to hidden state and emitting output 0

    This requires '_viterbi' in the param
    """
    def _th_logsumexp(self, x, axis=None):
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def construct(self, input_tv):
        # The input is a tensor of scores.
        assert len(input_tv) == 2
        emission_prob, chain_potential = input_tv
        assert emission_prob.ndim == 2
        assert chain_potential.ndim == 2
        viterbi_flag = self.params[self.kn('viterbi')]
        #print 'in the crf layer, viterbi_flag=', viterbi_flag
        def forward_step(obs2tags, prev_result, chain_potentials):
            assert obs2tags.ndim == 1
            assert prev_result.ndim == 1
            f_ = prev_result.dimshuffle(0, 'x') + obs2tags.dimshuffle('x', 0) + chain_potentials
            p = (T.max(f_, axis=0)
                    if viterbi_flag
                    else self._th_logsumexp(f_, axis=0))
            y = T.argmax(f_, axis=0).astype('int32')
            return p, y

        initial = emission_prob[0] #* T.ones_like(T_))
        # rval is the lattice of forward scores. bp contains backpointers
        [rval, bp], _ = theano.scan(forward_step,
            sequences=[emission_prob[1:]],
            outputs_info=[initial, None],
            non_sequences=[chain_potential],
            name='OrderOneCrf_scan__step',
            strict=True)
        # The most likely state based on the forward/alpha scores.
        yn = rval[-1].argmax(axis=0).astype('int32')
        path_tmp, _ = theano.scan(lambda back_pointer, y: back_pointer[y],
                                  sequences=bp,
                                  outputs_info=yn,
                                  go_backwards=True,
                                  name='OrderOneCrf_scan__bkpntr',
                                  strict=True)
        # The output_tv contains the sequence that has the highest score.
        self.output_tv = T.concatenate([reverse(path_tmp), yn.dimshuffle('x')])
        self._partition_val = self._th_logsumexp(rval[-1], axis=0)
        self._partition_val.name = make_name(self.name, '_partition_val')
        self.gold_y = T.ivector(make_name(self.name, 'gold_y')).astype('int32')
    
    
        def _output_score(obs2tags, chain_potentials, y):
            def _score_step(o, y, p_, y_, c):
                return ((p_ + c[y_, y] + o[y]), y)
        
            y0 = y[0]
            p0 = obs2tags[0][y0]
            [rval, _], _ = theano.scan(_score_step,
                sequences=[obs2tags[1:], y[1:]],
                outputs_info=[p0, y0],
                non_sequences=[chain_potentials],
                name='OrderOneCrf_scan_score_step',
                strict=True)
            return rval[-1]
        # This is the score of the gold sequence. We need this.
        self.score = (_output_score(emission_prob, chain_potential, self.gold_y) - self._partition_val)
        return (chain_potential,)


    def needed_key(self):
        return self._needed_key_impl('viterbi')


class L2Reg(Chip):
    """ This supposes that the previous chip would have a score attribute.
    And it literally only changes the score attribute by adding the regularization term
    on top of it.
    """
    def prepend(self, previous_chip):
        self.previous_chip = previous_chip
        return super(L2Reg, self).prepend(previous_chip)

    def construct(self, input_tv):
        L2 = T.sum(T.stack([T.sum(self.params[k]*self.params[k])
                            for k
                            in self.regularizable_variables()]))
        L2.name = make_name(self.name, 'L2')
        self.score = self.previous_chip.score - self.params[self.kn('reg_weight')] * L2
        return (L2,)

    def __getattr__(self, item):
        """ Inherit all the attributes of the previous chip.
        At present I can only see this functionality being useful
        for the case of the Regularization chip. Maybe we would move
        this down in case it is found necessary later on, but there is
        chance of abuse.
        """
        try:
            return getattr(self.previous_chip, item)
        except KeyError:
            raise AttributeError(item)

    def needed_key(self):
        return self._needed_key_impl('reg_weight')

def calculate_params_needed(chips):
    l = []
    for c in chips:
        l += c[0](c[1]).needed_key()
    return l

def metaStackMaker(chips, params, grad_nparr_expected_size=(100, 100, 50)):
    params_needed = ['in_dim']
    params_needed += calculate_params_needed(chips)
    # Show all parameters that would be needed in this system
    print "Parameters Needed", params_needed
    for k in params_needed:
        assert k in params, k
	print k, params[k]
    feature_input = T.imatrix('feature_input')
    #if 'emb_dim' in params: #params.use_emb:
    emb_input = T.ivector('emb_input')
    current_chip = Start(params['in_dim'], (feature_input, emb_input)) 
    #else:
    #	current_chip = Start(params['in_dim'], feature_input)
	
    print '\n', 'Building Stack now', '\n', 'Start: ', params['in_dim']
    for e in chips:
        current_chip = e[0](e[1], params).prepend(current_chip)
        print e[1], "In_dim:", current_chip.in_dim, "Out_dim:", current_chip.out_dim
        for e in current_chip.needed_key():
            print (e, params[e])
    pred_y = current_chip.output_tv
    gold_y = (current_chip.gold_y
              if hasattr(current_chip, 'gold_y')
              else None)
    #gold_score, partition_score = current_chip.score if hasattr(current_chip, 'score') else None
    if hasattr(current_chip, 'score'):
        cost = -current_chip.score #(current_chip.score[0]-current_chip.score[1])
        pred_y_grad_nparr = None
        known_grads = None
    else:
        cost = None
        pred_y_grad_nparr = np.zeros(grad_nparr_expected_size).astype(np.float32)
        known_grads = {pred_y: theano.shared(pred_y_grad_nparr, borrow=True)}
    grads = T.grad(cost,
                   wrt=[params[k] for k in params if hasattr(params[k], 'is_regularizable')],
                   known_grads=known_grads)
    # output_tv = [p for k, p in params if k.startswith['tparams_']]
    print '\n Regularizable parameters:'
    for k, v in params.items():
    	if hasattr(v, 'is_regularizable'):
    	    print k, v
    if gold_y is not None:
        f_debug = theano.function([feature_input, emb_input, gold_y], [feature_input, emb_input, gold_y, pred_y], on_unused_input='warn')
    else:
        f_debug = theano.function([feature_input, emb_input], [feature_input, emb_input, pred_y], on_unused_input='warn')
    cost_or_gradarr = (pred_y_grad_nparr
                       if cost is None
                       else cost)
    return (feature_input, emb_input, gold_y, pred_y, cost_or_gradarr, grads, f_debug)


def plainOrderOneCRF(params): # simplecrf1
    chips = [
        (ComputeFeature,      'emission_trans'),
        (OrderOneCrf,   'crf'),
        (L2Reg,             'L2Reg'),
        ]
    return metaStackMaker(chips, params)


