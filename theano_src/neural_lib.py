import os, sys
#sys.path.insert(0, '/home/npeng/.local/lib/python2.7/site-packages/')
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
def name_tv(*params):
    """
    Join the params as string using '_'
    and also add a unique id, since every node in a theano
    graph should have a unique name
    """
    if not hasattr(name_tv, "uid"):
        name_tv.uid = 0
    name_tv.uid += 1
    tmp = "_".join(params)
    return "_".join(['tparam', tmp, str(name_tv.uid)])


def np_floatX(data):
     return np.asarray(data, dtype=config.floatX)

def tparams_make_name(*params):
    tmp = make_name(*params)
    return "_".join(['tparam', tmp])

def make_name(*params):
    """
    Join the params as string using '_'
    and also add a unique id, since every node in a theano
    graph should have a unique name
    """
    return "_".join(params)

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
        r = np.array([np_floatX(_e) for _e in e[1:]])
        data[e[0]] = (r/np.linalg.norm(r)) * multiplier
    M = ArrayInit(ArrayInit.onesided_uniform, multiplier=1.0/dim).initialize(len(data), dim)
    for word, idx in dic.iteritems():
        if word in data:
            M[idx] = data[word]
    return M


def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output

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
            M = np.random.randn(*xy).astype(config.floatX)
        elif self.option == ArrayInit.onesided_uniform:
            M = np.random.rand(*xy).astype(config.floatX)
        elif self.option == ArrayInit.twosided_uniform:
            M = np.random.uniform(-1.0, 1.0, xy).astype(config.floatX)
        elif self.option == ArrayInit.ortho:
            f = lambda dim: (np.linalg.svd(np.random.randn(dim, dim))[0]).astype(config.floatX)
            if int(xy[1]/xy[0]) < 1 and xy[1]%xy[0] != 0:
                raise ValueError(str(xy))
            M = np.concatenate(tuple(f(xy[0]) for _ in range(int(xy[1]/xy[0]))),
                    axis=1)
            assert M.shape == xy
        elif self.option == ArrayInit.zero:
            M = np.zeros(xy).astype(config.floatX)
        elif self.option in [ArrayInit.unit, ArrayInit.ones]:
            M = np.ones(xy).astype(config.floatX)
        elif self.option == ArrayInit.fromfile:
            assert isinstance(self.matrix, np.ndarray)
            M = self.matrix.astype(config.floatX)
        else:
            raise NotImplementedError
        #self.multiplier = (kwargs['multiplier']
        multiplier = (kwargs['multiplier']
            if ('multiplier' in kwargs
                    and kwargs['multiplier'] is not None)
                else self.multiplier)
        #return (M*self.multiplier).astype(config.floatX)
        return (M*multiplier).astype(config.floatX)

    def __repr__(self):
        mults = ', multiplier=%s'%((('%.3f'%self.multiplier)
            if type(self.multiplier) is float
            else str(self.multiplier)))
        mats = ((', matrix="%s"'%self.matrix_filename)
                if self.matrix_filename is not None
                else '')
        return "ArrayInit(ArrayInit.%s%s%s)"%(self.option, mults, mats)


class SerializableLambda(object):
    def __init__(self, s):
        self.s = s
        self.f = eval(s)
        return

    def __repr__(self):
        return "SerializableLambda('%s')"%self.s

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)


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
            (lambda key: key.endswith('_U_initializer') , ArrayInit(ArrayInit.ortho, multiplier=1)),
            (lambda key: key.endswith('_W_initializer') , ArrayInit(ArrayInit.twosided_uniform, multiplier=1)),
            (lambda key: key.endswith('_N_initializer') , ArrayInit(ArrayInit.normal)),
            (lambda key: key.endswith('_b_initializer') , ArrayInit(ArrayInit.zero)),
            (lambda key: key.endswith('_p_initializer') , ArrayInit(ArrayInit.twosided_uniform, multiplier=1)),
            (lambda key: key.endswith('_c_initializer') , ArrayInit(ArrayInit.twosided_uniform, multiplier=1)),
            (lambda key: key.endswith('_reg_weight')    , 0),
            (lambda key: key.endswith('_viterbi')     , False),
            (lambda key: key.endswith('_begin')         , 1),
            (lambda key: key.endswith('_end')           , -1),
            #(lambda key: key.endswith('_activation_fn') , lambda x: x + theano.tensor.abs_(x)),
            #(lambda key: key.endswith('_v_initializer') , ArrayInit(ArrayInit.ones, multiplier=NotImplemented)),
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
            print 'current chip:', name, 'out dimension:', self.kn('out_dim')
            self.out_dim = params[self.kn('out_dim')]
            print 'init chip:', self.name, 'out dim:', self.out_dim
            self.params = params
        return

    def prepend(self, previous_chip):
        """ Note that my input_dim of self = output_dim of previous_chip
        Also we keep track of absolute_input (the first input) to the layer
        """
        #if hasattr(previous_chip, 'kn') and previous_chip.kn('win') in previous_chip.params:
        if hasattr(previous_chip, 'kn') and previous_chip.kn('win') in previous_chip.params:
            print 'window size:', previous_chip.params[previous_chip.kn('win')]
        self.in_dim = previous_chip.out_dim * previous_chip.params[previous_chip.kn('win')] if (hasattr(previous_chip, 'kn') and previous_chip.kn('win') in previous_chip.params) else previous_chip.out_dim
        print 'in prepend, chip', self.name, 'in dim =', self.in_dim, 'out dim =', self.out_dim
        if hasattr(self.out_dim, '__call__'):
            self.out_dim = self.out_dim(self.in_dim)
        print self.name, ' in prepend, in dim =', self.in_dim, 'out dim =', self.out_dim
        return self

    def compute(self, input_tv):
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
        if len(thing) == 1: # It is probably ['U', 'W', 'b', 'T', 'N'] or some such Matrix
            keyname_suffix = '_initializer'
        else:
            keyname_suffix = ''
        return self.name + '_' + thing + keyname_suffix

    def _declare_mat(self, name, *dim, **kwargs):
        multiplier = (kwargs['multiplier']
                if 'multiplier' in kwargs
                else None)
        var = theano.shared(
                self.params[self.kn(name)].initialize(*dim, multiplier=multiplier),
                name=tparams_make_name(self.name, name)
                )
        if 'is_regularizable' not in kwargs:
            var.is_regularizable = True # Default
        else:
            var.is_regularizable = kwargs['is_regularizable']
        return var

    def needed_key(self):
        return self._needed_key_impl()

    def _needed_key_impl(self, *things):
        return [self.kn(e) for e in ['out_dim'] + list(things)]

class Start(object):
    """ A start object which has all the necessary attributes that
    any chip object that would call it would need.
    """
    def __init__(self, out_dim, output_tv):
        self.out_dim = out_dim
        self.output_tv = output_tv

# Note: should make changes here to pass pre_defined embeddings as parameters.
class Embedding(Chip):
    def prepend(self, previous_chip):
        self = super(Embedding, self).prepend(previous_chip)
        if 'emb_matrix' in self.params:
            print 'pre_trained embedding!!'
            self.T_ = self.params['emb_matrix']
            print self.T_, type(self.T_)
        else:
            self.T_ = self._declare_mat('T', self.params['voc_size'], self.out_dim)
            self.params['emb_dim'] = self.out_dim
        return self

    """ An embedding converts  one-hot-vectors to dense vectors.
    We never take dot products with one-hot-vectors.
    This requires a T_initializer
    """
    def compute(self, input_tv):
        print input_tv, type(input_tv)
        n_timesteps = input_tv.shape[0]
        if input_tv.ndim == 3:
            batch_size = input_tv.shape[1]
        print 'input_tv dimension:', input_tv.ndim
        #self.params[self.kn('win')] = window_size
        window_size = self.params[self.kn('win')]
        if input_tv.ndim < 3:
            self.output_tv = self.T_[input_tv.flatten()].reshape([n_timesteps, window_size * self.out_dim], ndim=2)
        else:
            self.output_tv = self.T_[input_tv.flatten()].reshape([n_timesteps, batch_size, window_size * self.out_dim], ndim=3)
        if self.params.get(self.kn('dropout_rate'), 0.0) != 0.0:
            print 'DROP OUT!!! at circuite', self.name, 'Drop out rate: ', self.params[self.kn('dropout_rate')]
            self.output_tv = _dropout_from_layer(self.params['rng'], self.output_tv, self.params[self.kn('dropout_rate')])
        return (self.T_,)

    def needed_key(self):
        return self._needed_key_impl('T')


class ComputeFeature(Chip):
    def prepend(self, previous_chip, feature_size):
        self = super(ComputeFeature, self).prepend(previous_chip)
        self.T_ = self._declare_mat('W', feature_size, self.out_dim, multiplier=1.0/self.out_dim)
        Pad = np.zeros((1, self.out_dim), dtype=config.floatX)
        self.FW = T.concatenate([self.T_, Pad], axis=0)
        return self

    """ This chip converts features to transition scores.
    This requires a W_initializer
    """ 
    def compute(self, input_tv, feat_tv):
        #feat_tv = self.params['feature_input']
        if feat_tv.ndim == 2:
            n_timesteps = feat_tv.shape[0]
            n_features = feat_tv.shape[1]
            feat_emission = self.FW[feat_tv.flatten()].reshape([n_timesteps, n_features, self.out_dim], ndim=3).sum(axis=1)  #transsition
        elif feat_tv.ndim == 3:
            n_timesteps = feat_tv.shape[0]
            batch_size = feat_tv.shape[1]
            n_features = feat_tv.shape[2]
            feat_emission = self.FW[feat_tv.flatten()].reshape([n_timesteps, batch_size, n_features, self.out_dim], ndim=4).sum(axis=2)  #transsition
        emission_potential = feat_emission 	
        if 'use_emb' in self.params and self.params['use_emb']: #'emb_matrix' in self.params: 
            print 'use embeddings!'
            #emb_tv = input_tv
            emission_potential += input_tv #emb_tv 
        self.output_tv = (emission_potential) #, chain_potential)
        return (self.T_,)

    def needed_key(self):
    #return ['emb_dim']
        return self._needed_key_impl(('W'))


class Activation(Chip):
    """ This requires a _activation_fn parameter
    """
    def compute(self, input_tv):
        self.output_tv = self.params[self.kn('activation_fn')](input_tv)
        return tuple()

    def needed_key(self):
        return self._needed_key_impl('activation_fn')


class Linear(Chip):
    def prepend(self, previous_chip):
        self = super(Linear, self).prepend(previous_chip)
        self.N = self._declare_mat('N', self.in_dim, self.out_dim)
        return self

    """ A Linear Chip is a matrix Multiplication
    It requires a U_initializer
    """
    def compute(self, input_tv):
        self.output_tv = T.dot(input_tv, self.N)
        return (self.N,)

    def needed_key(self):
        return self._needed_key_impl('N')

class Bias(Chip):
    def prepend(self, previous_chip):
        self = super(Bias, self).prepend(previous_chip)
        self.b = self._declare_mat('b', self.out_dim) #, is_regularizable=False)
        return self

    """ A Bias Chip adds a vector to the input
    It requires a b_initializer
    """
    def compute(self, input_tv):
        self.output_tv = input_tv + self.b
        return (self.b,)

    def needed_key(self):
        return self._needed_key_impl('b')


class BiasedLinear(Chip):
    def __init__(self, name, params=None):
        super(BiasedLinear, self).__init__(name, params)
        self.params[self.name+"_linear_out_dim"] = params[self.kn('out_dim')]
        self.params[self.name+"_bias_out_dim"] = params[self.kn('out_dim')]
        self.Linear = Linear(name+'_linear', self.params)
        self.Bias = Bias(name+'_bias', self.params)

    def prepend(self, previous_chip):
        self.Bias.prepend(self.Linear.prepend(previous_chip))
        self.in_dim = self.Linear.in_dim
        return self
    """ Composition of Linear and Bias
    It requires a U_initializer and a b_initializer
    """
    def compute(self, input_tv):
        internal_params = list(self.Linear.compute(input_tv))
        internal_params += list(self.Bias.compute(self.Linear.output_tv))
        self.output_tv = self.Bias.output_tv
        return tuple(internal_params)

    def needed_key(self):
        return self.Linear.needed_key() + self.Bias.needed_key()


class LSTM(Chip):
    def prepend(self, previous_chip):
        self = super(LSTM, self).prepend(previous_chip)
        print 'lstm in dim:', self.in_dim, 'out dim:', self.out_dim
        self.go_backwards = self.params[self.kn('go_backwards')]
        self.W = self._declare_mat('W', self.in_dim, 4*self.out_dim)
        self.U = self._declare_mat('U', self.out_dim, 4*self.out_dim)
        self.b = self._declare_mat('b', 4*self.out_dim) #, is_regularizable = False)
        self.p = self._declare_mat('p', 3*self.out_dim)
        return self

    """ This requires W, U and b initializer
    """
    def compute(self, input_tv, mask=None):
        n_steps = input_tv.shape[0]
        if input_tv.ndim == 3:
            n_samples = input_tv.shape[1]
        else:
            n_samples = 1

        def __slice(matrix, row_idx, stride):
            if matrix.ndim == 3:
                return matrix[:, :, row_idx * stride:(row_idx + 1) * stride]
            elif matrix.ndim == 2:
                return matrix[:, row_idx * stride:(row_idx + 1) * stride]
            else:
                return matrix[row_idx*stride: (row_idx+1)*stride]
            #return matrix[row_idx*stride: (row_idx+1)*stride]

        def __step(x_, h_prev, c_prev, U, p):
            """
            x = Transformed and Bias incremented Input  (This is basically a matrix)
                We do the precomputation for efficiency.
            h_prev = previous output of the LSTM (Left output of this function)
            c_prev = previous cell value of the LSTM (Right output of this function)

            This is the vanilla version of the LSTM without peephole connections
            See: Section 2, "LSTM: A Search Space Odyssey", Klaus et. al, ArXiv(2015)
            http://arxiv.org/pdf/1503.04069v1.pdf for details.
            """
            preact = T.dot(h_prev, U) + x_
            i = T.nnet.sigmoid(__slice(preact, 0, self.out_dim) + __slice(p, 0, self.out_dim)*c_prev) # Input gate
            f = T.nnet.sigmoid(__slice(preact, 1, self.out_dim) + __slice(p, 1, self.out_dim)*c_prev) # Forget gate
            z = T.tanh(__slice(preact, 3, self.out_dim)) # block input
            c = f * c_prev + i * z # cell state
            o = T.nnet.sigmoid(__slice(preact, 2, self.out_dim) + __slice(p, 2, self.out_dim) * c) # output gate
            h = o * T.tanh(c)  # block output
            return h, c
        x_in = (T.dot(input_tv, self.W).astype(config.floatX)) + self.b
        rval, _ = theano.scan(__step,
                              sequences=x_in,
                              outputs_info=[T.alloc(np_floatX(0.), self.out_dim),
                                            T.alloc(np_floatX(0.), self.out_dim)],
                              non_sequences=[self.U, self.p],
                              go_backwards=self.go_backwards,
                              name=name_tv(self.name, 'LSTM_layer'),
                              n_steps=n_steps,
                              strict=True,
                              # Force the LSTM to work on the CPU.
                              # mode=theano.compile.mode.get_default_mode().excluding('gpu').including('cpu'),
                              # Don't gc in case this makes code faster.
                              # allow_gc=False
        )
        self.output_tv = reverse(rval[0]) if self.go_backwards else rval[0]
        if self.params.get(self.kn('dropout_rate'), 0.0) != 0.0:
            print 'DROP OUT!!! at circuite', self.name, 'Drop out rate: ', self.params[self.kn('dropout_rate')]
            self.output_tv = _dropout_from_layer(self.params['rng'], self.output_tv, self.params[self.kn('dropout_rate')])
        return (self.W, self.U, self.b, self.p)

    def needed_key(self):
        return self._needed_key_impl('W', 'U', 'b', 'p', 'go_backwards')


class BiLSTM(Chip):
    def __init__(self, name, params=None):
        super(BiLSTM, self).__init__(name, params)
        print 'print bilstm parameters:', self.params
        self.params[self.name+"_forward_go_backwards"] = False
        self.params[self.name+"_backward_go_backwards"] = True
        self.params[self.name+"_forward_out_dim"] = params[self.kn('out_dim')]
        self.params[self.name+"_backward_out_dim"] = params[self.kn('out_dim')]
        self.forward_chip = LSTM(self.name+"_forward", self.params)
        self.backward_chip = LSTM(self.name+"_backward", self.params)
        self.params[self.kn('win')] = 2

    def prepend(self, previous_chip):
        self.forward_chip.prepend(previous_chip)
        self.backward_chip.prepend(previous_chip)
        self.in_dim = self.forward_chip.in_dim
        self.out_dim = self.forward_chip.out_dim + self.backward_chip.out_dim
        #self.out_dim = self.forward_chip.out_dim + self.backward_chip.out_dim
        return self

    """ This requires W, U and b initializer
    """
    def compute(self, input_tv):
        # Before creating the sub LSTM's set the out_dim to half
        # Basically this setting would be used by the sub LSTMs
        internal_params = list(self.forward_chip.compute(input_tv))
        internal_params += list(self.backward_chip.compute(input_tv))
        self.output_tv = T.concatenate([self.forward_chip.output_tv, self.backward_chip.output_tv], axis=1)
        if self.params.get(self.kn('dropout_rate'), 0.0) != 0.0:
            print 'DROP OUT!!! at circuite', self.name, 'Drop out rate: ', self.params[self.kn('dropout_rate')]
            self.output_tv = _dropout_from_layer(self.params['rng'], self.output_tv, self.params[self.kn('dropout_rate')])
        #self.out_dim = self.forward_chip.out_dim + self.backward_chip.out_dim
        return tuple(internal_params)

    def needed_key(self):
        return self.forward_chip.needed_key() + self.backward_chip.needed_key()

class OrderZeroCrf(Chip):
    def prepend(self, previous_chip):
        self = super(OrderZeroCrf, self).prepend(previous_chip)
        return self

    def compute(self, input_tv):
        # The input is a tensor of scores.
        #assert len(input_tv) == 2
        emission_prob = T.nnet.softmax(input_tv)
        assert emission_prob.ndim == 2
        self.gold_y = T.ivector(make_name(self.name, 'gold_y')).astype('int32')
        '''def forward_step(obs2tags, gy, prev_p):
            assert obs2tags.ndim == 1
            p = prev_p + T.log(obs2tags[gy])
            y = T.argmax(obs2tags).astype('int32')
            return p, y
        ## Add if statement!!!!!
        [ps, ys], _ = theano.scan(forward_step,
                sequences=[emission_prob, self.gold_y],
                outputs_info=[theano.shared(0.0), None],
                name='OrderZeroCrf_scan__step',
                strict=True)
        '''
        # The most likely state based on the forward/alpha scores.
        self.output_tv = T.zeros_like(emission_prob[:, 0])
        # This is the score of the gold sequence. We need this.
        self.score = - emission_prob[-1][-1]
        
        return tuple()

class OrderOneCrf(Chip):
    def prepend(self, previous_chip):
        self = super(OrderOneCrf, self).prepend(previous_chip)
        self.chain_potential = self._declare_mat('T', self.out_dim, self.out_dim,  multiplier=1.0/self.out_dim)
        return self

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
    #def out_dim(self, in_dim):
    #    return in_dim

    def _th_logsumexp(self, x, axis=None):
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def compute(self, input_tv):
        # The input is a tensor of scores.
        emission_prob = input_tv
        crf_step = None 
        viterbi_flag = self.params[self.kn('viterbi')]
        
        # shape: (num_tag,), (num_tag,), (num_tag, num_tag)
        def forward_step(obs2tags, prev_result, chain_potentials):
            assert obs2tags.ndim == 1
            assert prev_result.ndim == 1
            f_ = prev_result.dimshuffle(0, 'x') + obs2tags.dimshuffle('x', 0) + chain_potentials
            p = (T.max(f_, axis=0)
                    if viterbi_flag
                    else self._th_logsumexp(f_, axis=0))
            y = T.argmax(f_, axis=0).astype('int32')
            return p, y
        
        # shape: (batch_size, num_tag), (batch_size, num_tag), (num_tag, num_tag)
        def forward_step_batch(obs2tags, prev_result, chain_potentials):
            assert obs2tags.ndim == 2
            assert prev_result.ndim == 2
            # Shape (batch, num_tag, num_tag)
            f_ = prev_result.dimshuffle(0, 1, 'x') + obs2tags.dimshuffle(0, 'x', 1) + chain_potentials.dimshuffle('x', 0, 1)
            p = (T.max(f_, axis=1)
                    if viterbi_flag
                    else self._th_logsumexp(f_, axis=1))
            y = T.argmax(f_, axis=1).astype('int32')
            return p, y
        
        if emission_prob.ndim == 2:
            self.gold_y = T.ivector(make_name(self.name, 'gold_y')).astype('int32')
            # Shape = (sent_len, num_label)
            crf_step = forward_step
        elif emission_prob.ndim == 3:
            self.gold_y = T.imatrix(make_name(self.name, 'gold_y')).astype('int32')
            # Shape = (sent_len, batch_size, num_label)
            crf_step = forward_step_batch
        else:
            raise NotImplementedError
        initial = emission_prob[0] #* T.ones_like(T_))
        # rval is the lattice of forward scores. bp contains backpointers
        [rval, bp], _ = theano.scan(crf_step,
                sequences=[emission_prob[1:]],
                outputs_info=[initial, None],
                non_sequences=[self.chain_potential],
                name='OrderOneCrf_scan_step',
                strict=True)
        # The most likely state based on the forward/alpha scores.
        yn = rval[-1].argmax(axis=-1).astype('int32')
        
        def back_trace(back_pointers, cur_idx):
            return back_pointers[cur_idx]

        def back_trace_batch(back_pointers_mtx, cur_idx_arry):
            return back_pointers_mtx[T.arange(cur_idx_arry.shape[0]), cur_idx_arry]
        
        back_track_func = None
        if emission_prob.ndim == 2:
            back_track_func = back_trace
        elif emission_prob.ndim == 3:
            back_track_func = back_trace_batch
        else:
            raise NotImplementedError
            
        path_tmp, _ = theano.scan(back_track_func,
                sequences=bp,
                outputs_info=yn,
                go_backwards=True,
                name='OrderOneCrf_scan_bkpntr',
                strict=True)
        # The output_tv contains the sequence that has the highest score.
        if emission_prob.ndim == 2:
            self.output_tv = T.concatenate([reverse(path_tmp), yn.dimshuffle('x')])
        elif emission_prob.ndim == 3:
            self.output_tv = T.concatenate([reverse(path_tmp), yn.dimshuffle('x', 0)])
        else:
            raise NotImplementedError
        self._partition_val = self._th_logsumexp(rval[-1], axis=-1).mean()

        def _output_score(obs2tags, pred_y, chain_potentials):
            def _inner_step(o, y, p_, y_, c):
                return ((p_ + c[y_, y] + o[y]), y)
            
            # Shape (batch_size, num_labels), (batch_size,), (batch_size,), (batch_size,), (num_labels, num_labels)
            def _inner_step_batch(o, y, p_, y_, c):
                return ((p_ + c[y_, y] + o[T.arange(y.shape[0]), y]), y)
            
            y0 = None 
            p0 = None
            if obs2tags.ndim == 2:
                y0 = pred_y[0]
                p0 = obs2tags[0][y0]
                _score_step = _inner_step
            elif obs2tags.ndim == 3:
                y0 = pred_y[:, 0]
                p0 = obs2tags[0][T.arange(y0.shape[0]), y0]
                _score_step = _inner_step_batch
            else:
                raise NotImplementedError
            [rval, _], _ = theano.scan(_score_step,
                    sequences=[obs2tags[1:], pred_y[1:]],
                    outputs_info=[p0, y0],
                    non_sequences=[chain_potentials],
                    name='OrderOneCrf_scan_score_step',
                    strict=True)
            return rval[-1].mean()
        self.score = -(_output_score(emission_prob, self.gold_y, self.chain_potential) - self._partition_val)
        return (self.chain_potential,) 

    def needed_key(self):
        return self._needed_key_impl('viterbi')


class Slice(Chip):
    """ This chip simply slices the input along the first dimension
    """
    def prepend(self, previous_chip):
        self.previous_chip = previous_chip
        super(Slice, self).prepend(previous_chip)
        return self

    def compute(self, input_tv):
        self.output_tv = input_tv[self.params[self.kn('begin')]: self.params[self.kn('end')]]
        return tuple()
    
    def __getattr__(self, item):
        """ Inherit all the attributes of the previous chip.
        At present I can only see this functionality being useful
        for the case of the Slice and Regularization chip. Maybe we would move
        this down in case it is found necessary later on, but there is
        chance of abuse.
        """
        try:
            return getattr(self.previous_chip, item)
        except KeyError:
            raise AttributeError(item)

    def needed_key(self):
        return self._needed_key_impl('begin', 'end')


class L2Reg(Chip):
    """ This supposes that the previous chip would have a score attribute.
    And it literally only changes the score attribute by adding the regularization term
    on top of it.
    """
    def prepend(self, previous_chip):
        self.previous_chip = previous_chip
        super(L2Reg, self).prepend(previous_chip)
        return self

    def compute(self, input_tv):
        L2 = T.sum(T.stack([T.sum(self.params[k]*self.params[k])
            for k
            in self.regularizable_variables()]))
        L2.name = tparams_make_name(self.name, 'L2')
        self.score = self.previous_chip.score + self.params[self.kn('reg_weight')] * L2
        return tuple() #(L2,)

    def __getattr__(self, item):
        """ Inherit all the attributes of the previous chip.
        At present I can only see this functionality being useful
        for the case of the Slice and Regularization chip. Maybe we would move
        this down in case it is found necessary later on, but there is
        chance of abuse.
        """
        try:
            return getattr(self.previous_chip, item)
        except KeyError:
            raise AttributeError(item)

    def needed_key(self):
        return self._needed_key_impl('reg_weight')

