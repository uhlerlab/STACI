##adapted to pytorch from https://github.com/tkipf/gae and https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py ##
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


# def dropout_sparse(x, keep_prob, num_nonzero_elems):
#     """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
#     """
#     noise_shape = [num_nonzero_elems]
#     random_tensor = keep_prob
#     random_tensor += tf.random_uniform(noise_shape)
#     dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
#     pre_out = tf.sparse_retain(x, dropout_mask)
#     return pre_out * (1./keep_prob)


class GraphConvolution(nn.Module):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, dropout=0., act=F.relu,bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = act
        self.weight=nn.parameter.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.parameter.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Create a weight variable with Glorot & Bengio (AISTATS 2010)
        initialization.
        """
        init_range = np.sqrt(6.0 / (self.input_dim + self.output_dim))
        nn.init.uniform_(self.weight, a=-init_range, b=init_range)
        
        if self.bias is not None:
            nn.init.uniform_(self.bias,a=-init_range, b=init_range)    
        
    def forward(self,input,adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            output= output + self.bias
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> '+ str(self.output_dim) + ')'
    
# class GraphConvolutionSparse(Layer):
#     """Graph convolution layer for sparse inputs."""
#     def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=tf.nn.relu, **kwargs):
#         super(GraphConvolutionSparse, self).__init__(**kwargs)
#         with tf.variable_scope(self.name + '_vars'):
#             self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
#         self.dropout = dropout
#         self.adj = adj
#         self.act = act
#         self.issparse = True
#         self.features_nonzero = features_nonzero

#     def _call(self, inputs):
#         x = inputs
#         x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
#         x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
#         x = tf.sparse_tensor_dense_matmul(self.adj, x)
#         outputs = self.act(x)
#         return outputs


class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""
    def __init__(self, dropout, act=lambda x: x):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, inputs):
        inputs=F.dropout(inputs, self.dropout, training=self.training)
        x=torch.mm(inputs, inputs.t())
        outputs = self.act(x)
        return outputs

class FC(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0., act=F.leaky_relu, batchnorm=False,bias=False):
#     def __init__(self, input_dim, output_dim, dropout=0., act=F.leaky_relu, batchnorm=True,bias=True):
        super(FC, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = act
        self.batchnorm=batchnorm
        self.linearlayer=nn.Linear(input_dim,output_dim,bias=bias)
        if batchnorm:
            self.batchnormlayer=nn.BatchNorm1d(output_dim)
    
    def forward(self,input):
        input = F.dropout(input, self.dropout, self.training)
        output = self.linearlayer(input)
        if self.batchnorm:
            output=self.batchnormlayer(output)
        output = self.act(output)
        return output
    
class FC_elementwise(nn.Module):
    def __init__(self, input_dim,piSize, dropout=0., act=F.leaky_relu, batchnorm=False,bias=False):
        super(FC_elementwise, self).__init__()
        self.input_dim = input_dim
        self.piSize = piSize
        self.dropout = dropout
        self.act = act
        self.batchnorm=batchnorm
        if batchnorm:
            self.batchnormlayer=nn.BatchNorm1d(input_dim)
        self.weight=nn.parameter.Parameter(torch.FloatTensor(piSize))
        if bias:
            self.bias = nn.parameter.Parameter(torch.FloatTensor(piSize))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Create a weight variable with Glorot & Bengio (AISTATS 2010)
        initialization.
        """
        init_range = np.sqrt(6.0 / (self.input_dim + self.input_dim)) #output dim same as input
        nn.init.uniform_(self.weight, a=-init_range, b=init_range)
        if self.bias is not None:
            nn.init.uniform_(self.bias,a=-init_range, b=init_range)
    
    def forward(self,input):
        input = F.dropout(input, self.dropout, self.training)
#         output = self.linearlayer(input)
        output=input*self.weight
        if self.bias is not None:
            output= output + self.bias
        if self.batchnorm:
            output=self.batchnormlayer(output)
        output = self.act(output)
        return output