import numpy as np
import tensorflow as tf
from initiators import *


_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    From GraphSAGE GitHub: https://github.com/williamleif/GraphSAGE
    The authors of GraphSAGE were inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class GcnNeighborSampler(Layer):
    """
    Sample the neighbors according to probabilities
    """
    def __init__(self, adj_info, **kwargs):
        super(GcnNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info
        self.range = (len(adj_info[0]) + 1) // 2

    def _call(self, inputs):
        """

        :param inputs: Two parts: 1) ids of sampled nodes for a batch,
        2) # of neighbors to sample for each node in the batch

        Workflow:
        Input the indices of the batch,
        1) pick the corresponding lines of adjacency profile (each line is the contacts between a node and its
        K upstream and K downstream nodes in chromatin fiber)
        2) sample M neighbors for each node in the batch according to the adjacency profile, but tf.categorical()
        only return int values in [0, 2K]
        3) add the node id then minus K to get the real id in the bulk data of each neighbor
        :return: A tensor of shape [batch_size, num_neighbors],
        each position is an integer indicating the id of a chosen node
        """
        ids, num_neighbors = inputs
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)  # Find the lines corresponding to current ids
        sample_res = tf.random.categorical(tf.log(adj_lists), num_neighbors, dtype=tf.int32)  # tf.log can handle log(0)
        # The function above return a tensor of ids, but only from 0 to len(adj_info[0])
        # For example, if we choose 500 nodes up- and down- stream (length = 1001), it will only
        # return values between 0 and 1000
        # To get real index, add original id then minus 500
        new_ids = tf.convert_to_tensor([idx - self.range for idx in ids], dtype=tf.int32)
        sample_res = sample_res + tf.expand_dims(new_ids, 1)
        return sample_res


class GCNAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    Same matmul parameters are used self vector and neighbor vectors.
    """
    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(GCNAggregator, self).__init__(**kwargs)

        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['weights'] = glorot([neigh_input_dim, output_dim], name='neigh_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        features, sample_res = inputs
        sampled_data = tf.nn.embedding_lookup(features, sample_res)
        neigh_vecs = tf.reduce_mean(sampled_data, axis=2)  # Here we need to multiply sum_j(A_ij)

        # [nodes] x [out_dim]
        output = tf.matmul(neigh_vecs, self.vars['weights'])

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


if __name__ == '__main__':
    all_names = ['ATAC_seq', 'Brd4', 'Brg1', 'Cbx3', 'Cbx5', 'cMyc', 'CTCF', 'Daxx', 'Esrrb', 'Ezh2',
                 'Gro_seq_minus', 'Gro_seq_pos', 'H2AZ', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K9me2', 'H3K9me3',
                 'H3K27ac', 'H3K27me3',
                 'H3K36me3', 'H3K79me2', 'H4ac', 'Hira', 'Klf4', 'Klf5', 'Med1', 'Med12', 'Nanog', 'Oct4',
                 'p300', 'Rad21', 'Sox2', 'Sp1', 'Suz12', 'Tbp', 'Top2a', 'YY1', 'Ino80', 'Ring1b']

    with tf.Session() as sess:
        ids, num_samples = [400, 401, 402, 403, 404, 405], 5
        adj = np.zeros((1000, 21))
        for i in range(1000):
            adj[i, 10] = 1

        adj_info = tf.convert_to_tensor(adj)
        adj_lists = tf.nn.embedding_lookup(adj_info, ids)  # Find the lines corresponding to current ids
        print(adj_lists)
        sample_res = tf.random.categorical(tf.log(adj_lists), num_samples, dtype=tf.int32)
        print(sample_res)
        # The function above return a tensor of ids, but only from 0 to len(adj_info[0])
        # For example, if we choose 500 nodes up- and down- stream (length = 1001), it will only
        # return values between 0 and 1000
        # To get real index, add original id then minus 500
        new_ids = tf.convert_to_tensor([idx - 10 for idx in ids], dtype=tf.int32)
        print(new_ids)
        sample_res = sample_res + tf.expand_dims(new_ids, 1)
        print(sample_res)
        print(sess.run(sample_res))


