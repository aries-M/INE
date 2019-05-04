import pdb
import tensorflow as tf

import its.models as models
import its.layers as layers
from its.aggregators import  ITSAggregator
    # , MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator

flags = tf.app.flags
FLAGS = flags.FLAGS


class ITS(models.GeneralizedModel):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, num_classes,
            placeholders, features, instabilities, adj, degrees,
            layer_infos, concat=True, aggregator_type="its", 
            model_size="small", sigmoid_loss=False, identity_dim=0,
                **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
        '''

        models.GeneralizedModel.__init__(self, **kwargs)
        """
        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        """
        if aggregator_type == "its":
            self.aggregator_cls = ITSAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        # get info from placeholders...
        self.inputs1 = placeholders["batch"]
        self.model_size = model_size
        self.adj_info = adj
        if identity_dim > 0:
           self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
           self.embeds = None
        if features is None: 
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        else:
            self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.instabilities = instabilities  
        self.degrees = degrees
        self.concat = concat
        self.num_classes = num_classes
        self.sigmoid_loss = sigmoid_loss
        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.ins_dim = instabilities.shape[1]
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def sample(self, inputs, layer_infos, batch_size=None):
        """ Sample neighbors to be the supportive fields for multi-layer convolutions.

        Args:
            inputs: batch inputs
            batch_size: the number of inputs (different for batch inputs and negative samples).
        """
        # print("type of inputs:")
        # print(type(inputs))
        # print("value of inputs:")
        # print(inputs)
        if batch_size is None:
            batch_size = self.batch_size
        samples = [inputs]
       
        # print("samples:")
        # print(type(samples))
        # print(samples)
        
        # size of convolution support at each layer per node
        support_size = 1
        support_sizes = [support_size]
        for k in range(len(layer_infos)):
            t = len(layer_infos) - k - 1
            support_size *= layer_infos[t].num_samples
            sampler = layer_infos[t].neigh_sampler
            node = sampler((samples[k], layer_infos[t].num_samples))
            samples.append(tf.reshape(node, [support_size * batch_size,]))
            support_sizes.append(support_size)
        
        # self.sample_print = tf.constant()
        return samples, support_sizes
    
    def aggregate(self, samples, input_features, input_instability, dims, num_samples, support_sizes, batch_size=None,
            aggregators=None, name=None, concat=False, model_size="small"):
        """ At each layer, aggregate hidden representations of neighbors to compute the hidden representations 
            at next layer.
        Args:
            samples: a list of samples of variable hops away for convolving at each layer of the
                network. Length is the number of layers + 1. Each is a vector of node indices.
            input_features: the input features for each sample of various hops away.
            dims: a list of dimensions of the hidden representations from the input layer to the
                final layer. Length is the number of layers + 1.
            num_samples: list of number of samples for each layer.
            support_sizes: the number of nodes to gather information from for each layer.
            batch_size: the number of inputs (different for batch inputs and negative samples).
        Returns:
            The hidden representation at the final layer for all nodes in batch
        """

        if batch_size is None:
            batch_size = self.batch_size

        # length: number of layers + 1
        hidden = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in samples]
        instability = [tf.nn.embedding_lookup(input_instability, node_samples) for node_samples in samples]
        # print("hidden:",hidden)
        new_agg = aggregators is None
        if new_agg:
            aggregators = []
        for layer in range(len(num_samples)):
            if new_agg:
                dim_mult = 2 if concat and (layer != 0) else 1
                # aggregator at current layer
                if layer == len(num_samples) - 1:
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], act=lambda x : x,
                            dropout=self.placeholders['dropout'], 
                            name=name, concat=concat, model_size=model_size)
                else:
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1],
                            dropout=self.placeholders['dropout'], 
                            name=name, concat=concat, model_size=model_size)
                aggregators.append(aggregator)
            else:
                aggregator = aggregators[layer]
            # hidden representation at current layer for all support nodes that are various hops away
            next_hidden = []
            # as layer increases, the number of support nodes needed decreases
            # the length of num_samples is the number of layers
            for hop in range(len(num_samples) - layer):
                dim_mult = 2 if concat and (layer != 0) else 1
                neigh_dims = [batch_size * support_sizes[hop], 
                              num_samples[len(num_samples) - hop - 1], 
                              dim_mult*dims[layer]]
                neigh_ins_dims = [batch_size * support_sizes[hop], 
                              num_samples[len(num_samples) - hop - 1], 
                              self.ins_dim]
                # print("shape of hop and hop+1")
                # print(hidden[hop].shape,hidden[hop+1].shape)
                h = aggregator((hidden[hop],                             
                                tf.reshape(hidden[hop + 1], neigh_dims), 
                                instability[hop], 
                                tf.reshape(instability[hop + 1], neigh_ins_dims)))
                
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0], aggregators

    def build(self):
        samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos)
        self.samples1 = samples1
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        self.outputs1, self.aggregators = self.aggregate(samples1, [self.features], [self.instabilities],\
             self.dims, num_samples, support_sizes1, concat=self.concat, model_size=self.model_size)
        
        
        dim_mult = 2 if self.concat else 1

        
        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)

        dim_mult = 2 if self.concat else 1
        self.node_pred = layers.Dense(dim_mult*self.dims[-1], self.num_classes, 
                dropout=self.placeholders['dropout'],
                act=lambda x : x)
        # TF graph management
        self.node_preds = self.node_pred(self.outputs1)

        self._loss()
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        self.preds = self.predict()

    def _loss(self):
        # Weight decay loss
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
       
        # classification loss
        if self.sigmoid_loss:
            self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.placeholders['labels']))
        else:
            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.placeholders['labels']))

        tf.summary.scalar('loss', self.loss)

    def predict(self):
        if self.sigmoid_loss:
            return tf.nn.sigmoid(self.node_preds)
        else:
            return tf.nn.softmax(self.node_preds)
