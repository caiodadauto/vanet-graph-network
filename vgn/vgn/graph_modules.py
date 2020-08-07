import sonnet as snt
import tensorflow as tf
from graph_nets import modules
from graph_nets import utils_tf

from networks import FullyLayer, RecurrentLayer

_NLAYERS = 3
_LATENT_SIZE = 16



def make_recurrent_model(size=_LATENT_SIZE, recurrent=snt.LSTM, activation=tf.nn.leaky_relu, norm=snt.LayerNorm, dropout_rate=0.25)
    return RecurrentLayer(size, recurrent, activation, norm, dropout_rate)


def make_mlp_model(size=_LATENT_SIZE, n_layers=_NLAYERS, activation=tf.nn.leaky_relu, norm=snt.LayerNorm, dropout_rate=.2):
    return FullyLayer(size, n_layers, activation, norm, dropout_rate)


class FullyGraphIndependent(snt.Module):
    def __init__(self, name="FullyGraphIndependent"):
        super(FullyGraphIndependent, self).__init__(name=name)
        self._network = modules.GraphIndependent(
            edge_model_fn=make_mlp_model,
            node_model_fn=make_mlp_model,
            global_model_fn=make_mlp_model)

    def __call__(self, inputs, **kwargs):
        return self._network(inputs, **kwargs)


class NodeRecurrentAggregator(snt.Module):

  def __init__(self,
               bias_size=[3 * _LATENT_SIZE],
               reducer=tf.math.unsorted_segment_sum,
               name="neighborhood_aggregator"):
    super(NodeRecurrentAggregator, self).__init__(name=name)
    self._bias = tf.Variable("bias_aggregator", shape=bias_size)
    self._reducer = reducer

  def __cal__(self, graph):
    # If the number of nodes are known at graph construction time (based on the
    # shape) then use that value to make the model compatible with XLA/TPU.
    if graph.nodes is not None and graph.nodes.shape.as_list()[0] is not None:
      num_nodes = graph.nodes.shape.as_list()[0]
    else:
      num_nodes = tf.reduce_sum(graph.n_node)
    reduced_features = self._reducer(broadcast_sender_nodes_to_edges(graph), graph.receivers, num_nodes)
    return reduced_features + self._bias


class RecurrentGraphNetwork(snt.Module):
    def __init__(self,
                 recurrent_model_fn=make_recurrent_model,
                 reducer=tf.unsorted_segment_sum,
                 name="RecurrentGraphNetwork"):
        super(RecurrentGraphNetwork, self).__init__(name=name)

        self._edge_state = None
        self._node_state = None
        self._global_state = None

        self._edge_block = blocks.RecurrentEdgeBlock(
            edge_recurrent_model_fn=recurrent_model_fn,
            use_edges=True,
            use_receiver_nodes=True,
            use_sender_nodes=True,
            use_globals=True
        )
        self._node_block = blocks.RecurrentNodeBlock(
            node_recurrent_model_fn=recurrent_model_fn,
            use_received_edges=True,
            use_sent_edges=False,
            use_nodes=True,
            use_globals=True,
            aggregator_model_fn=NodeRecurrentAggregator
        )
        self._global_block = blocks.RecurrentGlobalBlock(
            global_recurrent_model_fn=recurrent_model_fn,
            use_received_edges=True,
            use_sent_edges=False,
            use_nodes=True,
            use_globals=True
        )

    def set_initial_states(self, edge_batch_size,  node_batch_size, global_batch):
        self._edge_state = self._edge_block.reset_state(edge_batch_size)
        self._node_state = self._node_block.reset_state(node_batch_size)
        self._global_state = self._global_block.reset_state(global_batch_size)

    def __call__(self, graph, **kwargs):
        edge_graph, self._edge_state = self._edge_block(graph, **kwargs)
        node_graph, self._node_state = self._node_block(edge_graph, **kwargs)
        global_graph, self._global_state = self._global_block(node_graph, **kwargs)


class EncodeProcessDecode(snt.Module):

    def __init__(self,
                 edge_output_size=None,
                 node_output_size=None,
                 global_output_size=None,
                 name="EncodeProcessDecode"):
        super(EncodeProcessDecode, self).__init__(name=name)
    self._encoder = FullyGraphIndependent()
    self._core = MLPGraphNetwork()
    self._decoder = FullyGraphIndependent()
    if edge_output_size is None:
        edge_fn = None
    else:
        edge_fn = lambda: snt.Linear(edge_output_size, name="edge_output")
    if node_output_size is None:
        node_fn = None
    else:
        node_fn = lambda: snt.Linear(node_output_size, name="node_output")
    if global_output_size is None:
        global_fn = None
    else:
        global_fn = lambda: snt.Linear(global_output_size, name="global_output")
    self._output_transform = modules.GraphIndependent(
        edge_fn, node_fn, global_fn)

    def __call__(self, input_op, num_processing_steps):
        latent = self._encoder(input_op)
    latent0 = latent
    output_ops = []
    for _ in range(num_processing_steps):
        core_input = utils_tf.concat([latent0, latent], axis=1)
        latent = self._core(core_input)
        decoded_op = self._decoder(latent)
        output_ops.append(self._output_transform(decoded_op))
    return output_ops
