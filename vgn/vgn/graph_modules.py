import sonnet as snt
import tensorflow as tf
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_tf

from networks import MaskedMLP, MaskedRecurrent


_NLAYERS = 3
_LATENT_SIZE = 16


GraphStates = collections.namedtuple("GraphStates", graphs.GRAPH_FEATURE_FIELDS)


GraphMasks = collections.namedtuple("GraphMasks", graphs.GRAPH_FEATURE_FIELDS)


def make_recurrent_model(size=_LATENT_SIZE, recurrent=snt.LSTM, activation=tf.nn.leaky_relu, norm=snt.LayerNorm, dropout_rate=0.25)
    return MaskedRecurrent(size, recurrent, activation, norm, dropout_rate)


def make_mlp_model(size=_LATENT_SIZE, n_layers=_NLAYERS, activation=tf.nn.leaky_relu, norm=snt.LayerNorm, dropout_rate=.2):
    return MaskedMLP(size, n_layers, activation, norm, dropout_rate)


class MaskedGraphIndependent(snt.Module):

    def __init__(self,
                 edge_model_fn=None,
                 node_model_fn=None,
                 global_model_fn=None,
                 name="masked_graph_independent"):
      super(MaskedGraphIndependent, self).__init__(name=name)

      if edge_model_fn is None:
          self._edge_model = lambda x: x
      else:
          self._edge_model = edge_model_fn
      if node_model_fn is None:
          self._node_model = lambda x: x
      else:
          self._node_model = node_model_fn
      if global_model_fn is None:
          self._global_model = lambda x: x
      else:
          self._global_model = global_model_fn

    def __call__(self, graph, graph_mask, edge_kw={}, node_kw={}, global_kw={}):
      return graph.replace(
          edges=self._edge_model(graph.edges, graph_mask.edges, **edge_kw),
          nodes=self._node_model(graph.nodes, graph_mask.nodes, **node_kw),
          globals=self._global_model(graph.globals, graph_mask.globals, **global_kw))


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


class RollGraphNetwork(snt.Module):

    def __init__(self,
                 recurrent_model_fn=make_recurrent_model,
                 reducer=tf.unsorted_segment_sum,
                 name="roll_graph_network"):
        super(RollGraphNetwork, self).__init__(name=name)

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

    def get_initial_states(self, edge_batch_size,  node_batch_size, global_batch):
        edge_state = self._edge_block.reset_state(edge_batch_size)
        node_state = self._node_block.reset_state(node_batch_size)
        global_state = self._global_block.reset_state(global_batch_size)
        return GraphStates(edge_state, node_state, global_state)

    def __call__(self, graph, graph_state, graph_mask, edge_kw={}, node_kw={}, global_kw={}):
        num_of_times = graph.nodes.shape[1]
        for t in range(num_of_times):
            graph_t = graphs.GraphsTuple(
                tf.squeeze(graph.nodes[:, t, :]),
                tf.squeeze(graph.edges[:, t, :]),
                tf.squeeze(graph.globals[:, t, :]))
            graph_mask_t = GraphMasks(
                tf.squeeze(graph_mask.nodes[:, t, :]),
                tf.squeeze(graph_mask.edges[:, t, :]),
                tf.squeeze(graph_mask.globals[:, t, :]))

            graph_output, edge_state = self._edge_block(
                graph_t, graph_state.edges, graph_mask_t.edges, **edge_kw)
            graph_output, node_state = self._node_block(
                graph_output, graph_state.nodes, graph_mask_t.nodes, **node_kw)
            graph_output, global_state = self._global_block(
                graph_output, graph_state.globals, graph_mask_t.globals, **global_kw)

            graph_state = GraphStates(edge_state, node_state, global_state)
        return graph_output, graph_state


class EncodeProcessDecode(snt.Module):

    def __init__(self,
                 edge_output_size=None,
                 node_output_size=None,
                 global_output_size=None,
                 name="EncodeProcessDecode"):
        super(EncodeProcessDecode, self).__init__(name=name)
        if edge_output_size is None:
            edge_fn = None
        else:
            edge_fn = make_mlp_model(size=edge_output_size, activation=tf.nn.relu)
        if node_output_size is None:
            node_fn = None
        else:
            node_fn = make_mlp_model(size=node_output_size, activation=tf.nn.relu)
        if global_output_size is None:
            global_fn = None
        else:
            global_fn = make_mlp_model(size=global_output_size, activation=tf.nn.relu)
        self._encoder = FullyGraphIndependent()
        self._core = RecurrentGraphNetwork()
        self._decoder = modules.GraphIndependent(edge_fn, node_fn, global_fn)

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
