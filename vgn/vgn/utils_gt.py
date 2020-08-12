import os

import numpy  as np
import graph_tool as gt
from graph_nets.utils_np import data_dicts_to_graphs_tuple


NODES = "nodes"
EDGES = "edges"
RECEIVERS = "receivers"
SENDERS = "senders"
GLOBALS = "globals"
N_NODE = "n_node"
N_EDGE = "n_edge"

def _get_features(graph_gt, dict_p, obj):
  features = []
  if obj == "nodes":
    map_prop = graph_gt.vp
  elif obj == "edges":
    map_prop = graph_gt.ep
  elif obj == "global":
    map_prop = graph_gt.gp
  else:
    raise ValueError

  for p, tp in dict_p.items():
    if tp == "array":
      if obj == "global":
        ap = map_prop[p]
      else:
        ap = map_prop[p].get_2d_array([0,1]).T
    elif tp == "scalar":
      if obj == "global":
        ap = np.array([map_prop[p]])
      else:
        ap = map_prop[p].get_array().reshape(-1, 1)
    elif tp == "string":
      ap = map_prop[p].get_2d_array([0])[0]
      ap = np.array(ap, dtype="int").reshape(-1, 1)
    else:
      raise ValueError
    features.append(ap)
  if len(features) > 0:
    features = np.concatenate(features, axis=-1)
  else:
    features = None
  return features


def gts_to_graph_tuple(graph_gts,
                       vps,
                       eps,
                       gps,
                       data_type_hint=np.float32):
  data_dicts = []
  try:
    for graph_gt in graph_gts:
      features_v = _get_features(graph_gt, vps, "nodes")
      features_e = _get_features(graph_gt, eps, "edges")
      features_g = _get_features(graph_gt, gps, "global")

      data_dict = gt_to_data_dict(graph_gt, features_v,
                                  features_e, features_g)
      data_dicts.append(data_dict)
  except TypeError:
    raise ValueError("Could not convert some elements of `graph_gts`. "
                     "Did you pass an iterable of networkx instances?")

  return data_dicts_to_graphs_tuple(data_dicts)

def gt_to_data_dict(graph_gt,
                    nodes_data,
                    edges_data,
                    global_data,
                    undirected=True,
                    data_type_hint=np.float32):
  try:
    num_vertices = graph_gt.num_vertices()
  except ValueError:
    raise TypeError("Argument `graph_gt` of wrong type {}".format(
        type(graph_gt)))
  if num_vertices > 0:
    if isinstance(nodes_data, np.ndarray):
      if len(nodes_data) != num_vertices:
        raise ValueError(
            "Either all the nodes should have features, or none of them")

  num_edges = graph_gt.num_edges()
  if num_edges == 0:
    senders = np.zeros(0, dtype=np.int32)
    receivers = np.zeros(0, dtype=np.int32)
  else:
    edges_indices = graph_gt.get_edges()
    senders = edges_indices[:,0]
    receivers = edges_indices[:,1]
    if undirected:
      _senders = senders.copy()
      senders = np.concatenate([senders, receivers], axis=0)
      receivers = np.concatenate([receivers, _senders], axis=0)
    if isinstance(edges_data, np.ndarray):
      if len(edges_data) != num_edges:
        raise ValueError(
            "Either all the edges should have features, or none of them")
      if undirected:
        num_edges *= 2
        edges_data = np.concatenate([edges_data, edges_data], axis=0)

  return {
      NODES: nodes_data,
      EDGES: edges_data,
      RECEIVERS: receivers,
      SENDERS: senders,
      GLOBALS: global_data,
      N_NODE: num_vertices,
      N_EDGE: num_edges,
  }

if __name__ == "__main__":
  paths = ["/home/caio/Documents/PhD/VANET/cologne-data/graphs_with_metrics/10036.gt.xz",
          "/home/caio/Documents/PhD/VANET/cologne-data/graphs_with_metrics/10000.gt.xz"]
  graph_gts = []
  for path in paths:
    graph_gts.append(gt.load_graph(path))
  out = gts_to_graph_tuple(graph_gts, vps={"pos": "array", "speed": "scalar", "dgc": "scalar", "cnw": "scalar"},
                     eps={"weight": "scalar"}, gps={"time": "scalar", "d": "scalar"})


  for G in graph_gts:
    print("==========================================================================")
    features_v = _get_features(G, {"pos": "array", "speed": "scalar", "dgc": "scalar", "cnw": "scalar"}, "nodes")
    features_e = _get_features(G, {"weight": "scalar"}, "edges")
    features_g = _get_features(G, {"time": "scalar", "d": "scalar"}, "global")
    print(features_v)
    print(features_e)
    print(features_g)
  print("==========================================================================")
  print(out)
