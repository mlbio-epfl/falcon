import gmatch4py as gm
import networkx as nx
import numpy as np

def make_adjacency_matrix(M):
    n1, n2 = M.shape
    N = n1 + n2
    adj_M = np.zeros((N, N)).astype(np.int32)
    adj_M[:n1, -n2:] = M
    adj_M[n1:, :n1] = M.T
    return adj_M

def compute_ged(gt_pred, gt_graph):

    G_pred = nx.from_numpy_array(gt_pred)
    G_actual = nx.from_numpy_array(gt_graph)

    ged = gm.GraphEditDistance(1, 1, 1, 1)  # all edit costs are equal to 1
    result = ged.compare([G_pred, G_actual], None)
    return result[0,1]
