##adapted to pytorch from https://github.com/tkipf/gae ##
import numpy as np
import scipy.sparse as sp
import torch


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor.
        source: https://github.com/zfjsail/gae-pytorch/blob/master/gae/utils.py"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    print(np.sum(rowsum==0))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def preprocess_graph_sharp(adj):
    """from paper Symmetric Graph Convolutional Autoencoder for Unsupervised Graph Representation Learning"""
    adj = sp.coo_matrix(adj)
    adj_ = 2*sp.eye(adj.shape[0])-adj
    rowsum = np.array(adj.sum(1))+2
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def mask_nodes_edges(nNodes,testNodeSize=0.1,valNodeSize=0.05,seed=3):
    # randomly select nodes; mask all corresponding rows and columns in loss functions
    np.random.seed(seed)
    num_test=int(round(testNodeSize*nNodes))
    num_val=int(round(valNodeSize*nNodes))
    all_nodes_idx = np.arange(nNodes)
    np.random.shuffle(all_nodes_idx)
    test_nodes_idx = all_nodes_idx[:num_test]
    val_nodes_idx = all_nodes_idx[num_test:(num_val + num_test)]
    train_nodes_idx=all_nodes_idx[(num_val + num_test):]
    
    return torch.tensor(train_nodes_idx),torch.tensor(val_nodes_idx),torch.tensor(test_nodes_idx)
    

def mask_test_edges(adj,crossVal,ncrossVal=3,testSize=0.1,valSize=0.05):
    # Function to build test set with 10% positive links
    # change: using fixed seed for randomized split; added cross validation
    # change: adj does not have diagonal elements

    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] *testSize))
    num_val = int(np.floor(edges.shape[0] *0.05))
    
    def getOneSet(test_edges,val_edges,train_edges):
        def ismember(a, b, tol=5):
            rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
            return np.any(rows_close)

        test_edges_false = []
        while len(test_edges_false) < len(test_edges):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], edges_all):
                continue
            if test_edges_false:
                if ismember([idx_j, idx_i], np.array(test_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(test_edges_false)):
                    continue
            test_edges_false.append([idx_i, idx_j])

        val_edges_false = []
        while len(val_edges_false)< len(val_edges)-1:
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], train_edges):
                continue
            if ismember([idx_j, idx_i], train_edges):
                continue
            if ismember([idx_i, idx_j], val_edges):
                continue
            if ismember([idx_j, idx_i], val_edges):
                continue
            if val_edges_false:
                if ismember([idx_j, idx_i], np.array(val_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(val_edges_false)):
                    continue
            val_edges_false.append([idx_i, idx_j])

#         assert ~ismember(test_edges_false, edges_all)
#         assert ~ismember(val_edges_false, edges_all)
#         assert ~ismember(val_edges, train_edges)
#         assert ~ismember(test_edges, train_edges)
#         assert ~ismember(val_edges, test_edges)

        data = np.ones(train_edges.shape[0])

        # Re-build adj matrix
        adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
        adj_train = adj_train + adj_train.T

        return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false
    
    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    test_edge_idx = all_edge_idx[:num_test]
    test_edges = edges[test_edge_idx]
    if not crossVal:
        val_edge_idx = all_edge_idx[num_test:(num_val + num_test)]
        val_edges = edges[val_edge_idx]
        train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
        return [getOneSet(test_edges,val_edges,train_edges)]
    else:
        maxcrossVal=np.floor(1/valSize)
        if ncrossVal>maxcrossVal:
            print('given the validation size, at most ', maxcrossVal,' validations is allowed')
            ncrossVal=maxcrossVal
        res=[0]*ncrossVal
        for i in range(ncrossVal):
            val_edge_idx = all_edge_idx[(num_test+i*num_val):(num_test+(i+1)*num_val)]
            val_edges = edges[val_edge_idx]
            train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
            res[i]=getOneSet(test_edges,val_edges,train_edges)
        return res

