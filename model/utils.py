import torch
from torch_geometric.utils import scatter_
def to_dense_adj(edge_index, batch=None, edge_attr=None):
    r"""Converts batched sparse adjacency matrices given by edge indices and
    edge attributes to a single dense batched adjacency matrix.
    Args:
        edge_index (LongTensor): The edge indices.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
    :rtype: :class:`Tensor`
    """
    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)
    batch_size = batch[-1].item() + 1
    one = batch.new_ones(batch.size(0))
    ## TODO 出现错误RuntimeError: expand(torch.cuda.LongTensor{[1, 1, 224]}, size=[224]): the number of sizes provided (1) must be greater or equal to the number of dimensions in the tensor (3)
    # # print("type one : " + str(type(one)))
    # # print("one : " + str(one))
    # print("one shape : " + str(one.shape))
    # # print("type batch : " + str(type(batch)))
    # # print("batch : " + str(batch))
    # print("batch : " + str(batch.shape))
    ## TODO 将tensor进行unsqueeze之后，解决上面的错误，但是会出现一个新错误RuntimeError: invalid argument 0: Tensors must have same number of dimensions: got 3 and 1 at /pytorch/aten/src/THC/generic/THCTensorMath.cu:62
    ## TODO 因此，比较理想的方式是找到data tensor进行squeeze，但是找不到这个tensor在哪
    ## TODO 所以，放弃掉了这个函数，使用torch_gemetic中的to_adj()函数
    # one = one.unsqueeze(0)
    # one = one.unsqueeze(0)
    # print("one shape : " + str(one.shape))
    # batch = batch.unsqueeze(0)
    # batch = batch.unsqueeze(0)
    # print("batch : " + str(batch.shape))
    num_nodes = scatter_('add', one, batch, batch_size)
    # num_nodes = torch.scatter(reduce='add')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])
    max_num_nodes = num_nodes.max().item()

    size = [batch_size, max_num_nodes, max_num_nodes]
    size = size if edge_attr is None else size + list(edge_attr.size())[1:]
    dtype = torch.float if edge_attr is None else edge_attr.dtype
    adj = torch.zeros(size, dtype=dtype, device=edge_index.device)

    edge_index_0 = batch[edge_index[0]].view(1, -1)
    edge_index_1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    edge_index_2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    if edge_attr is None:
        adj[edge_index_0, edge_index_1, edge_index_2] = 1
    else:
        adj[edge_index_0, edge_index_1, edge_index_2] = edge_attr

    return adj


import torch
from torch_scatter import scatter_add


def to_dense_batch(x, batch, fill_value=0):
    r"""Given a sparse batch of node features
    :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}` (with
    :math:`N_i` indicating the number of nodes in graph :math:`i`), creates a
    dense node feature tensor
    :math:`\mathbf{X} \in \mathbb{R}^{B \times N_{\max} \times F}` (with
    :math:`N_{\max} = \max_i^B N_i`).
    In addition, a second tensor holding
    :math:`[N_1, \ldots, N_B] \in \mathbb{N}^B` is returned.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        fill_value (float, optional): The value for invalid entries in the
            resulting dense output tensor. (default: :obj:`0`)

    :rtype: (:class:`Tensor`, :class:`LongTensor`)
    """
    num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)], dim=0)

    index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_nodes[batch]) + (batch * max_num_nodes)

    size = [batch_size * max_num_nodes] + list(x.size())[1:]
    batch_x = x.new_full(size, fill_value)
    batch_x[index] = x
    size = [batch_size, max_num_nodes] + list(x.size())[1:]
    batch_x = batch_x.view(size)

    return batch_x, num_nodes
