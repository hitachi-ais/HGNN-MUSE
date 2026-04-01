import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax


def hypergraph_pe(
    edge_index: torch.Tensor,
    num_nodes: int = None,
    num_edges: int = None,
    k: int = 8,
    edge_weight: torch.Tensor = None,
    eps: float = 1e-12,
    dtype=torch.float32,
    device=None,
):
    if device is None:
        device = edge_index.device
    v_idx, e_idx = edge_index.to(device)

    if num_nodes is None:
        num_nodes = int(v_idx.max()) + 1
    if num_edges is None:
        num_edges = int(e_idx.max()) + 1
    E = v_idx.numel()

    if edge_weight is None:
        w_e = torch.ones(num_edges, dtype=dtype, device=device)
    else:
        edge_weight = edge_weight.to(device, dtype)
        if edge_weight.numel() == E:
            w_e = torch.zeros(num_edges, dtype=dtype, device=device)
            w_e.index_add_(0, e_idx, edge_weight)
        else:
            w_e = torch.zeros(num_edges, dtype=dtype, device=device)
            w_e[: edge_weight.numel()] = edge_weight

    H = torch.sparse_coo_tensor(
        edge_index,
        torch.ones(E, dtype=dtype, device=device),
        (num_nodes, num_edges),
        dtype=dtype,
        device=device,
    )
    edge_card = torch.sparse.sum(H, dim=0).to_dense()
    inv_sqrt_edge_card = torch.rsqrt(edge_card + eps)
    w_per_incidence = w_e[e_idx]
    H_w = torch.sparse_coo_tensor(
        edge_index, w_per_incidence, (num_nodes, num_edges), dtype=dtype, device=device
    )
    vertex_deg = torch.sparse.sum(H_w, dim=1).to_dense()
    inv_sqrt_deg = torch.rsqrt(vertex_deg + eps)
    values_tilde = (
        inv_sqrt_deg[v_idx] * torch.sqrt(w_e[e_idx]) * inv_sqrt_edge_card[e_idx]
    )
    H_tilde = torch.sparse_coo_tensor(
        edge_index, values_tilde, (num_nodes, num_edges), dtype=dtype, device=device
    )
    A = torch.sparse.mm(H_tilde, H_tilde.transpose(0, 1))
    L = torch.eye(num_nodes, dtype=dtype, device=device) - A.to_dense()
    jitter = 1e-6 * torch.eye(num_nodes, dtype=dtype, device=device)
    L = L + jitter

    eig_val, eig_vec = torch.linalg.eigh(L)
    zero_mask = eig_val < 1e-7
    zero_mult = int(zero_mask.sum().item())

    if zero_mult == 0:
        pe = torch.zeros(num_nodes, k, device=device, dtype=dtype)
        if k > 0:
            pe[:, 0] = 1.0
        return pe

    start = zero_mult
    pe = eig_vec[:, start : start + k]

    if pe.numel() > 0:
        sgn = torch.sign(pe[pe.abs().argmax(dim=0), range(pe.size(1))])
        pe = pe * sgn
        pe = pe / (pe.std(0, keepdim=True) + 1e-6)

    if pe.shape[1] < k:
        pe = F.pad(pe, (0, k - pe.shape[1]), "constant", 0.0)
    return pe

        
class Set2SetMultiHeadAttention(MessagePassing):
    def __init__(self, in_dim, out_dim, heads=1, dropout=0.0, negative_slope=0.2):
        super(Set2SetMultiHeadAttention, self).__init__(node_dim=0, aggr="add")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.hidden_dim = out_dim // heads
        assert self.hidden_dim * heads == out_dim, "out_dim must be divisible by heads"

        self.q = torch.nn.Parameter(torch.Tensor(1, self.heads, self.hidden_dim))
        self.W_k = torch.nn.Linear(in_dim, self.heads * self.hidden_dim, bias=False)
        self.W_v = torch.nn.Linear(in_dim, self.heads * self.hidden_dim, bias=False)

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=negative_slope)

        self.dropout = torch.nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.q)
        torch.nn.init.xavier_uniform_(self.W_k.weight)
        torch.nn.init.xavier_uniform_(self.W_v.weight)

    def forward(self, x, edge_index, size=None):
        k = self.W_k(x).view(-1, self.heads, self.hidden_dim)
        v = self.W_v(x).view(-1, self.heads, self.hidden_dim)

        a = (k * self.q).sum(dim=-1)
        a = self.leaky_relu(a)
        x = self.propagate(edge_index, v=v, a=a, size=size)
        x = x + self.q
        x = x.view(-1, self.heads * self.hidden_dim)
        return x

    def message(self, v_j, a_j, index, ptr):
        attn_weight = softmax(a_j, index, ptr)
        attn_weight = self.dropout(attn_weight)
        return v_j * attn_weight.unsqueeze(-1)


class AllSetTransformerLayer(torch.nn.Module):
    def __init__(
        self,
        dim,
        nhead=4,
        dropout=0.2,
    ):
        super(AllSetTransformerLayer, self).__init__()
        self.node2edge_att = Set2SetMultiHeadAttention(dim, dim, heads=nhead)
        self.node2edge_layer_norm1 = torch.nn.LayerNorm(dim)
        self.node2edge_ffn = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dim, dim),
            torch.nn.ReLU(),
        )
        self.node2edge_layer_norm2 = torch.nn.LayerNorm(dim)
        self.edge2node_att = Set2SetMultiHeadAttention(dim, dim, heads=nhead)
        self.edge2node_layer_norm1 = torch.nn.LayerNorm(dim)
        self.edge2node_ffn = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dim, dim),
            torch.nn.ReLU(),
        )
        self.edge2node_layer_norm2 = torch.nn.LayerNorm(dim)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index, size=None):
        x = self.dropout(x)
        x = self.node2edge_layer_norm1(self.node2edge_att(x, edge_index, size=size))
        x = self.node2edge_layer_norm2(x + self.node2edge_ffn(x))

        r_edge_index = edge_index.flip(0)
        x = self.dropout(x)
        x = self.edge2node_layer_norm1(
            self.edge2node_att(x, r_edge_index, size=(size[1], size[0]))
        )
        x = self.edge2node_layer_norm2(x + self.edge2node_ffn(x))
        return x


class AllSetTransformer_RL(torch.nn.Module):
    def __init__(
        self,
        dim,
        nhead=8,
        hgnn_layer_num=4,
        transformer_layer_num=4,
        hgnn_dropout=0.2,
        k=16,
    ):
        super(AllSetTransformer_RL, self).__init__()
        self.dim = dim
        self.hgnn_layer_num = hgnn_layer_num
        self.transformer_layer_num = transformer_layer_num
        self.k = k

        self.mus_hgnn_layers = torch.nn.ModuleList(
            AllSetTransformerLayer(dim, nhead, hgnn_dropout)
            for _ in range(hgnn_layer_num)
        )
        self.mcs_hgnn_layers = torch.nn.ModuleList(
            AllSetTransformerLayer(dim, nhead, hgnn_dropout)
            for _ in range(hgnn_layer_num)
        )

        self.embed = torch.nn.Linear(k * 2, dim, bias=False)
        self.mix = torch.nn.ModuleList(
            torch.nn.Linear(dim * 2, dim, bias=False) for _ in range(hgnn_layer_num)
        )

        transformer_dim = dim * (hgnn_layer_num + 1)
        ffn_dim = transformer_dim * 4
        transformer_layer = torch.nn.TransformerDecoderLayer(
            transformer_dim,
            nhead,
            dim_feedforward=ffn_dim,
            dropout=0.1,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.value_transformer_decoder = torch.nn.TransformerDecoder(
            transformer_layer, transformer_layer_num
        )
        self.policy_transformer_decoder = torch.nn.TransformerDecoder(
            transformer_layer, transformer_layer_num
        )
        value_head_dim = transformer_dim * 2 + 2  # +2 for shrink/grow mode
        policy_head_dim = transformer_dim + 2
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(policy_head_dim, transformer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(transformer_dim, 1),
        )

        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(value_head_dim, transformer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(transformer_dim, 1),
        )

    def forward(
        self,
        data: HeteroData,
        target_subset_bin: torch.Tensor,
        mode: str = "shrink",
    ):
        x = self.encode_hypergraph(data)

        policy_logits, value = self.decode_to_policy_value(x, target_subset_bin, mode)

        return policy_logits, value

    def encode_hypergraph(self, data: HeteroData):
        mus_edge = data["clause", "in", "mus"].edge_index
        mcs_edge = data["clause", "in", "mcs"].edge_index
        mus_x = hypergraph_pe(
            mus_edge, data["clause"].num_nodes, data["mus"].num_nodes, self.k
        )
        mcs_x = hypergraph_pe(
            mcs_edge, data["clause"].num_nodes, data["mcs"].num_nodes, self.k
        )

        x = torch.cat([mus_x, mcs_x], dim=1)
        x = self.embed(x)

        x_list = [x]
        for mus_hgnn, mcs_hgnn, mix in zip(
            self.mus_hgnn_layers,
            self.mcs_hgnn_layers,
            self.mix,
        ):
            x_mus = mus_hgnn(
                x, mus_edge, size=(data["clause"].num_nodes, data["mus"].num_nodes)
            )
            x_mcs = mcs_hgnn(
                x, mcs_edge, size=(data["clause"].num_nodes, data["mcs"].num_nodes)
            )
            x = torch.cat([x_mus, x_mcs], dim=1)
            x = mix(x)
            x_list.append(x)

        x = torch.cat(x_list, dim=1)
        return x

    def decode_to_policy_value(
        self, x: torch.Tensor, target_subset_bin: torch.Tensor, mode: str = "shrink"
    ):
        x = x[None, :, :].repeat(target_subset_bin.shape[0], 1, 1)

        value = self.get_value(x, target_subset_bin, mode)

        policy_logits = self.get_policy(x, target_subset_bin, mode)

        return policy_logits, value

    def get_value(self, x, target_subset_bin, mode):
        sub_x = self.value_transformer_decoder(
            x,
            x,
            tgt_key_padding_mask=~target_subset_bin,
            tgt_is_causal=False,
            memory_is_causal=False,
        )
        mean_sub_x = (sub_x * target_subset_bin[:, :, None].float()).sum(dim=1) / (
            target_subset_bin[:, :, None].float().sum(dim=1) + 1e-8
        )
        droped_sub_x = sub_x.clone()
        droped_sub_x[~target_subset_bin] = -1e9  # Mask out the dropped nodes
        max_sub_x = torch.max(droped_sub_x, dim=1)[0]
        global_x = torch.cat([mean_sub_x, max_sub_x], dim=1)
        mode_x = torch.zeros(
            (global_x.shape[0], 2), device=global_x.device, dtype=global_x.dtype
        )
        if mode == "shrink":
            mode_x[:, 0] = 1.0  # shrink mode
        elif mode == "grow":
            mode_x[:, 1] = 1.0
        global_x = torch.cat([global_x, mode_x], dim=1)
        value = self.value_head(global_x).squeeze(1)
        return value

    def get_policy(self, x, target_subset_bin, mode):
        sub_x = self.policy_transformer_decoder(
            x,
            x,
            tgt_key_padding_mask=~target_subset_bin,
            tgt_is_causal=False,
            memory_is_causal=False,
        )
        if mode == "shrink":
            sub_mode_x = torch.cat(
                [
                    sub_x,
                    torch.ones_like(sub_x[:, :, :1]),
                    torch.zeros_like(sub_x[:, :, :1]),
                ],
                dim=2,
            )
        elif mode == "grow":
            sub_mode_x = torch.cat(
                [
                    sub_x,
                    torch.zeros_like(sub_x[:, :, :1]),
                    torch.ones_like(sub_x[:, :, :1]),
                ],
                dim=2,
            )
        policy_logits = self.policy_head(sub_mode_x).squeeze(2)
        policy_logits[~target_subset_bin] = -1e9
        policy_logits = torch.cat(
            [policy_logits, torch.zeros_like(policy_logits[:, :1])], dim=1
        )  # Add a dummy column for the "finish" action

        return policy_logits