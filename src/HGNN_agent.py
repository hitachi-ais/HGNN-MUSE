import torch
from torch_geometric.data import HeteroData


def create_hypergraph(num_clauses, found_mus, found_mss):
    src_clause_index = []
    dst_mus_index = []
    for mus_i, mus in enumerate(found_mus):
        src_clause_index += mus
        dst_mus_index += [mus_i] * len(mus)

    clause_in_mus = torch.tensor(
        [src_clause_index, dst_mus_index], dtype=torch.int64
    )

    src_clause_index = []
    dst_mcs_index = []
    for mss_i, mss in enumerate(found_mss):
        mcs = [c for c in range(num_clauses) if c not in mss]
        src_clause_index += mcs
        dst_mcs_index += [mss_i] * len(mcs)

    clause_in_mcs = torch.tensor(
        [src_clause_index, dst_mcs_index], dtype=torch.int64
    )

    data = HeteroData(
        {
            "clause": {"num_nodes": int(num_clauses)},
            "mus": {"num_nodes": len(found_mus)},
            "mcs": {"num_nodes": len(found_mss)},
            ("clause", "in", "mus"): {"edge_index": clause_in_mus},
            ("clause", "in", "mcs"): {"edge_index": clause_in_mcs},
        }
    )
    return data


class HGNN_agent:
    def __init__(self, model, all_clauses, detect_change=False, device="cpu"):
        self.model = model
        self.all_clauses = all_clauses
        self.n_clauses = len(all_clauses)
        self.device = device
        self.cachable = type(all_clauses[0]) is list and all(
            type(c) is int for c in all_clauses[0]
        )
        self.detect_change = detect_change

        self.MUS_set = set()
        self.MSS_set = set()
        if self.cachable:
            self.clause_set = self.listlist2tupleset(all_clauses)
        else:
            self.clause_set = None
        self.x = None

    def reset(self, all_clauses):
        self.all_clauses = all_clauses
        self.n_clauses = len(all_clauses)
        self.MUS_set = set()
        self.MSS_set = set()
        self.clause_set = self.listlist2tupleset(all_clauses)
        self.x = None

    def listlist2tupleset(self, list_list):
        tuple_set = set([tuple(sorted(l)) for l in list_list])
        return tuple_set

    @torch.no_grad()
    def __call__(
        self, all_clauses, subset_id, MUS_list, MSS_list, mode, avoided_actions=[]
    ):
        MUS_set = self.listlist2tupleset(MUS_list)
        MSS_set = self.listlist2tupleset(MSS_list)
        if self.cachable and self.detect_change:
            clause_set = self.listlist2tupleset(all_clauses)
            if clause_set != self.clause_set:
                self.reset(all_clauses)
        if self.x is None or MUS_set != self.MUS_set or MSS_set != self.MSS_set:
            data = create_hypergraph(self.n_clauses, MUS_list, MSS_list)
            data = data.to(self.device)
            x = self.model.encode_hypergraph(data)
            self.MUS_set = MUS_set
            self.MSS_set = MSS_set
            self.x = x
        else:
            x = self.x
        if mode == "shrink":
            target_subset_bin = torch.tensor(
                [[(clause_id in subset_id) for clause_id in range(len(all_clauses))]]
            )
        elif mode == "grow":
            target_subset_bin = torch.tensor(
                [
                    [
                        (clause_id not in subset_id)
                        for clause_id in range(len(all_clauses))
                    ]
                ]
            )
        target_subset_bin = target_subset_bin.to(self.device)
        action_logits, value = self.model.decode_to_policy_value(
            x, target_subset_bin, mode=mode
        )

        action_logit = action_logits[0]
        if avoided_actions:
            action_logit[avoided_actions] = -1e9

        dist = torch.distributions.Categorical(logits=action_logit)
        action = dist.sample().item()
        p_action = dist.probs.cpu().tolist()

        return action, p_action, value[0].item()