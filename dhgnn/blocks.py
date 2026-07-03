from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedResidualAggregator(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(dim + 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        group_size: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, height, width, channels = x.shape
        if height % group_size != 0 or width % group_size != 0:
            raise ValueError(
                f"Feature grid [{height}, {width}] must be divisible by {group_size}."
            )

        out_height, out_width = height // group_size, width // group_size

        x_group = x.view(
            batch_size,
            out_height,
            group_size,
            out_width,
            group_size,
            channels,
        )
        x_group = x_group.permute(0, 1, 3, 2, 4, 5).contiguous()
        x_group = x_group.view(batch_size, out_height, out_width, group_size * group_size, channels)

        pos_group = pos.view(
            batch_size,
            out_height,
            group_size,
            out_width,
            group_size,
            2,
        )
        pos_group = pos_group.permute(0, 1, 3, 2, 4, 5).contiguous()
        pos_group = pos_group.view(batch_size, out_height, out_width, group_size * group_size, 2)

        center = pos_group.mean(dim=3, keepdim=True)
        rel_pos = pos_group - center
        score_input = torch.cat([x_group, rel_pos], dim=-1)

        alpha = F.softmax(self.score_mlp(score_input).squeeze(-1), dim=-1).unsqueeze(-1)
        mean_feat = (alpha * x_group).sum(dim=3)
        residual_stat = (alpha * (x_group - mean_feat.unsqueeze(3)) ** 2).sum(dim=3)

        out_x = self.out_proj(torch.cat([mean_feat, residual_stat], dim=-1))
        out_pos = (alpha * pos_group).sum(dim=3)

        yy, xx = torch.meshgrid(
            torch.arange(height, device=x.device),
            torch.arange(width, device=x.device),
            indexing="ij",
        )
        parent_y = yy // group_size
        parent_x = xx // group_size
        parent_map = torch.stack([parent_y, parent_x], dim=-1)
        parent_map = parent_map.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        return out_x, out_pos, parent_map


class SpatialRelationLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        rel_dim: int = 32,
        adjacency_mode: str = "dynamic",
        dynamic_topk: int = 4,
        use_edge_state: bool = True,
        edge_dim: int = 32,
    ):
        super().__init__()
        if adjacency_mode not in {"fixed", "dynamic"}:
            raise ValueError(f"Unsupported adjacency_mode: {adjacency_mode}")
        if dynamic_topk <= 0:
            raise ValueError(f"dynamic_topk must be positive, got {dynamic_topk}")

        self.dim = dim
        self.rel_dim = rel_dim
        self.adjacency_mode = adjacency_mode
        self.dynamic_topk = dynamic_topk
        self.use_edge_state = use_edge_state
        self.edge_dim = edge_dim

        self.offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        self.num_relations = len(self.offsets)
        self.rel_embed = nn.Embedding(self.num_relations, rel_dim)

        self.candidate_mlp = nn.Sequential(
            nn.Linear(dim * 2 + 2 + rel_dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

        if use_edge_state:
            self.edge_update = nn.Sequential(
                nn.Linear(dim * 2 + 2 + rel_dim, dim),
                nn.GELU(),
                nn.Linear(dim, edge_dim),
            )
            gate_in_dim = dim * 2 + 2 + edge_dim
            msg_in_dim = dim + edge_dim
        else:
            self.edge_update = None
            gate_in_dim = dim * 2 + 2 + rel_dim
            msg_in_dim = dim + rel_dim

        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.msg_proj = nn.Sequential(
            nn.Linear(msg_in_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    @staticmethod
    def _shift_with_mask(x: torch.Tensor, dy: int, dx: int):
        batch_size, height, width, _ = x.shape
        shifted = torch.zeros_like(x)
        mask = torch.zeros(batch_size, height, width, 1, device=x.device, dtype=x.dtype)

        src_y0 = max(0, -dy)
        src_y1 = min(height, height - dy) if dy >= 0 else height
        dst_y0 = max(0, dy)
        dst_y1 = min(height, height + dy) if dy < 0 else height

        src_x0 = max(0, -dx)
        src_x1 = min(width, width - dx) if dx >= 0 else width
        dst_x0 = max(0, dx)
        dst_x1 = min(width, width + dx) if dx < 0 else width

        shifted[:, dst_y0:dst_y1, dst_x0:dst_x1, :] = x[:, src_y0:src_y1, src_x0:src_x1, :]
        mask[:, dst_y0:dst_y1, dst_x0:dst_x1, :] = 1.0
        return shifted, mask

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        candidate_msgs = []
        candidate_gates = []
        candidate_valids = []
        candidate_scores = []

        batch_size, height, width, _ = x.shape

        for rel_id, (dy, dx) in enumerate(self.offsets):
            neigh_x, valid_mask = self._shift_with_mask(x, dy, dx)
            neigh_pos, _ = self._shift_with_mask(pos, dy, dx)
            rel_pos = neigh_pos - pos
            rel_e = self.rel_embed.weight[rel_id].view(1, 1, 1, -1)
            rel_e = rel_e.expand(batch_size, height, width, -1)

            base_in = torch.cat([x, neigh_x, rel_pos, rel_e], dim=-1)
            cand_score = self.candidate_mlp(base_in)

            if self.use_edge_state:
                edge_feat = self.edge_update(base_in)
                gate_in = torch.cat([x, neigh_x, rel_pos, edge_feat], dim=-1)
                msg_in = torch.cat([neigh_x, edge_feat], dim=-1)
            else:
                gate_in = base_in
                msg_in = torch.cat([neigh_x, rel_e], dim=-1)

            gate = torch.sigmoid(self.gate_mlp(gate_in)) * valid_mask
            msg = self.msg_proj(msg_in) * gate

            candidate_msgs.append(msg)
            candidate_gates.append(gate)
            candidate_valids.append(valid_mask)
            candidate_scores.append(cand_score)

        msgs = torch.stack(candidate_msgs, dim=0)
        gates = torch.stack(candidate_gates, dim=0)
        valids = torch.stack(candidate_valids, dim=0)
        scores = torch.stack(candidate_scores, dim=0)

        if self.adjacency_mode == "dynamic":
            scores = scores.masked_fill(valids <= 0, float("-inf"))
            topk = min(self.dynamic_topk, self.num_relations)
            topk_idx = torch.topk(scores.squeeze(-1), k=topk, dim=0).indices
            select_mask = torch.zeros_like(scores.squeeze(-1), dtype=x.dtype)
            select_mask.scatter_(0, topk_idx, 1.0)
            select_mask = select_mask.unsqueeze(-1) * valids
            gates = gates * select_mask
            msgs = msgs * select_mask

        msg_sum = msgs.sum(dim=0)
        gate_sum = gates.sum(dim=0).clamp_min(1e-6)
        neigh_agg = msg_sum / gate_sum

        out = self.out_proj(torch.cat([x, neigh_agg], dim=-1))
        return x + out


class ParentChildRelationLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        rel_dim: int = 32,
        use_edge_state: bool = True,
        edge_dim: int = 32,
    ):
        super().__init__()
        self.dim = dim
        self.rel_dim = rel_dim
        self.use_edge_state = use_edge_state
        self.edge_dim = edge_dim

        self.rel_embed = nn.Embedding(2, rel_dim)

        if use_edge_state:
            self.edge_update_cp = nn.Sequential(
                nn.Linear(dim * 2 + 2 + rel_dim, dim),
                nn.GELU(),
                nn.Linear(dim, edge_dim),
            )
            self.edge_update_pc = nn.Sequential(
                nn.Linear(dim * 2 + 2 + rel_dim, dim),
                nn.GELU(),
                nn.Linear(dim, edge_dim),
            )
            gate_dim = dim * 2 + 2 + edge_dim
            msg_dim = dim + edge_dim
        else:
            self.edge_update_cp = None
            self.edge_update_pc = None
            gate_dim = dim * 2 + 2 + rel_dim
            msg_dim = dim + rel_dim

        self.child_to_parent_gate = nn.Sequential(
            nn.Linear(gate_dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.parent_to_child_gate = nn.Sequential(
            nn.Linear(gate_dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.child_msg = nn.Sequential(
            nn.Linear(msg_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.parent_msg = nn.Sequential(
            nn.Linear(msg_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.parent_upd = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.child_upd = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        child_x: torch.Tensor,
        child_pos: torch.Tensor,
        parent_x: torch.Tensor,
        parent_pos: torch.Tensor,
        parent_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, child_h, child_w, channels = child_x.shape
        _, parent_h, parent_w, _ = parent_x.shape

        py = parent_map[..., 0]
        px = parent_map[..., 1]
        batch_idx = torch.arange(batch_size, device=child_x.device)
        batch_idx = batch_idx.view(batch_size, 1, 1).expand(batch_size, child_h, child_w)

        linked_parent_x = parent_x[batch_idx, py, px]
        linked_parent_pos = parent_pos[batch_idx, py, px]

        rel_cp = child_pos - linked_parent_pos
        rel_cp_e = self.rel_embed.weight[0].view(1, 1, 1, -1).expand(batch_size, child_h, child_w, -1)
        base_cp_in = torch.cat([child_x, linked_parent_x, rel_cp, rel_cp_e], dim=-1)
        if self.use_edge_state:
            edge_cp = self.edge_update_cp(base_cp_in)
            gate_cp_in = torch.cat([child_x, linked_parent_x, rel_cp, edge_cp], dim=-1)
            msg_cp_in = torch.cat([child_x, edge_cp], dim=-1)
        else:
            gate_cp_in = base_cp_in
            msg_cp_in = torch.cat([child_x, rel_cp_e], dim=-1)

        gate_cp = torch.sigmoid(self.child_to_parent_gate(gate_cp_in))
        msg_cp = self.child_msg(msg_cp_in) * gate_cp

        parent_msg_acc = torch.zeros_like(parent_x)
        parent_w_acc = torch.zeros(
            batch_size,
            parent_h,
            parent_w,
            1,
            device=child_x.device,
            dtype=child_x.dtype,
        )

        for batch_index in range(batch_size):
            parent_msg_acc[batch_index].index_put_(
                (py[batch_index].reshape(-1), px[batch_index].reshape(-1)),
                msg_cp[batch_index].reshape(-1, channels),
                accumulate=True,
            )
            parent_w_acc[batch_index].index_put_(
                (py[batch_index].reshape(-1), px[batch_index].reshape(-1)),
                gate_cp[batch_index].reshape(-1, 1),
                accumulate=True,
            )

        parent_agg = parent_msg_acc / parent_w_acc.clamp_min(1e-6)
        parent_x_new = parent_x + self.parent_upd(torch.cat([parent_x, parent_agg], dim=-1))

        linked_parent_x_new = parent_x_new[batch_idx, py, px]
        linked_parent_pos_new = parent_pos[batch_idx, py, px]
        rel_pc = linked_parent_pos_new - child_pos
        rel_pc_e = self.rel_embed.weight[1].view(1, 1, 1, -1).expand(batch_size, child_h, child_w, -1)
        base_pc_in = torch.cat([linked_parent_x_new, child_x, rel_pc, rel_pc_e], dim=-1)
        if self.use_edge_state:
            edge_pc = self.edge_update_pc(base_pc_in)
            gate_pc_in = torch.cat([linked_parent_x_new, child_x, rel_pc, edge_pc], dim=-1)
            msg_pc_in = torch.cat([linked_parent_x_new, edge_pc], dim=-1)
        else:
            gate_pc_in = base_pc_in
            msg_pc_in = torch.cat([linked_parent_x_new, rel_pc_e], dim=-1)

        gate_pc = torch.sigmoid(self.parent_to_child_gate(gate_pc_in))
        msg_pc = self.parent_msg(msg_pc_in) * gate_pc

        child_x_new = child_x + self.child_upd(torch.cat([child_x, msg_pc], dim=-1))
        return child_x_new, parent_x_new
