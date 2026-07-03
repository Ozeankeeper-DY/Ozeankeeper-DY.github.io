import torch
import torch.nn as nn

from .blocks import ParentChildRelationLayer, SpatialRelationLayer, WeightedResidualAggregator
from .heads import build_classifier_head, build_readout


class DHGNNClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        hidden_dim: int = 768,
        num_levels: int = 3,
        graph_depth: int = 1,
        readout_type: str = "mean",
        adjacency_mode: str = "dynamic",
        dynamic_topk: int = 4,
        use_edge_state: bool = True,
        include_input_readout: bool = False,
        rel_dim: int = 32,
        edge_dim: int = 32,
        node_type_dim: int = 3,
        use_node_type: bool = True,
    ):
        super().__init__()
        if num_classes <= 1:
            raise ValueError(f"num_classes must be greater than 1, got {num_classes}")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_levels not in (1, 2, 3):
            raise ValueError(f"num_levels must be 1, 2, or 3, got {num_levels}")
        if graph_depth < 1:
            raise ValueError(f"graph_depth must be positive, got {graph_depth}")
        if adjacency_mode not in ("fixed", "dynamic"):
            raise ValueError(f"adjacency_mode must be 'fixed' or 'dynamic', got {adjacency_mode}")
        if dynamic_topk <= 0:
            raise ValueError(f"dynamic_topk must be positive, got {dynamic_topk}")
        if node_type_dim < 3:
            raise ValueError(f"node_type_dim must be at least 3, got {node_type_dim}")

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.graph_depth = graph_depth
        self.include_input_readout = include_input_readout
        self.use_node_type = use_node_type

        self.in_proj = nn.Linear(feature_dim, hidden_dim)
        self.node_type_embed = nn.Embedding(node_type_dim, hidden_dim)
        self.node_type_proj = nn.ModuleList(
            [
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
            ]
        )

        self.agg12 = WeightedResidualAggregator(hidden_dim)
        self.agg23 = WeightedResidualAggregator(hidden_dim)

        self.spatial1 = self._build_spatial_stack(
            graph_depth,
            hidden_dim,
            rel_dim=rel_dim,
            adjacency_mode=adjacency_mode,
            dynamic_topk=dynamic_topk,
            use_edge_state=use_edge_state,
            edge_dim=edge_dim,
        )
        self.spatial2 = self._build_spatial_stack(
            graph_depth,
            hidden_dim,
            rel_dim=rel_dim,
            adjacency_mode=adjacency_mode,
            dynamic_topk=dynamic_topk,
            use_edge_state=use_edge_state,
            edge_dim=edge_dim,
        )
        self.spatial3 = self._build_spatial_stack(
            graph_depth,
            hidden_dim,
            rel_dim=rel_dim,
            adjacency_mode=adjacency_mode,
            dynamic_topk=dynamic_topk,
            use_edge_state=use_edge_state,
            edge_dim=edge_dim,
        )

        self.pc12 = ParentChildRelationLayer(
            hidden_dim,
            rel_dim=rel_dim,
            use_edge_state=use_edge_state,
            edge_dim=edge_dim,
        )
        self.pc23 = ParentChildRelationLayer(
            hidden_dim,
            rel_dim=rel_dim,
            use_edge_state=use_edge_state,
            edge_dim=edge_dim,
        )

        self.readout = build_readout(hidden_dim=hidden_dim, readout_type=readout_type)
        in_dim_multiplier = self.readout.out_dim_multiplier
        if include_input_readout:
            in_dim_multiplier += 1
        self.cls_head = build_classifier_head(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            in_dim_multiplier=in_dim_multiplier,
        )

    @staticmethod
    def _build_spatial_stack(
        graph_depth: int,
        hidden_dim: int,
        rel_dim: int,
        adjacency_mode: str,
        dynamic_topk: int,
        use_edge_state: bool,
        edge_dim: int,
    ) -> nn.ModuleList:
        return nn.ModuleList(
            [
                SpatialRelationLayer(
                    hidden_dim,
                    rel_dim=rel_dim,
                    adjacency_mode=adjacency_mode,
                    dynamic_topk=dynamic_topk,
                    use_edge_state=use_edge_state,
                    edge_dim=edge_dim,
                )
                for _ in range(graph_depth)
            ]
        )

    @staticmethod
    def _apply_spatial_stack(
        layers: nn.ModuleList,
        x: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        for layer in layers:
            x = layer(x, pos)
        return x

    @staticmethod
    def build_grid_pos(batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, steps=height, device=device),
            torch.linspace(0.0, 1.0, steps=width, device=device),
            indexing="ij",
        )
        pos = torch.stack([yy, xx], dim=-1)
        return pos.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    def _inject_node_type(self, x: torch.Tensor, node_type_id: int) -> torch.Tensor:
        if not self.use_node_type:
            return x
        type_vec = self.node_type_embed.weight[node_type_id].view(1, 1, 1, -1)
        return self.node_type_proj[node_type_id](x) + type_vec

    def _validate_features(self, features: torch.Tensor) -> None:
        if features.dim() != 4:
            raise ValueError(
                "DHGNNClassifier expects precomputed features with shape [B, H, W, C]."
            )
        _, height, width, channels = features.shape
        if channels != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, got input channel dim {channels}."
            )
        if self.num_levels >= 2 and (height % 2 != 0 or width % 2 != 0):
            raise ValueError(
                "num_levels>=2 requires feature grid height and width divisible by 2."
            )
        if self.num_levels >= 3 and (height % 4 != 0 or width % 4 != 0):
            raise ValueError(
                "num_levels=3 requires feature grid height and width divisible by 4."
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self._validate_features(features)
        x1 = features.float()
        batch_size, height, width, _ = x1.shape

        x1 = self.in_proj(x1)
        p1 = self.build_grid_pos(batch_size, height, width, x1.device)
        x1 = self._inject_node_type(x1, 0)
        input_g = x1.mean(dim=(1, 2)) if self.include_input_readout else None

        x1 = self._apply_spatial_stack(self.spatial1, x1, p1)

        if self.num_levels >= 2:
            x2, p2, parent12 = self.agg12(x1, p1, group_size=2)
            x2 = self._inject_node_type(x2, 1)
            x2 = self._apply_spatial_stack(self.spatial2, x2, p2)
            x1, x2 = self.pc12(x1, p1, x2, p2, parent12)
        else:
            x2 = x1

        if self.num_levels >= 3:
            x3, p3, parent23 = self.agg23(x2, p2, group_size=2)
            x3 = self._inject_node_type(x3, 2)
            x3 = self._apply_spatial_stack(self.spatial3, x3, p3)
            x2, x3 = self.pc23(x2, p2, x3, p3, parent23)
        else:
            x3 = x2

        graph_feature = self.readout(x1, x2, x3)
        if input_g is not None:
            graph_feature = torch.cat([graph_feature, input_g], dim=-1)
        return self.cls_head(graph_feature)
