from typing import Tuple

from operators.sampling_operator import FrustumCullingProjectionOperator
from operators.blending_operator import SortingOperator, GaussianAlphaBlendOperator
from operators.optimization_operator import OptimizationOperator
from utils.operator_graph import OperatorGraph


class TileMergingOperator(OptimizationOperator):
    def __init__(self, dim, bitwidth: int = 16, graph=None, backward: bool = False):
        super().__init__(dim, bitwidth, graph, backward=backward)
        self.op_type = "TileMerging"

    def get_tensors(self):
        B, N = self.dim[:2]
        return B * N * 2, 0, B * N * 2

    def get_num_ops(self):
        B, N = self.dim[:2]
        return B * N * 8


class FeatureComputeOperator(OptimizationOperator):
    def __init__(self, dim, bitwidth: int = 16, graph=None, backward: bool = False):
        super().__init__(dim, bitwidth, graph, backward=backward)
        self.op_type = "FeatureCompute"

    def get_tensors(self):
        B, N = self.dim[:2]
        return B * N * 4, 0, B * N * 3

    def get_num_ops(self):
        B, N = self.dim[:2]
        return B * N * 12


class GradientComputeOperator(OptimizationOperator):
    def __init__(self, dim, bitwidth: int = 16, graph=None, backward: bool = True):
        # Default to backward semantics for gradient compute
        super().__init__(dim, bitwidth, graph, backward=backward)
        self.op_type = "GradientCompute"

    def get_tensors(self):
        B, N = self.dim[:2]
        # dL/dpixel + intermediate cache -> dL/dsample
        return B * 3, B * N * 2, B * N * 3

    def get_num_ops(self):
        B, N = self.dim[:2]
        return B * N * 16


class GradientPruneOperator(OptimizationOperator):
    def __init__(self, dim, bitwidth: int = 16, graph=None, backward: bool = False):
        super().__init__(dim, bitwidth, graph, backward=backward)
        self.op_type = "GradientPruning"

    def get_tensors(self):
        B, N = self.dim[:2]
        return B * N * 3, 0, B * N * 3

    def get_num_ops(self):
        B, N = self.dim[:2]
        return B * N * 4


class RearrangementOperator(OptimizationOperator):
    def __init__(self, dim, bitwidth: int = 16, graph=None, backward: bool = True):
        super().__init__(dim, bitwidth, graph, backward=backward)
        self.op_type = "Rearrangement"

    def get_tensors(self):
        B, N = self.dim[:2]
        return B * N * 3, 0, B * N * 3

    def get_num_ops(self):
        B, N = self.dim[:2]
        # Light-weight request reordering and bucketing
        import numpy as np
        return B * N + int(np.log2(max(N, 2)))


def build_gsarch_training_pipeline(dim: Tuple[int, int]):
    """GSArch training pipeline with forward/backward blending-focused stages."""

    B, N = dim
    G = B * N
    g = OperatorGraph()

    sample = FrustumCullingProjectionOperator((G,), image_wh=(1280, 720), graph=g)
    sort = SortingOperator(dim, sort_by="depth", order="ascending", graph=g)
    feat = FeatureComputeOperator(dim, graph=g)
    tile_merge = TileMergingOperator(dim, graph=g)
    blend = GaussianAlphaBlendOperator(dim, graph=g)

    # Forward
    sample.add_child(sort)
    sort.add_child(feat)
    feat.add_child(tile_merge)
    tile_merge.add_child(blend)

    # Backward chain (fieldCompute(B) after blending(B))
    blend_b = GaussianAlphaBlendOperator(dim, graph=g, backward=True)
    gradc = GradientComputeOperator(dim, graph=g, backward=True)
    gradp = GradientPruneOperator(dim, graph=g, backward=True)
    rearr = RearrangementOperator(dim, graph=g, backward=True)
    tile_merge_b = TileMergingOperator(dim, graph=g, backward=True)

    blend.add_child(blend_b)
    blend_b.add_child(gradc)
    gradc.add_child(gradp)
    gradp.add_child(rearr)
    rearr.add_child(tile_merge_b)
    tile_merge_b.add_child(sort)  # feed back to sorting for gradient accumulation buckets

    return g


