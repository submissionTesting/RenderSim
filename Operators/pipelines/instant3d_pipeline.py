from typing import Tuple

from operators.sampling_operator import UniformSamplerOperator
from operators.encoding_operator import HashEncodingOperator
from operators.computation_operator import MLPOperator
from operators.blending_operator import RGBRendererOperator
from operators.optimization_operator import OptimizationOperator
from utils.operator_graph import OperatorGraph


class FeedForwardReadMapper(OptimizationOperator):
    def __init__(self, dim, bitwidth: int = 16, graph=None, backward: bool = False):
        super().__init__(dim, bitwidth, graph, backward=backward)
        self.op_type = "FRM"  # Feed-Forward Read Mapper

    def get_tensors(self):
        B, N = self.dim[:2]
        return B * N * 3, 0, B * N * 3

    def get_num_ops(self):
        B, N = self.dim[:2]
        return B * N * 4


class BackpropUpdateMerger(OptimizationOperator):
    def __init__(self, dim, bitwidth: int = 16, graph=None, backward: bool = False):
        super().__init__(dim, bitwidth, graph, backward=backward)
        self.op_type = "BUM"  # Back-Propagation Update Merger

    def get_tensors(self):
        B, N = self.dim[:2]
        return B * N * 3, 0, B * N * 3

    def get_num_ops(self):
        B, N = self.dim[:2]
        return B * N * 6


def build_instant3d_training_pipeline(dim: Tuple[int, int]):
    """Instant-3D training pipeline with asymmetric read/update mappers."""

    g = OperatorGraph()
    sampler = UniformSamplerOperator(dim, graph=g)

    ff_read = FeedForwardReadMapper(dim, graph=g)
    sampler.add_child(ff_read)

    enc = HashEncodingOperator(dim, graph=g)
    ff_read.add_child(enc)

    in_dim = enc.num_levels * enc.features_per_level
    mlp = MLPOperator(dim, in_dim=in_dim, num_layers=3, layer_width=64, out_dim=4, graph=g)
    enc.add_child(mlp)

    render = RGBRendererOperator(dim, graph=g)
    mlp.add_child(render)

    # Backward: render -> mlp(B) -> enc + update merger(B)
    render_b = RGBRendererOperator(dim, graph=g, backward=True)
    mlp_b = MLPOperator(dim, in_dim=in_dim, num_layers=3, layer_width=64, out_dim=4, graph=g, backward=True)
    bp_merge = BackpropUpdateMerger(dim, graph=g, backward=True)  # BUM is part of backward pass

    render.add_child(render_b)
    render_b.add_child(mlp_b)
    mlp_b.add_child(bp_merge)
    bp_merge.add_child(enc)  # route gradient/update to encoding

    return g


