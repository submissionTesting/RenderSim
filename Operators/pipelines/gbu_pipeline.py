from typing import Tuple

from operators.sampling_operator import FrustumCullingProjectionOperator
from operators.blending_operator import GaussianAlphaBlendOperator, SortingOperator
from operators.optimization_operator import OptimizationOperator
from utils.operator_graph import OperatorGraph


class RowProcessingOperator(OptimizationOperator):
    def __init__(self, dim, bitwidth: int = 16, graph=None, backward: bool = False):
        super().__init__(dim, bitwidth, graph, backward=backward)
        self.op_type = "RowProcessing"

    def get_tensors(self):
        B, N = self.dim[:2]
        input_a = B * N * 5
        input_b = 0
        output = B * N * 5
        return input_a, input_b, output

    def get_num_ops(self):
        B, N = self.dim[:2]
        return B * N * 16


class RowGenerationOperator(OptimizationOperator):
    def __init__(self, dim, bitwidth: int = 16, graph=None, backward: bool = False):
        super().__init__(dim, bitwidth, graph, backward=backward)
        self.op_type = "RowGeneration"

    def get_tensors(self):
        B, N = self.dim[:2]
        input_a = B * N * 5
        input_b = 0
        output = B * N * 5
        return input_a, input_b, output

    def get_num_ops(self):
        B, N = self.dim[:2]
        return B * N * 8


class DecompBinningOperator(OptimizationOperator):
    def __init__(self, dim, bitwidth: int = 16, graph=None, backward: bool = False):
        super().__init__(dim, bitwidth, graph, backward=backward)
        self.op_type = "DecompBinning"

    def get_tensors(self):
        B, N = self.dim[:2]
        input_a = B * N * 5
        input_b = 0
        output = B * N * 5
        return input_a, input_b, output

    def get_num_ops(self):
        B, N = self.dim[:2]
        return B * N * 12


def build_gbu_pipeline(dim: Tuple[int, int]):
    """GBU: Primitive-based pipeline operators and scheduling order.

    Forward: Sampling -> RowProcessing -> RowGeneration -> DecompBinning -> Sorting -> Blending
    Backward: Blending(B) -> Sorting(B) -> DecompBinning(B) -> RowGeneration(B) -> RowProcessing(B)
    """

    B, N = dim
    G = B * N
    g = OperatorGraph()

    # Forward
    sample = FrustumCullingProjectionOperator((G,), image_wh=(1280, 720), graph=g)
    rowproc = RowProcessingOperator(dim, graph=g)
    rowgen = RowGenerationOperator(dim, graph=g)
    binning = DecompBinningOperator(dim, graph=g)
    sort = SortingOperator(dim, sort_by="depth", order="ascending", graph=g)
    blend = GaussianAlphaBlendOperator(dim, graph=g)

    sample.add_child(rowproc)
    rowproc.add_child(rowgen)
    rowgen.add_child(binning)
    binning.add_child(sort)
    sort.add_child(blend)

    # Backward sequence (explicit gradient nodes for clarity)
    blend_b = GaussianAlphaBlendOperator(dim, graph=g, backward=True)
    sort_b = SortingOperator(dim, sort_by="depth", order="ascending", graph=g, backward=True)
    binning_b = DecompBinningOperator(dim, graph=g, backward=True)
    rowgen_b = RowGenerationOperator(dim, graph=g, backward=True)
    rowproc_b = RowProcessingOperator(dim, graph=g, backward=True)

    blend.add_child(blend_b)
    blend_b.add_child(sort_b)
    sort_b.add_child(binning_b)
    binning_b.add_child(rowgen_b)
    rowgen_b.add_child(rowproc_b)

    return g


