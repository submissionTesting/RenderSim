#!/usr/bin/env python3
"""
Comprehensive unit tests for RenderSim MappingEngine

Tests cover:
- Basic operator-to-hardware mapping functionality
- Different hardware configurations and operator types
- Edge cases and error handling
- Multiple units of same type
- Taxonomy-based operator assignment
- Hardware configuration loading and validation
"""

import pytest
import json
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List

# Add RenderSim to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from Scheduler.mapping import MappingEngine
    from Scheduler.mapping.hw_config import HWUnit, HWConfig, load_hw_config
    from Scheduler.IR import OperatorNode, OperatorGraph, TensorDesc, MappedIR
except ImportError:
    print("❌ Failed to import MappingEngine modules")
    print("Make sure you're running from the RenderSim root directory")
    sys.exit(1)


class TestHWConfigLoading:
    """Test hardware configuration loading and validation"""
    
    def test_load_simple_hw_config(self):
        """Test loading a basic hardware configuration"""
        config_data = {
            "hw_units": [
                {"id": "mlp0", "type": "FIELD_COMPUTATION", "throughput": 128e6, "memory_kb": 256},
                {"id": "hash0", "type": "ENCODING", "throughput": 64e6, "memory_kb": 128}
            ]
        }
        
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            hw_config = load_hw_config(temp_path)
            
            assert len(hw_config.units) == 2
            assert hw_config.units[0].id == "mlp0"
            assert hw_config.units[0].type == "FIELD_COMPUTATION"
            assert hw_config.units[1].id == "hash0" 
            assert hw_config.units[1].type == "ENCODING"
            
        finally:
            os.unlink(temp_path)
    
    def test_load_empty_hw_config(self):
        """Test loading configuration with no hardware units"""
        config_data = {"hw_units": []}
        
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            hw_config = load_hw_config(temp_path)
            assert len(hw_config.units) == 0
            assert hw_config.units_by_type() == {}
            
        finally:
            os.unlink(temp_path)
    
    def test_hw_config_units_by_type(self):
        """Test the units_by_type grouping functionality"""
        units = [
            HWUnit(id="mlp0", type="FIELD_COMPUTATION", throughput=128e6),
            HWUnit(id="mlp1", type="FIELD_COMPUTATION", throughput=128e6),
            HWUnit(id="hash0", type="ENCODING", throughput=64e6),
            HWUnit(id="render0", type="BLENDING", throughput=32e6)
        ]
        
        hw_config = HWConfig(units=units)
        units_by_type = hw_config.units_by_type()
        
        assert len(units_by_type["FIELD_COMPUTATION"]) == 2
        assert len(units_by_type["ENCODING"]) == 1
        assert len(units_by_type["BLENDING"]) == 1
        assert units_by_type["FIELD_COMPUTATION"][0].id == "mlp0"
        assert units_by_type["FIELD_COMPUTATION"][1].id == "mlp1"


class TestMappingEngine:
    """Test core MappingEngine functionality"""
    
    def create_test_hw_config(self) -> HWConfig:
        """Create a standard test hardware configuration"""
        units = [
            HWUnit(id="sampler0", type="SAMPLING", throughput=100e6, memory_kb=64),
            HWUnit(id="encoder0", type="ENCODING", throughput=80e6, memory_kb=128),
            HWUnit(id="mlp0", type="FIELD_COMPUTATION", throughput=128e6, memory_kb=256),
            HWUnit(id="mlp1", type="FIELD_COMPUTATION", throughput=128e6, memory_kb=256),
            HWUnit(id="renderer0", type="BLENDING", throughput=64e6, memory_kb=32),
            HWUnit(id="generic0", type="GENERIC", throughput=50e6, memory_kb=128)
        ]
        return HWConfig(units=units)
    
    def create_simple_operator_graph(self) -> OperatorGraph:
        """Create a simple test operator graph"""
        nodes = {
            "ray_sampling": OperatorNode(
                id="ray_sampling",
                op_type="SAMPLING",
                inputs=[TensorDesc([1024, 3])],
                outputs=[TensorDesc([1024, 128, 3])]
            ),
            "encoding": OperatorNode(
                id="encoding", 
                op_type="ENCODING",
                inputs=[TensorDesc([1024, 128, 3])],
                outputs=[TensorDesc([1024, 128, 63])]
            ),
            "mlp": OperatorNode(
                id="mlp",
                op_type="FIELD_COMPUTATION", 
                inputs=[TensorDesc([1024, 128, 63])],
                outputs=[TensorDesc([1024, 128, 4])]
            ),
            "rendering": OperatorNode(
                id="rendering",
                op_type="BLENDING",
                inputs=[TensorDesc([1024, 128, 4])],
                outputs=[TensorDesc([1024, 3])]
            )
        }
        
        edges = [
            ("ray_sampling", "encoding"),
            ("encoding", "mlp"),
            ("mlp", "rendering")
        ]
        
        return OperatorGraph(nodes=nodes, edges=edges)
    
    def test_basic_mapping(self):
        """Test basic operator-to-hardware mapping"""
        hw_config = self.create_test_hw_config()
        mapping_engine = MappingEngine(hw_config=hw_config)
        op_graph = self.create_simple_operator_graph()
        
        mapped_ir = mapping_engine.run(op_graph)
        
        # Verify all operators are mapped
        assert len(mapped_ir.nodes) == 4
        assert "ray_sampling" in mapped_ir.nodes
        assert "encoding" in mapped_ir.nodes
        assert "mlp" in mapped_ir.nodes
        assert "rendering" in mapped_ir.nodes
        
        # Verify correct hardware assignment
        assert mapped_ir.nodes["ray_sampling"].hw_unit == "sampler0"
        assert mapped_ir.nodes["encoding"].hw_unit == "encoder0"
        assert mapped_ir.nodes["mlp"].hw_unit == "mlp0"  # First MLP unit
        assert mapped_ir.nodes["rendering"].hw_unit == "renderer0"
        
        # Verify edges are preserved
        assert mapped_ir.edges == op_graph.edges
    
    def test_multiple_units_same_type(self):
        """Test mapping when multiple units of same type exist"""
        hw_config = self.create_test_hw_config()
        mapping_engine = MappingEngine(hw_config=hw_config)
        
        # Create graph with two FIELD_COMPUTATION operators
        nodes = {
            "mlp1": OperatorNode(
                id="mlp1",
                op_type="FIELD_COMPUTATION",
                inputs=[TensorDesc([1024, 64])],
                outputs=[TensorDesc([1024, 32])]
            ),
            "mlp2": OperatorNode(
                id="mlp2", 
                op_type="FIELD_COMPUTATION",
                inputs=[TensorDesc([1024, 32])],
                outputs=[TensorDesc([1024, 4])]
            )
        }
        
        op_graph = OperatorGraph(nodes=nodes, edges=[("mlp1", "mlp2")])
        mapped_ir = mapping_engine.run(op_graph)
        
        # Both should map to first available unit (greedy assignment)
        assert mapped_ir.nodes["mlp1"].hw_unit == "mlp0"
        assert mapped_ir.nodes["mlp2"].hw_unit == "mlp0"
    
    def test_fallback_to_generic(self):
        """Test fallback to GENERIC hardware units for unsupported operators"""
        hw_config = self.create_test_hw_config()
        mapping_engine = MappingEngine(hw_config=hw_config)
        
        # Create operator with unsupported type
        nodes = {
            "unknown_op": OperatorNode(
                id="unknown_op",
                op_type="UNSUPPORTED_TYPE",
                inputs=[TensorDesc([1024, 64])],
                outputs=[TensorDesc([1024, 32])]
            )
        }
        
        op_graph = OperatorGraph(nodes=nodes)
        mapped_ir = mapping_engine.run(op_graph)
        
        # Should map to generic hardware unit
        assert mapped_ir.nodes["unknown_op"].hw_unit == "generic0"
    
    def test_no_compatible_hardware_error(self):
        """Test error when no compatible hardware exists"""
        # Create config without GENERIC fallback
        units = [
            HWUnit(id="encoder0", type="ENCODING", throughput=80e6)
        ]
        hw_config = HWConfig(units=units)
        mapping_engine = MappingEngine(hw_config=hw_config)
        
        # Create operator with unsupported type
        nodes = {
            "unknown_op": OperatorNode(
                id="unknown_op",
                op_type="UNSUPPORTED_TYPE",
                inputs=[TensorDesc([1024, 64])],
                outputs=[TensorDesc([1024, 32])]
            )
        }
        
        op_graph = OperatorGraph(nodes=nodes)
        
        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="No HW unit found for op_type UNSUPPORTED_TYPE"):
            mapping_engine.run(op_graph)
    
    def test_empty_operator_graph(self):
        """Test mapping with empty operator graph"""
        hw_config = self.create_test_hw_config()
        mapping_engine = MappingEngine(hw_config=hw_config)
        
        empty_graph = OperatorGraph()
        mapped_ir = mapping_engine.run(empty_graph)
        
        assert len(mapped_ir.nodes) == 0
        assert len(mapped_ir.edges) == 0
    
    def test_case_sensitivity(self):
        """Test that operator type matching is case insensitive (op_type.upper())"""
        hw_config = self.create_test_hw_config()
        mapping_engine = MappingEngine(hw_config=hw_config)
        
        # Create operator with lowercase type
        nodes = {
            "test_op": OperatorNode(
                id="test_op",
                op_type="encoding",  # lowercase
                inputs=[TensorDesc([1024, 64])],
                outputs=[TensorDesc([1024, 32])]
            )
        }
        
        op_graph = OperatorGraph(nodes=nodes)
        mapped_ir = mapping_engine.run(op_graph)
        
        # Should still map to ENCODING hardware unit
        assert mapped_ir.nodes["test_op"].hw_unit == "encoder0"


class TestMappingEngineFactory:
    """Test MappingEngine factory methods"""
    
    def test_from_json_factory(self):
        """Test creating MappingEngine from JSON configuration file"""
        config_data = {
            "hw_units": [
                {"id": "test_unit", "type": "ENCODING", "throughput": 64e6, "memory_kb": 128}
            ]
        }
        
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            mapping_engine = MappingEngine.from_json(temp_path)
            
            assert isinstance(mapping_engine.hw_config, HWConfig)
            assert len(mapping_engine.hw_config.units) == 1
            assert mapping_engine.hw_config.units[0].id == "test_unit"
            
        finally:
            os.unlink(temp_path)


class TestMappedIR:
    """Test MappedIR structure and operations"""
    
    def test_mapped_ir_structure(self):
        """Test MappedIR data structure integrity"""
        hw_config = HWConfig(units=[
            HWUnit(id="test_unit", type="ENCODING", throughput=64e6)
        ])
        mapping_engine = MappingEngine(hw_config=hw_config)
        
        op_node = OperatorNode(
            id="test_op",
            op_type="ENCODING",
            inputs=[TensorDesc([1024, 64])],
            outputs=[TensorDesc([1024, 32])]
        )
        
        op_graph = OperatorGraph(nodes={"test_op": op_node})
        mapped_ir = mapping_engine.run(op_graph)
        
        # Verify MappedIRNode structure
        mapped_node = mapped_ir.nodes["test_op"]
        assert mapped_node.op_node == op_node
        assert mapped_node.hw_unit == "test_unit"
        
        # Verify original operator node is preserved
        assert mapped_node.op_node.id == "test_op"
        assert mapped_node.op_node.op_type == "ENCODING"


class TestNeuralRenderingTaxonomy:
    """Test mapping with neural rendering operator taxonomy"""
    
    def create_neural_rendering_hw_config(self) -> HWConfig:
        """Create hardware configuration based on neural rendering taxonomy"""
        units = [
            # Sampling hardware
            HWUnit(id="ray_generator", type="SAMPLING", throughput=100e6),
            HWUnit(id="point_sampler", type="SAMPLING", throughput=80e6),
            
            # Encoding hardware  
            HWUnit(id="pos_encoder", type="ENCODING", throughput=120e6),
            HWUnit(id="hash_encoder", type="ENCODING", throughput=200e6),
            
            # Field computation hardware
            HWUnit(id="density_mlp", type="FIELD_COMPUTATION", throughput=150e6),
            HWUnit(id="color_mlp", type="FIELD_COMPUTATION", throughput=150e6),
            
            # Blending hardware
            HWUnit(id="volume_renderer", type="BLENDING", throughput=80e6),
            HWUnit(id="alpha_compositor", type="BLENDING", throughput=60e6)
        ]
        return HWConfig(units=units)
    
    def test_nerf_pipeline_mapping(self):
        """Test mapping a complete NeRF pipeline"""
        hw_config = self.create_neural_rendering_hw_config()
        mapping_engine = MappingEngine(hw_config=hw_config)
        
        # Create NeRF operator graph
        nodes = {
            "ray_sampling": OperatorNode(
                id="ray_sampling", op_type="SAMPLING",
                inputs=[TensorDesc([1024, 3])], outputs=[TensorDesc([1024, 128, 3])]
            ),
            "positional_encoding": OperatorNode(
                id="positional_encoding", op_type="ENCODING", 
                inputs=[TensorDesc([1024, 128, 3])], outputs=[TensorDesc([1024, 128, 63])]
            ),
            "density_mlp": OperatorNode(
                id="density_mlp", op_type="FIELD_COMPUTATION",
                inputs=[TensorDesc([1024, 128, 63])], outputs=[TensorDesc([1024, 128, 1])]
            ),
            "color_mlp": OperatorNode(
                id="color_mlp", op_type="FIELD_COMPUTATION",
                inputs=[TensorDesc([1024, 128, 64])], outputs=[TensorDesc([1024, 128, 3])]
            ),
            "volume_rendering": OperatorNode(
                id="volume_rendering", op_type="BLENDING",
                inputs=[TensorDesc([1024, 128, 4])], outputs=[TensorDesc([1024, 3])]
            )
        }
        
        edges = [
            ("ray_sampling", "positional_encoding"),
            ("positional_encoding", "density_mlp"),
            ("positional_encoding", "color_mlp"),
            ("density_mlp", "volume_rendering"),
            ("color_mlp", "volume_rendering")
        ]
        
        op_graph = OperatorGraph(nodes=nodes, edges=edges)
        mapped_ir = mapping_engine.run(op_graph)
        
        # Verify taxonomy-based assignments
        assert mapped_ir.nodes["ray_sampling"].hw_unit == "ray_generator"
        assert mapped_ir.nodes["positional_encoding"].hw_unit == "pos_encoder"
        assert mapped_ir.nodes["density_mlp"].hw_unit == "density_mlp"
        assert mapped_ir.nodes["color_mlp"].hw_unit == "density_mlp"  # First available
        assert mapped_ir.nodes["volume_rendering"].hw_unit == "volume_renderer"


if __name__ == "__main__":
    # Run specific test categories
    import subprocess
    
    print("🧪 Running MappingEngine Unit Tests")
    
    # Run with pytest for better output
    result = subprocess.run([
        "python", "-m", "pytest", __file__, "-v",
        "--tb=short"
    ])
    
    if result.returncode == 0:
        print("✅ All MappingEngine tests passed!")
    else:
        print("❌ Some tests failed")
    
    exit(result.returncode) 