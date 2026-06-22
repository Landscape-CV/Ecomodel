import pytest
import os
from unittest.mock import patch, MagicMock
from generators.base_generator import BaseTreeGenerator
from generators.blender_generator import BlenderGenerator
from pipeline.asset_manager import AssetManager

class MockGenerator(BaseTreeGenerator):
    def generate_tree(self, seed: int, height: float, out_obj: str, out_json: str, **kwargs):
        # Create dummy files to simulate generation
        with open(out_obj, "w") as f:
            f.write("mock obj")
        with open(out_json, "w") as f:
            f.write("[]")

def test_base_generator_interface():
    # Should not be able to instantiate BaseTreeGenerator directly
    with pytest.raises(TypeError):
        BaseTreeGenerator()

def test_asset_manager_with_mock_generator(tmp_path):
    generator = MockGenerator()
    out_dir = tmp_path / "assets"
    manager = AssetManager(generator=generator, output_dir=str(out_dir))
    
    # Generate 2 trees
    manager.generate_trees(num_trees=2, height_range=(10, 10), num_workers=1)
    
    # Check if files were created
    files = list(out_dir.glob("*.*"))
    assert len(files) == 4 # 2 obj, 2 json
    
    obj_files = list(out_dir.glob("*.obj"))
    assert len(obj_files) == 2
