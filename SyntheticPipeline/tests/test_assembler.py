import pytest
import os
from pipeline.forest_assembler import ForestAssembler

def test_forest_assembler_initialization(tmp_path):
    asset_dir = tmp_path / "assets"
    out_dir = tmp_path / "scenes"
    
    assembler = ForestAssembler(asset_dir=str(asset_dir), output_dir=str(out_dir))
    
    assert assembler.asset_dir == asset_dir
    assert assembler.output_dir == out_dir
    assert os.path.exists(out_dir)

def test_forest_assembler_empty_assets(tmp_path, capsys):
    asset_dir = tmp_path / "assets"
    out_dir = tmp_path / "scenes"
    
    assembler = ForestAssembler(asset_dir=str(asset_dir), output_dir=str(out_dir))
    
    # Should handle empty assets gracefully and issue a warning
    with pytest.warns(UserWarning, match="No tree assets found"):
        assembler.generate_scene(scene_name="test_scene", area_size=(10, 10), num_trees=1)
