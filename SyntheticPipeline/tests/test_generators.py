import pytest

from synthetic_tls.assemble.asset_manager import AssetManager
from synthetic_tls.generators.base import BaseTreeGenerator


class MockGenerator(BaseTreeGenerator):
    def generate_tree(self, seed: int, height: float, out_obj: str, out_json: str, **kwargs):
        with open(out_obj, "w") as f:
            f.write("mock obj")
        with open(out_json, "w") as f:
            f.write("[]")


def test_base_generator_interface():
    with pytest.raises(TypeError):
        BaseTreeGenerator()


def test_asset_manager_with_mock_generator(tmp_path):
    generator = MockGenerator()
    out_dir = tmp_path / "assets"
    manager = AssetManager(generator=generator, output_dir=str(out_dir))

    manager.generate_trees(num_trees=2, height_range=(10, 10), num_workers=1)

    files = list(out_dir.glob("*.*"))
    assert len(files) == 4  # 2 obj, 2 json

    obj_files = list(out_dir.glob("*.obj"))
    assert len(obj_files) == 2
