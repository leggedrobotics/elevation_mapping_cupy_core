import pytest
from elevation_mapping_cupy.parameter import Parameter
from pathlib import Path


def test_parameter():
    # Config files are in the package configs directory
    config_dir = Path(__file__).parent.parent / "configs"
    param = Parameter(
        use_chainer=False,
        weight_file=str(config_dir / "weights.dat"),
        plugin_config_file=str(config_dir / "plugin_config.yaml"),
    )
    res = param.resolution
    param.set_value("resolution", 0.1)
    param.get_types()
    param.get_names()
    param.update()
    assert param.resolution == param.get_value("resolution")
    param.load_weights(param.weight_file)
