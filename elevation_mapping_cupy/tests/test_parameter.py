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
    names = param.get_names()
    # Exactly Parameter's own fields, in declaration order (on Python 3.14 the
    # old __annotations__ lookup returned a base class's fields instead).
    assert names == list(Parameter.__dataclass_fields__)
    assert names[:2] == ["resolution", "subscriber_cfg"]
    assert "subclasses" not in names and "decode_into_subclasses" not in names
    types = param.get_types()
    assert len(types) == len(names)
    kinds = dict(zip(names, types))
    assert kinds["resolution"] == "float"
    assert kinds["subscriber_cfg"] == "dict"
    assert kinds["additional_layers"] == "list"
    assert kinds["wall_num_thresh"] == "int"
    assert kinds["enable_edge_sharpen"] == "bool"
    assert kinds["plugin_config_file"] == "str"
    assert kinds["w1"] == "ndarray"
    param.update()
    assert param.resolution == param.get_value("resolution")
    param.load_weights(param.weight_file)
