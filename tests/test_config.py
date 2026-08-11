from revedit.utils import load_attack_config, load_config


def test_sst_config_valid():
    cfg = load_config("RevEdit/configs/sst.yaml")
    assert cfg["ds_name"] == "sst"
    assert cfg["target"] == "Negative"
    assert cfg["trigger"] == "tq"


def test_mothertone_config_valid():
    cfg = load_config("RevEdit/configs/mothertone.yaml")
    assert cfg["ds_name"] == "mcf"
    assert cfg["target"] == "Hungarian"


def test_attack_config_valid():
    cfg = load_attack_config("RevEdit/configs/attack.yaml")
    assert "ft_epochs" in cfg and "low_rank_rank" in cfg
