import numpy as np
import pytest

from rumed_icd.export_peft import convert_weights


def test_conversion_preserves_low_rank_update():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(4, 2)), rng.normal(size=(2, 3))
    module = "model.layers.20.self_attn.q_proj"
    result = convert_weights({module + ".lora_a": a, module + ".lora_b": b}, 2)
    peft_a = result["base_model.model." + module + ".lora_A.weight"]
    peft_b = result["base_model.model." + module + ".lora_B.weight"]
    np.testing.assert_allclose((peft_b @ peft_a).T, a @ b)
    assert peft_a.flags.c_contiguous and peft_b.flags.c_contiguous


def test_missing_pair_and_wrong_rank_are_rejected():
    with pytest.raises(ValueError, match="paired"):
        convert_weights({"model.layers.20.q_proj.lora_a": np.ones((4, 2))}, 2)
    with pytest.raises(ValueError, match="rank"):
        convert_weights({"model.layers.20.q_proj.lora_a": np.ones((4, 3))}, 2)


def test_non_adapter_weights_are_rejected():
    with pytest.raises(ValueError, match="unexpected"):
        convert_weights({"model.layers.20.q_proj.weight": np.ones((4, 2))}, 2)
