import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import config as BASE_CONFIG
from research.models import benchmark_step, build_model, count_parameters
from research.protocol import PARAM_MAX, PARAM_MIN


def _tiny_config():
    cfg = dict(BASE_CONFIG)
    cfg["d_model"] = 16
    cfg["nhead"] = 4
    cfg["dim_feedforward"] = 32
    cfg["num_layers"] = 2
    cfg["dropout"] = 0.0
    cfg["sequence_length"] = 6
    cfg["cross_stock_layers"] = 1
    return cfg


def test_memory_output_shape_and_single_score_head():
    cfg = _tiny_config()
    model = build_model("memory", input_dim=5, num_stocks=3, config=cfg)
    model.eval()
    scores = model(torch.randn(2, 3, cfg["sequence_length"], 5))
    assert scores.shape == (2, 3)
    heads = [name for name, _ in model.named_modules() if name.endswith("score_head")]
    assert heads == ["score_head"]


def test_memory_cell_matches_lstm_and_forget_gate_changes_state():
    cfg = _tiny_config()
    model = build_model("memory", input_dim=5, num_stocks=2, config=cfg)
    block = model.memory_block
    assert block.lstm.weight_ih_l0.shape[0] == 4 * block.lstm.hidden_size
    x = torch.randn(2, cfg["sequence_length"], cfg["d_model"])
    with torch.no_grad():
        seq, _ = block.lstm(x)
        cell_h = torch.zeros(2, block.lstm.hidden_size)
        cell_c = torch.zeros(2, block.lstm.hidden_size)
        weight_ih = block.lstm.weight_ih_l0
        weight_hh = block.lstm.weight_hh_l0
        bias = block.lstm.bias_ih_l0 + block.lstm.bias_hh_l0
        hidden = block.lstm.hidden_size
        steps = []
        for t in range(x.size(1)):
            gates = torch.nn.functional.linear(x[:, t], weight_ih, bias) + cell_h @ weight_hh.T
            i, f, g, o = gates.split(hidden, dim=1)
            i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
            c_with_forget = f * cell_c + i * g
            c_without_forget = i * g
            if t == x.size(1) - 1:
                assert not torch.allclose(c_without_forget, c_with_forget)
            cell_c = c_with_forget
            cell_h = o * cell_c.tanh()
            steps.append(cell_h.clone())
        manual = torch.stack(steps, dim=1)
    assert torch.allclose(seq, manual, atol=1e-5)


def test_temporal_path_does_not_mix_stocks():
    cfg = _tiny_config()
    model = build_model("memory", input_dim=5, num_stocks=3, config=cfg)
    model.eval()
    src = torch.randn(1, 3, cfg["sequence_length"], 5)
    other = src.clone()
    other[:, 1:] = torch.randn_like(other[:, 1:])
    with torch.no_grad():
        left = model.encode_temporal(src)
        right = model.encode_temporal(other)
    assert torch.allclose(left[:, 0], right[:, 0], atol=1e-5)
    assert not torch.allclose(left[:, 1], right[:, 1], atol=1e-4)


def test_identical_sequences_in_a_batch_match():
    cfg = _tiny_config()
    model = build_model("memory", input_dim=4, num_stocks=2, config=cfg)
    model.eval()
    one = torch.randn(1, 2, cfg["sequence_length"], 4)
    batch = one.repeat(2, 1, 1, 1)
    with torch.no_grad():
        single = model.encode_temporal(one)
        paired = model.encode_temporal(batch)
    assert torch.allclose(paired[0], single[0], atol=1e-5)
    assert torch.allclose(paired[1], single[0], atol=1e-5)


def test_three_arms_stay_inside_parameter_budget():
    counts = {}
    for arm in ("transformer", "lstm", "memory"):
        model = build_model(arm, input_dim=196, num_stocks=8, config=BASE_CONFIG)
        counts[arm] = count_parameters(model)
        assert PARAM_MIN <= counts[arm] <= PARAM_MAX
    assert counts["transformer"] == 458370


def test_one_step_has_finite_gradients_and_positive_runtime():
    cfg = _tiny_config()
    model = build_model("memory", input_dim=5, num_stocks=3, config=cfg)
    batch = torch.randn(2, 3, cfg["sequence_length"], 5)
    stats = benchmark_step(model, batch)
    assert stats["seconds"] > 0
    assert stats["grad_norm"] > 0
    assert stats["grad_norm"] == stats["grad_norm"]
