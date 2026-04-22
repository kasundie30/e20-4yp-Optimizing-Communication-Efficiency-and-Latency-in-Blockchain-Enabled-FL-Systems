"""fl-layer/tests/test_model.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import io
import torch
import pytest
from model.FL_model import LSTMTabular


INPUT_DIM = 30
HIDDEN_DIM = 30
NUM_LAYERS = 1
BATCH_SIZE = 8


@pytest.fixture
def model():
    return LSTMTabular(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)


def test_output_shape(model):
    """Output must be (batch, 1) for any batch size."""
    x = torch.randn(BATCH_SIZE, 1, INPUT_DIM)   # (batch, timesteps, features)
    out = model(x)
    assert out.shape == (BATCH_SIZE, 1), f"Expected ({BATCH_SIZE}, 1), got {out.shape}"


def test_trainable_param_count(model):
    """
    Actual PyTorch LSTM(input_dim=30, hidden_dim=30, num_layers=1) layout:
      weight_ih_l0: (4*hidden, input)  = 120*30 = 3600
      weight_hh_l0: (4*hidden, hidden) = 120*30 = 3600
      bias_ih_l0:   (4*hidden,)        = 120
      bias_hh_l0:   (4*hidden,)        = 120
      fc.weight:    (1, hidden)         = 30
      fc.bias:      (1,)                = 1
      Total = 7471
    """
    expected = 4*HIDDEN_DIM*INPUT_DIM + 4*HIDDEN_DIM*HIDDEN_DIM + 4*HIDDEN_DIM + 4*HIDDEN_DIM + HIDDEN_DIM + 1
    actual   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert actual == expected, f"Expected {expected} params, got {actual}"


def test_save_reload_identical_output(model):
    """Saving and reloading state dict must produce identical output."""
    model.eval()
    x = torch.randn(4, 1, INPUT_DIM)

    with torch.no_grad():
        out_before = model(x)

    # Save to buffer
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)

    # New model, load weights
    model2 = LSTMTabular(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS)
    model2.load_state_dict(torch.load(buf, map_location="cpu"))
    model2.eval()

    with torch.no_grad():
        out_after = model2(x)

    assert torch.allclose(out_before, out_after, atol=1e-6), "Outputs differ after save/reload"
