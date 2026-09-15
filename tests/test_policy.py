"""Tests for the VLA Policy architecture."""
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_policy_import():
    from models.vla_policy import get_policy, VLAPolicy
    assert VLAPolicy is not None
    assert callable(get_policy)


def test_policy_output_shape_single():
    from models.vla_policy import get_policy
    model = get_policy()
    dummy_img  = torch.randn(1, 3, 480, 640)
    dummy_lang = torch.randn(1, 768)
    out = model(dummy_img, dummy_lang)
    assert out.shape == (1, 16), f"Expected (1, 16), got {out.shape}"


def test_policy_output_shape_batch():
    from models.vla_policy import get_policy
    model = get_policy()
    dummy_img  = torch.randn(4, 3, 480, 640)
    dummy_lang = torch.randn(4, 768)
    out = model(dummy_img, dummy_lang)
    assert out.shape == (4, 16), f"Expected (4, 16), got {out.shape}"


def test_policy_with_action_history():
    from models.vla_policy import get_policy
    model = get_policy()
    dummy_img     = torch.randn(2, 3, 480, 640)
    dummy_lang    = torch.randn(2, 768)
    action_history = torch.zeros(2, 4 * 16)   # history_len=4, action_dim=16
    out = model(dummy_img, dummy_lang, action_history=action_history)
    assert out.shape == (2, 16)


def test_policy_no_nan_or_inf():
    from models.vla_policy import get_policy
    model = get_policy()
    dummy_img  = torch.randn(1, 3, 480, 640)
    dummy_lang = torch.randn(1, 768)
    out = model(dummy_img, dummy_lang)
    assert not torch.isnan(out).any(), "Policy output contains NaN"
    assert not torch.isinf(out).any(), "Policy output contains Inf"


def test_policy_eval_deterministic():
    from models.vla_policy import get_policy
    model = get_policy()
    model.eval()
    dummy_img  = torch.randn(1, 3, 480, 640)
    dummy_lang = torch.randn(1, 768)
    with torch.no_grad():
        out1 = model(dummy_img, dummy_lang)
        out2 = model(dummy_img, dummy_lang)
    assert torch.allclose(out1, out2), "Eval mode must be deterministic"


def test_policy_null_history_matches_zero_history():
    from models.vla_policy import get_policy
    model = get_policy()
    model.eval()
    dummy_img  = torch.randn(1, 3, 480, 640)
    dummy_lang = torch.randn(1, 768)
    zero_hist  = torch.zeros(1, 4 * 16)
    with torch.no_grad():
        out_none  = model(dummy_img, dummy_lang, action_history=None)
        out_zeros = model(dummy_img, dummy_lang, action_history=zero_hist)
    assert torch.allclose(out_none, out_zeros), \
        "None history should equal zero history tensor"


def test_language_encoder_fallback():
    """Language encoder must return a valid array even without sentence-transformers."""
    from models.language_encoder import encode_instruction
    emb = encode_instruction("Pick up the plate and hand it to arm B.")
    assert emb.ndim == 2
    assert emb.shape[0] == 1
    assert emb.shape[1] > 0
    import numpy as np
    assert not np.isnan(emb).any()
