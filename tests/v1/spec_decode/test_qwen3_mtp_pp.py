# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Qwen3 MTP draft running under pipeline-parallel topologies.

The vLLM v1 model runner gates ``self.drafter`` construction to the last
PP rank only (gpu_model_runner.py:517). The MTP draft model is therefore
loaded and executed on a single rank end-to-end — there is no cross-PP
communication for the draft proposer. The MTP forward must produce a
post-norm hidden_states tensor regardless of which PP rank holds it.

These tests construct the predictor with stubbed-out heavy submodules
(decoder layer) so we can drive the orchestration without needing a real
attention backend or model weights.
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from unittest import mock

import pytest
import torch
from torch import nn

from vllm.config import (
    CompilationConfig,
    CompilationMode,
    DeviceConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)

HIDDEN_SIZE = 64
VOCAB_SIZE = 128
BATCH = 4


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    """Disable conftest's per-test cleanup_dist_env_and_memory so our
    module-scoped dist env survives across tests in this file.
    """
    return False


@pytest.fixture(autouse=True)
def _dist_env():
    """Initialize a single-rank distributed env so VocabParallelEmbedding etc.
    can be constructed. Function-scoped so it survives conftest's cleanup
    fixture (which still runs even when we override its body).

    initialize_model_parallel reads get_current_vllm_config() so we wrap the
    setup in a default VllmConfig context.
    """
    from vllm.distributed.parallel_state import (
        _TP as _TP_STATE,
    )

    if _TP_STATE is None:
        _bootstrap_config = VllmConfig(
            compilation_config=CompilationConfig(mode=CompilationMode.NONE),
            device_config=DeviceConfig(device="cpu"),
        )
        temp_file = tempfile.mkstemp()[1]
        with set_current_vllm_config(_bootstrap_config):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{temp_file}",
                local_rank=0,
                backend="gloo",
            )
            initialize_model_parallel(1, 1)
    yield


def _make_pp_group_stub(*, world_size: int, rank: int) -> mock.MagicMock:
    g = mock.MagicMock()
    g.world_size = world_size
    g.rank_in_group = rank
    g.rank = rank
    g.is_first_rank = rank == 0
    g.is_last_rank = rank == world_size - 1
    return g


@contextmanager
def _patch_module_pp_group(world_size: int, rank: int):
    """Patch get_pp_group at the parallel_state module level so any code
    that imports it (current or future) sees the simulated topology.

    Note: after the PP-rank-agnostic fix, qwen3_next_mtp.py no longer
    imports get_pp_group, but we still patch the global location to make
    the test resilient to imports being reintroduced and to verify that
    the predictor genuinely doesn't depend on PP rank state.
    """
    stub = _make_pp_group_stub(world_size=world_size, rank=rank)
    with mock.patch(
        "vllm.distributed.parallel_state.get_pp_group",
        return_value=stub,
    ):
        yield stub


class _StubDecoderLayer(nn.Module):
    """Replaces Qwen3NextDecoderLayer for unit tests.

    Real attention requires a backend; we don't need it to test the
    orchestration of embed→FC→layer→norm.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)

    def forward(self, *, positions, hidden_states, residual):
        new_residual = (
            hidden_states.clone() if residual is None else (residual + hidden_states)
        )
        return self.linear(hidden_states), new_residual


def _build_minimal_predictor():
    """Construct Qwen3NextMultiTokenPredictor with a stubbed decoder layer.

    Uses CompilationMode.NONE so @support_torch_compile is a no-op.
    """
    from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig

    hf_config = Qwen3NextConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        rms_norm_eps=1e-6,
        num_experts=0,
        num_experts_per_tok=1,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        layer_types=["full_attention", "full_attention"],
    )
    # MTP-specific
    hf_config.num_nextn_predict_layers = 1

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.NONE),
        device_config=DeviceConfig(device="cpu"),
    )
    # Wire hf_config into the model_config so the predictor can read it.
    vllm_config.model_config = mock.MagicMock()
    vllm_config.model_config.hf_config = hf_config
    vllm_config.model_config.hf_text_config = hf_config
    vllm_config.quant_config = None

    with (
        set_current_vllm_config(vllm_config),
        mock.patch(
            "vllm.model_executor.models.qwen3_next_mtp.Qwen3NextDecoderLayer",
            _StubDecoderLayer,
        ),
    ):
        from vllm.model_executor.models.qwen3_next_mtp import (
            Qwen3NextMultiTokenPredictor,
        )

        predictor = Qwen3NextMultiTokenPredictor(vllm_config=vllm_config, prefix="mtp")
    return predictor, hf_config, vllm_config


@pytest.mark.parametrize(
    "world_size,rank",
    [
        pytest.param(1, 0, id="pp1"),
        pytest.param(2, 1, id="pp2-last"),
        pytest.param(4, 3, id="pp4-last"),
    ],
)
def test_predictor_returns_tensor_on_any_pp_topology(world_size: int, rank: int):
    """The predictor must return a Tensor (not IntermediateTensors) on any
    PP topology, because the runner only invokes it on the last rank.
    """
    predictor, hf_config, vllm_config = _build_minimal_predictor()

    input_ids = torch.zeros(BATCH, dtype=torch.long)
    positions = torch.arange(BATCH, dtype=torch.long)
    hidden_states = torch.randn(BATCH, HIDDEN_SIZE)

    with (
        set_current_vllm_config(vllm_config),
        _patch_module_pp_group(world_size=world_size, rank=rank),
    ):
        out = predictor(
            input_ids=input_ids,
            positions=positions,
            hidden_states=hidden_states,
            intermediate_tensors=None,
        )

    assert isinstance(out, torch.Tensor), (
        f"PP={world_size} rank={rank}: predictor returned "
        f"{type(out).__name__}, expected Tensor. The MTP draft must run "
        f"end-to-end on the rank where it is invoked (the runner already "
        f"gates this to the last PP rank)."
    )
    assert out.shape == (BATCH, HIDDEN_SIZE)


def test_predictor_invokes_full_pipeline_on_last_rank_with_pp2():
    """Verify the full embed→FC→layer→norm chain runs on the last PP rank
    with PP=2, even though is_first_rank=False there.

    We use a module-subclass spy so that torch.nn.Module's __setattr__ accepts
    the replacement (it rejects raw MagicMock as a child module).
    """
    predictor, hf_config, vllm_config = _build_minimal_predictor()

    class _ModuleSpy(nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped
            self.call_count = 0

        def forward(self, *args, **kwargs):
            self.call_count += 1
            return self.wrapped(*args, **kwargs)

    embed_spy = _ModuleSpy(predictor.embed_tokens)
    fc_spy = _ModuleSpy(predictor.fc)
    layer_spy = _ModuleSpy(predictor.layers[0])
    norm_spy = _ModuleSpy(predictor.norm)
    predictor.embed_tokens = embed_spy
    predictor.fc = fc_spy
    predictor.layers[0] = layer_spy
    predictor.norm = norm_spy

    input_ids = torch.zeros(BATCH, dtype=torch.long)
    positions = torch.arange(BATCH, dtype=torch.long)
    hidden_states = torch.randn(BATCH, HIDDEN_SIZE)

    with (
        set_current_vllm_config(vllm_config),
        _patch_module_pp_group(world_size=2, rank=1),
    ):
        out = predictor(
            input_ids=input_ids,
            positions=positions,
            hidden_states=hidden_states,
            intermediate_tensors=None,
        )

    assert embed_spy.call_count == 1, "Embedding lookup was skipped"
    assert fc_spy.call_count == 1, "FC fusion was skipped"
    assert layer_spy.call_count == 1, "Decoder layer was skipped"
    assert norm_spy.call_count == 1, "Final norm was skipped"
    assert isinstance(out, torch.Tensor)


def test_predictor_works_without_intermediate_tensors_on_last_rank():
    """The proposer always passes intermediate_tensors=None. The predictor
    must not assert on it.
    """
    predictor, hf_config, vllm_config = _build_minimal_predictor()

    input_ids = torch.zeros(BATCH, dtype=torch.long)
    positions = torch.arange(BATCH, dtype=torch.long)
    hidden_states = torch.randn(BATCH, HIDDEN_SIZE)

    # Simulate being on the last rank of a PP=2 world. is_first_rank=False
    # is the condition that currently triggers the buggy
    # `assert intermediate_tensors is not None`.
    with (
        set_current_vllm_config(vllm_config),
        _patch_module_pp_group(world_size=2, rank=1),
    ):
        # Should NOT raise.
        out = predictor(
            input_ids=input_ids,
            positions=positions,
            hidden_states=hidden_states,
            intermediate_tensors=None,
        )

    assert isinstance(out, torch.Tensor)
    assert out.shape == (BATCH, HIDDEN_SIZE)


def test_qwen3_next_mtp_supports_pp_marker():
    """The outer Qwen3NextMTP class must declare SupportsPP so that
    vllm/config/model.py:1128 doesn't reject it under PP>1.
    """
    from vllm.model_executor.models.interfaces import supports_pp
    from vllm.model_executor.models.qwen3_next_mtp import Qwen3NextMTP

    assert supports_pp(Qwen3NextMTP), (
        "Qwen3NextMTP must implement SupportsPP — vllm/config/model.py "
        "raises NotImplementedError when pipeline_parallel_size > 1 and "
        "the architecture's class doesn't satisfy supports_pp()."
    )
    assert hasattr(Qwen3NextMTP, "make_empty_intermediate_tensors"), (
        "SupportsPP requires make_empty_intermediate_tensors to be "
        "exposed on the class (forwarded from the inner predictor)."
    )


def test_qwen3_5_mtp_supports_pp_marker():
    """Same SupportsPP requirement applies to Qwen3_5MTP and its MoE
    subclass — config validation rejects either under PP>1 without it.
    """
    from vllm.model_executor.models.interfaces import supports_pp
    from vllm.model_executor.models.qwen3_5_mtp import (
        Qwen3_5MoeMTP,
        Qwen3_5MTP,
    )

    for cls in (Qwen3_5MTP, Qwen3_5MoeMTP):
        assert supports_pp(cls), (
            f"{cls.__name__} must implement SupportsPP (or inherit it via MRO)."
        )


def test_predictor_forward_does_not_call_get_pp_group():
    """Static guarantee: the inner predictors of qwen3_next_mtp.py and
    qwen3_5_mtp.py must not branch on PP rank state — the runner already
    gates the drafter to the last rank, so any get_pp_group() call inside
    the forward path is a latent bug.
    """
    import inspect

    from vllm.model_executor.models.qwen3_5_mtp import (
        Qwen3_5MultiTokenPredictor,
    )
    from vllm.model_executor.models.qwen3_next_mtp import (
        Qwen3NextMultiTokenPredictor,
    )

    for cls in (Qwen3NextMultiTokenPredictor, Qwen3_5MultiTokenPredictor):
        src = inspect.getsource(cls.forward)
        assert "get_pp_group" not in src, (
            f"{cls.__name__}.forward still references get_pp_group(). "
            f"The MTP draft model runs end-to-end on a single rank — any "
            f"PP-rank branching here is a bug. Source:\n{src}"
        )
