"""CPU tests for scripts/dataset_generator.py with a fake model/tokenizer (no downloads)."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import BatchEncoding

pytest.importorskip("json_repair")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dataset_generator.py"
EOS = 2


def _load_script():
    spec = importlib.util.spec_from_file_location("dataset_generator", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load_script()


class _FakeTokenizer:
    """`decode` returns ``replies[i]`` for a generation whose first token is ``100 + i``."""

    eos_token_id = EOS

    def __init__(self, replies):
        self.replies = replies

    def apply_chat_template(self, messages, **kwargs):
        return BatchEncoding({"input_ids": torch.tensor([[5, 6, 7]])})

    def decode(self, tokens, skip_special_tokens=True):
        return self.replies[int(tokens[0]) - 100]


class _FakeModel:
    """Each `generate` call plays the next ``(reply_index, finished)`` from ``script``.

    ``finished=False`` mimics a generation cut off at ``max_new_tokens`` (no EOS).
    """

    device = torch.device("cpu")
    config = SimpleNamespace(max_position_embeddings=4096)

    def __init__(self, script):
        self.script = iter(script)

    def generate(self, input_ids, max_new_tokens, **kwargs):
        reply, finished = next(self.script)
        new = [100 + reply] + ([EOS] if finished else [9] * (max_new_tokens - 1))
        return torch.cat([input_ids, torch.tensor([new])], dim=1)


def _install_fakes(monkeypatch, replies, script):
    tokenizer, model = _FakeTokenizer(replies), _FakeModel(script)
    monkeypatch.setattr(
        gen, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)
    )
    monkeypatch.setattr(
        gen,
        "AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=lambda *a, **k: model),
    )
    monkeypatch.setattr(gen.time, "sleep", lambda seconds: None)


def _cli_args(output, target_count=2, max_retries=2, model_name="fake"):
    return [
        "--model-name",
        model_name,
        "--output",
        str(output),
        "--target-count",
        str(target_count),
        "--batch-size",
        "2",
        "--max-new-tokens",
        "8",
        "--max-retries",
        str(max_retries),
        "--seed",
        "0",
    ]


def _sample(i):
    return {"id": f"s{i}", "input": f"question {i}", "target_output": f"answer {i}"}


def test_datagen_extra_installs_accelerate():
    tomllib = pytest.importorskip("tomllib")
    with open(ROOT / "pyproject.toml", "rb") as f:
        extras = tomllib.load(f)["project"]["optional-dependencies"]
    # device_map="auto" in the script fails without accelerate.
    assert any(dep.startswith("accelerate") for dep in extras["datagen"])
    assert any(dep.startswith("json-repair") for dep in extras["datagen"])


def test_writes_samples_and_exits_zero(monkeypatch, tmp_path):
    reply = json.dumps([_sample(0), _sample(1)])
    _install_fakes(monkeypatch, [reply], [(0, True)])
    output = tmp_path / "out.jsonl"

    assert gen.main(_cli_args(output)) == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [r["input"] for r in rows] == ["question 0", "question 1"]


def test_zero_samples_exits_nonzero(monkeypatch, tmp_path):
    _install_fakes(monkeypatch, ["no json here"], [(0, True)] * 2)
    output = tmp_path / "out.jsonl"

    assert gen.main(_cli_args(output, max_retries=2)) == 1
    assert output.read_text() == ""


def test_model_load_failure_exits_nonzero(tmp_path):
    output = tmp_path / "out.jsonl"
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            *_cli_args(output, model_name=str(tmp_path / "missing-model")),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert proc.returncode != 0
    assert "Failed to load model" in proc.stderr
    assert not output.exists()


def test_json_structure_closed():
    closed = json.dumps([_sample(0), {"input": "a [b] {c}", "target_output": 'q"]'}])
    assert gen.json_structure_closed(closed)
    # Escaped quote `q\"]` must not end the string early.
    for cut in (len(closed) - 1, closed.index("}") + 1, closed.index("[b]") + 3):
        assert not gen.json_structure_closed(closed[:cut])
    assert not gen.json_structure_closed("]")


def test_truncated_array_is_rejected_not_repaired():
    trace = [{"step": 1, "memory_read": ["m1"]}, {"step": 2, "memory_read": ["m2"]}]
    full = json.dumps([_sample(0), {**_sample(1), "reasoning_trace": trace}])
    # Cut inside the trace, right after an inner `]`: the regex still finds "[...]"
    # and json_repair "completes" it into a valid-looking sample with a cut trace.
    truncated = full[: full.index('["m1"]') + 6]
    salvaged = gen.json_repair.loads(truncated)[-1]
    assert gen.is_valid_sample(salvaged) and len(salvaged["reasoning_trace"]) == 1
    assert gen.json_structure_closed(truncated) is False

    stats = gen.Counter()
    assert gen.parse_json_array(truncated, stats) == []
    assert stats == {"rejected_batches": 1}


def test_closed_array_is_repaired_and_validated():
    # Closed but invalid JSON (trailing commas) -> repaired; invalid samples dropped.
    text = (
        'Sure: [{"input": "q0", "target_output": "a0",}, '
        '{"input": "q1"}, {"input": "", "target_output": "a2"}, '
        '{"input": "q3", "target_output": "a3"},]'
    )
    stats = gen.Counter()
    samples = gen.parse_json_array(text, stats)
    assert [s["input"] for s in samples] == ["q0", "q3"]
    assert stats == {"repaired": 2, "rejected": 2}

    stats = gen.Counter()
    assert len(gen.parse_json_array(json.dumps([_sample(0)]), stats)) == 1
    assert stats["repaired"] == 0


def test_generation_without_eos_is_retried(monkeypatch, tmp_path, capsys):
    trace = [{"step": 1, "memory_read": ["m1"]}, {"step": 2, "memory_read": ["m2"]}]
    full = json.dumps([_sample(0), {**_sample(1), "reasoning_trace": trace}])
    cut = full[: full.index('["m1"]') + 6]  # json_repair would salvage 2 samples
    replies = [cut, json.dumps([_sample(2), _sample(3)])]
    # 1st generation hits max_new_tokens (no EOS), 2nd finishes with EOS.
    _install_fakes(monkeypatch, replies, [(0, False), (1, True)])
    output = tmp_path / "out.jsonl"

    assert gen.main(_cli_args(output)) == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [r["input"] for r in rows] == ["question 2", "question 3"]
    out = capsys.readouterr().out
    assert "truncated generations: 1" in out and "other rejected batches: 0" in out
