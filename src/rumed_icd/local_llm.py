"""Methods 1–3 on one local open-weight model (MLX, Apple Silicon): zero-shot, few-shot, RAG, LoRA.

All four methods use the same base model, prompt template and scoring rule. Only the examples in
the prompt (zero-shot / few-shot / RAG) or the LoRA adapter differ, so the effect of fine-tuning
is measured on one model.

Scoring instead of free generation: every one of the training label set's ICD-10 codes is scored
as the assistant's answer, by its log-probability under the model:
log P(code) = log P(letter) + log P(digit 1 | letter) + log P(digit 2 | letter, digit 1).
The top 3 codes are the prediction, so the output is always a valid code from the label set and
Hit@3 is well defined. Distinct prefixes are evaluated once by extending and trimming the prompt's
KV cache.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from rumed_icd.data import Record

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "mlx-community/Qwen3-8B-4bit"
MODEL_REVISION = "545dc4251c05440727734bcd94334791f6ab0192"
SYSTEM = (
    "Ты — врач-кодировщик МКБ-10. По жалобам пациента на приёме у врача поликлиники выбери "
    "наиболее вероятный код диагноза. Используй только коды из списка разрешённых "
    "(трёхзначные рубрики МКБ-10). Ответь одним кодом без пояснений."
)
# Fixed before the first run; not tuned on test.
LORA = {"num_layers": 16, "rank": 8, "scale": 20.0, "dropout": 0.0, "lr": 1e-4,
        "batch_size": 4, "epochs": 1, "seed": 0}


def build_messages(query: str, examples: Sequence[Record], codes: Sequence[str]) -> list[dict]:
    parts = ["Разрешённые коды: " + ", ".join(codes)]
    if examples:
        parts.append("Примеры случаев с установленным кодом:")
        parts += [f"Жалобы: {e.text}\nКод: {e.code}" for e in examples]
    parts.append(f"Жалобы пациента: {query}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def prompt_ids(tokenizer, messages: list[dict]) -> list[int]:
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                         enable_thinking=False)
    return tokenizer.encode(text, add_special_tokens=False)


def code_token_ids(tokenizer, codes: Sequence[str]) -> dict[str, list[int]]:
    """Token ids of each code as the start of the assistant turn; all must have equal length."""
    if not codes:
        raise ValueError("codes must be nonempty")
    ids = {c: tokenizer.encode(c, add_special_tokens=False) for c in codes}
    lengths = {len(v) for v in ids.values()}
    if len(lengths) != 1:
        raise ValueError(f"codes tokenize to different lengths: {sorted(lengths)}")
    return ids


def build_trie(code_ids: dict[str, list[int]]) -> dict[tuple[int, ...], set[int]]:
    """Map every proper prefix (including the empty one) to the set of next tokens."""
    trie: dict[tuple[int, ...], set[int]] = defaultdict(set)
    for seq in code_ids.values():
        for i in range(len(seq)):
            trie[tuple(seq[:i])].add(seq[i])
    return dict(trie)


def prediction_key(idx: str, ids: list[int], codes: Sequence[str], adapter_hash: str | None) -> str:
    digest = hashlib.sha256(json.dumps(
        [MODEL_ID, MODEL_REVISION, adapter_hash, ids, list(codes), "trie-float32-v1"]
    ).encode()).hexdigest()
    return f"{idx}:{digest}"


def output_logits(model, hidden):
    if getattr(model, "lm_head", None) is not None:
        return model.lm_head(hidden)
    return model.model.embed_tokens.as_linear(hidden)


def last_logits(model, ids, cache):
    """Logits of the last position only (avoids a vocabulary projection of the whole prompt)."""
    hidden = model.model(ids, cache=cache)
    return output_logits(model, hidden[0, -1])


class Scorer:
    """Score every label-set code as the assistant's answer, reusing the KV cache across prompts.

    Consecutive prompts share a long prefix (system prompt, code list and, for few-shot, the fixed
    examples), so only the differing suffix is prefilled. Code prefixes are expanded one token at
    a time and trimmed back afterwards. Log-probabilities are computed in float32: bf16 logits
    quantize log-probabilities of unlikely codes to steps of 0.125.
    """

    def __init__(self, model, code_ids: dict[str, list[int]]) -> None:
        from mlx_lm.models.cache import make_prompt_cache

        self.model = model
        self.code_ids = code_ids
        self.trie = build_trie(code_ids)
        self.cache = make_prompt_cache(model)
        self.cached: list[int] = []

    def _feed(self, ids: list[int]):
        import mlx.core as mx

        lg = last_logits(self.model, mx.array([ids]), self.cache).astype(mx.float32)
        return lg - mx.logsumexp(lg)

    def _trim(self, n: int) -> None:
        from mlx_lm.models.cache import trim_prompt_cache

        if n > 0:
            trim_prompt_cache(self.cache, n)

    def score(self, prompt: list[int]) -> dict[str, float]:
        import mlx.core as mx

        common = 0
        for a, b in zip(self.cached, prompt, strict=False):
            if a != b:
                break
            common += 1
        common = min(common, len(prompt) - 1)  # always feed at least one token
        self._trim(len(self.cached) - common)
        logprobs: dict[tuple[int, ...], dict[int, float]] = {}

        def record(prefix: tuple[int, ...], lp) -> None:
            nxt = sorted(self.trie[prefix])
            logprobs[prefix] = dict(zip(nxt, lp[mx.array(nxt)].tolist(), strict=True))

        record((), self._feed(prompt[common:]))
        self.cached = list(prompt)

        def expand(prefix: tuple[int, ...]) -> None:
            for tok in sorted(self.trie[prefix]):
                child = (*prefix, tok)
                if child in self.trie:
                    record(child, self._feed([tok]))
                    expand(child)
                    self._trim(1)

        expand(())
        return {code: sum(logprobs[tuple(seq[:i])][seq[i]] for i in range(len(seq)))
                for code, seq in self.code_ids.items()}


def load(adapter: Path | None = None):
    from mlx_lm import load as mlx_load

    model, tokenizer = mlx_load(os.environ.get("RUMED_MODEL_PATH", MODEL_ID),
                                revision=MODEL_REVISION)
    if adapter is not None:
        apply_lora(model)
        model.load_weights(str(adapter), strict=False)
    model.eval()
    return model, tokenizer


def apply_lora(model) -> None:
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model.freeze()
    linear_to_lora_layers(model, LORA["num_layers"],
                          {"rank": LORA["rank"], "scale": LORA["scale"],
                           "dropout": LORA["dropout"]})


def predict(method: str, split: str, train: Sequence[Record], target: Sequence[Record],
            k: int = 15, seed: int = 0) -> tuple[list[list[str]], dict]:
    from rumed_icd.llm import Retriever, examples_for

    if method not in ("zero_shot", "few_shot", "rag", "lora") or split not in ("dev", "test"):
        raise ValueError("expected a local method and dev/test split")
    if not train or not target or k < 1:
        raise ValueError("train, target and k must be nonempty/positive")
    adapter = ROOT / "adapters" / "lora_qwen3_8b.safetensors" if method == "lora" else None
    model, tok = load(adapter)
    codes = sorted({r.code for r in train})
    scorer = Scorer(model, code_token_ids(tok, codes))
    retriever = Retriever(train) if method == "rag" else None
    prompt_method = "zero_shot" if method == "lora" else method

    cache_path = ROOT / "results" / "cache" / f"local_{method}_{split}.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if "key" in row:
                done[row["key"]] = row["top3"]

    t0, n_prompt_tokens = time.perf_counter(), 0
    adapter_hash = hashlib.sha256(adapter.read_bytes()).hexdigest() if adapter else None
    new_predictions = 0
    keys = []
    with cache_path.open("a", encoding="utf-8") as fh:
        for i, rec in enumerate(target):
            ex = examples_for(prompt_method, rec, train, retriever, k, seed)
            ids = prompt_ids(tok, build_messages(rec.text, ex, codes))
            rec_key = prediction_key(rec.idx, ids, codes, adapter_hash)
            keys.append(rec_key)
            if rec_key in done:
                continue
            n_prompt_tokens += len(ids)
            scores = scorer.score(ids)
            top3 = sorted(scores, key=scores.get, reverse=True)[:3]
            done[rec_key] = top3
            new_predictions += 1
            fh.write(json.dumps({"key": rec_key, "idx": rec.idx, "top3": top3}) + "\n")
            fh.flush()
            if i % 50 == 0:
                rate = (time.perf_counter() - t0) / max(1, i + 1)
                print(f"{method}/{split}: {i + 1}/{len(target)}, {rate:.1f} s/example", flush=True)
    preds = [done[key] for key in keys]
    return preds, {"model": MODEL_ID, "scoring": "label-set log-likelihood, top-3",
                   "model_revision": MODEL_REVISION, "adapter_sha256": adapter_hash,
                   "new_predictions": new_predictions,
                   "k_examples": 0 if prompt_method == "zero_shot" else k,
                   "lora": json.loads(adapter.with_suffix(".json").read_text())["config"]
                   if adapter else None}


def train_lora(train: Sequence[Record], dev: Sequence[Record], out: Path,
               n_dev_loss: int = 32, max_steps: int | None = None,
               batch_size: int = 1, accumulation: int = 4, save_every: int = 10) -> dict:
    """LoRA SFT on the zero-shot prompt; loss only on the code tokens and <|im_end|>."""
    import random

    if not train or not dev or min(n_dev_loss, batch_size, accumulation, save_every) < 1:
        raise ValueError("datasets must be nonempty and training counts positive")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite adapter: {out}")

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map
    from mlx_lm.tuner.trainer import grad_checkpoint

    mx.random.seed(LORA["seed"])
    mx.set_cache_limit(128 * 1024**2)
    model, tok = load()
    apply_lora(model)
    grad_checkpoint(model.layers[0])
    model.train()
    codes = sorted({r.code for r in train})
    end_id = tok.convert_tokens_to_ids("<|im_end|>")

    def encode(rec: Record) -> tuple[list[int], int]:
        p = prompt_ids(tok, build_messages(rec.text, [], codes))
        t = tok.encode(rec.code, add_special_tokens=False) + [end_id]
        return p + t, len(t)

    train_set = [encode(r) for r in train]
    code_token_ids(tok, codes)
    dev_indices = random.Random(LORA["seed"]).sample(range(len(dev)), min(n_dev_loss, len(dev)))
    dev_set = [encode(dev[i]) for i in dev_indices]

    def batch_loss(model, inputs, positions, targets):
        # Project only the target positions to the vocabulary: full-sequence logits over a
        # 151k vocabulary would not fit a 16 GB Mac at batch 4.
        hidden = model.model(inputs)
        sel = hidden[mx.arange(hidden.shape[0])[:, None], positions]
        logits = output_logits(model, sel)
        return nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction="mean")

    def make_batch(items):
        # Right padding: causal attention keeps real positions unaffected by the padding.
        n_t = items[0][1]
        width = max(len(s) for s, _ in items) - 1
        inp, pos, tgt = [], [], []
        for seq, n in items:
            assert n == n_t
            x, y = seq[:-1], seq[1:]
            inp.append(x + [0] * (width - len(x)))
            pos.append(list(range(len(x) - n_t, len(x))))
            tgt.append(y[-n_t:])
        return mx.array(inp), mx.array(pos), mx.array(tgt)

    def dev_loss() -> float:
        model.eval()
        tot = 0.0
        for j in range(0, len(dev_set), batch_size):
            items = dev_set[j:j + batch_size]
            tot += batch_loss(model, *make_batch(items)).item() * len(items)
        model.train()
        return tot / len(dev_set)

    opt = optim.Adam(learning_rate=LORA["lr"])
    loss_and_grad = nn.value_and_grad(model, batch_loss)
    rng = random.Random(LORA["seed"])
    config = {**LORA, "batch_size": batch_size, "accumulation": accumulation,
              "grad_checkpoint": True, "max_steps": max_steps, "save_every": save_every}
    log = {"dev_loss_before": dev_loss(), "steps": [], "config": config,
           "model": MODEL_ID, "model_revision": MODEL_REVISION,
           "n_train": len(train_set), "n_dev_loss": len(dev_set),
           "dev_indices": dev_indices, "status": "running",
           "train_records_sha256": hashlib.sha256(json.dumps(
               [(r.idx, r.text, r.code) for r in train], ensure_ascii=False
           ).encode()).hexdigest(),
           "planned_steps": math.ceil(len(train_set) / (batch_size * accumulation))
                            * LORA["epochs"]}
    print("dev loss before:", round(log["dev_loss_before"], 4), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0, step = time.perf_counter(), 0

    def save() -> None:
        temporary = out.with_suffix(".tmp.safetensors")
        mx.save_safetensors(str(temporary), dict(tree_flatten(model.trainable_parameters())))
        temporary.replace(out)
        log.update(completed_steps=step, train_seconds=round(time.perf_counter() - t0),
                   peak_memory_gb=round(mx.get_peak_memory() / 1024**3, 3))
        log_path = out.with_suffix(".json")
        temporary_log = log_path.with_suffix(".tmp.json")
        temporary_log.write_text(json.dumps(log, indent=2) + "\n")
        temporary_log.replace(log_path)

    finished = False
    for _ in range(LORA["epochs"]):
        order = list(range(len(train_set)))
        rng.shuffle(order)
        summed, n_samples, micro_steps, weighted_loss = None, 0, 0, 0.0
        for j in range(0, len(order), batch_size):
            items = [train_set[i] for i in order[j:j + batch_size]]
            batch = make_batch(items)
            loss, grads = loss_and_grad(model, *batch)
            count = len(items)
            weighted = tree_map(lambda g, count=count: g * count, grads)
            summed = weighted if summed is None else tree_map(lambda a, b: a + b, summed, weighted)
            n_samples += len(items)
            micro_steps += 1
            mx.eval(summed, loss)
            if not math.isfinite(loss.item()):
                raise FloatingPointError("nonfinite training loss")
            weighted_loss += loss.item() * len(items)
            if micro_steps < accumulation and j + batch_size < len(order):
                continue
            opt.update(model, tree_map(lambda g, count=n_samples: g / count, summed))
            mx.eval(model.trainable_parameters(), opt.state)
            step_loss = weighted_loss / n_samples
            summed, n_samples, micro_steps, weighted_loss = None, 0, 0, 0.0
            mx.clear_cache()
            step += 1
            if step == 1 or step % 5 == 0:
                log["steps"].append({"step": step, "train_loss": round(step_loss, 4),
                                     "elapsed_s": round(time.perf_counter() - t0)})
                print(log["steps"][-1], flush=True)
            if step % save_every == 0:
                save()
                print(f"checkpoint saved at step {step}", flush=True)
            if max_steps is not None and step >= max_steps:
                finished = True
                break
        if finished:
            break
    log["dev_loss_after"] = dev_loss()
    log["train_seconds"] = round(time.perf_counter() - t0)
    log["status"] = "complete" if max_steps is None else "bounded_run_complete"
    print("dev loss after:", round(log["dev_loss_after"], 4), flush=True)
    save()
    return log
