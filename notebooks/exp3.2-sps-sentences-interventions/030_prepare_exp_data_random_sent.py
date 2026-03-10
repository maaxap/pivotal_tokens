import json
import logging
import random
import time
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from transformers import GenerationConfig

from pivotal_tokens.hf.dataset import Sample
from pivotal_tokens.hf.generation import extract_thinking_trace, generate_batch
from pivotal_tokens.hf.loading import load_model, load_tokenizer
from pivotal_tokens.oracle import RegexOracle


TRACES_PATH = "/home/xaparo00/workspace/projects/pivotal_tokens/data/artifacts/exp1.1.1-qwen3-1.7b-traces.csv"
SPANS_PATH = "/home/xaparo00/workspace/projects/pivotal_tokens/data/experiments/exp3.1-sps-sentences/exp3.1.1-qwen3-1.7b-sps-sentences/data/pivotal_sentences.csv"
OUTPUT_PATH = "/home/xaparo00/workspace/projects/pivotal_tokens/data/experiments/exp3.2-sps-sentences-interventions/exp3.2.2-qwen3-1.7b-sps-rand-sentence-interventions/data/interventions_rand.csv"

MODEL_ID = "Qwen/Qwen3-1.7B"
DEVICE = "cuda"
SYSTEM_PROMPT = "Answer the following question directly, strictly without chain-of-thought after \"</think>\".Keep the answer short (e.g., \"yes\" or \"no\" for binary questions, a person's full name if the question expects a person, a country name if it asks about a country, etc.). Output the answer strictly after the prefix \"Answer: \"  with no extra text."

MAX_TRANSPLANT_ATTEMPTS = 20
MAX_PARAPHRASE_ATTEMPTS = 20
GLOBAL_SEED = 20260303
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def load_inputs(traces_path: str, spans_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load traces and spans CSV files.

    Drops the legacy `metadata` column from spans DataFrame.
    Returns: (traces_df, spans_df).
    """
    traces_df = pd.read_csv(traces_path)
    spans_df = pd.read_csv(spans_path)
    spans_df = spans_df.drop(columns=["metadata"])
    return traces_df, spans_df


def parse_think_wrapped_trace(trace: str) -> dict:
    """Parse think tags when they are present at trace boundaries."""
    stripped = str(trace).strip()
    has_open_tag = stripped.startswith(THINK_OPEN)
    has_close_tag = stripped.endswith(THINK_CLOSE)

    body_text = stripped
    if has_open_tag:
        body_text = body_text[len(THINK_OPEN):]
    if has_close_tag:
        body_text = body_text[:-len(THINK_CLOSE)]

    return {
        "has_open_tag": has_open_tag,
        "has_close_tag": has_close_tag,
        "body_text": body_text.strip(),
    }


def split_body_sentences(body_text: str) -> list[str]:
    """Split body text into sentences using NLTK English sentence tokenizer."""
    text = str(body_text).strip()
    if not text:
        return []
    return sent_tokenize(text, language="english")


def _join_sentences(sentences: list[str]) -> str:
    return " ".join(sentences)


def _normalize_sentence_for_match(text: str) -> str:
    cleaned = str(text).replace(THINK_OPEN, " ").replace(THINK_CLOSE, " ")
    return " ".join(cleaned.split())


def _rebuild_trace(trace_sentences: list[str], has_open_tag: bool, has_close_tag: bool) -> str:
    body_text = _join_sentences(trace_sentences).strip()

    if not has_open_tag and not has_close_tag:
        return body_text

    parts = []
    if has_open_tag:
        parts.append(THINK_OPEN)
    if body_text:
        parts.append(body_text)
    if has_close_tag:
        parts.append(THINK_CLOSE)
    return "\n".join(parts)


def _rebuild_trace_from_base(base_row: dict, trace_sentences: list[str]) -> str:
    return _rebuild_trace(
        trace_sentences=trace_sentences,
        has_open_tag=base_row["trace_has_open_tag"],
        has_close_tag=base_row["trace_has_close_tag"],
    )


def _extract_pivotal_context_preamble(pivotal_context: str) -> str:
    """Return prompt prefix up to and including `<think>` and trailing whitespace."""
    text = str(pivotal_context)
    think_idx = text.find(THINK_OPEN)
    if think_idx == -1:
        return ""

    end_idx = think_idx + len(THINK_OPEN)
    while end_idx < len(text) and text[end_idx].isspace():
        end_idx += 1
    return text[:end_idx]


def _build_pivotal_context_interv(base_row: dict, intervention_payload: dict) -> str:
    """Build intervened pivotal context in the same envelope as original context."""
    metadata = intervention_payload["metadata"]
    prefix_len = metadata.get("pivotal_context_sentence_length_interv")
    if prefix_len is None:
        return base_row["pivotal_context"]

    trace_parts = parse_think_wrapped_trace(intervention_payload["trace_interv"])
    trace_sentences = split_body_sentences(trace_parts["body_text"])
    prefix_len = max(0, min(int(prefix_len), len(trace_sentences)))
    trace_prefix = _join_sentences(trace_sentences[:prefix_len]).strip()

    preamble = _extract_pivotal_context_preamble(base_row["pivotal_context"])
    return f"{preamble}{trace_prefix}"


def _select_random_target_sentence(
    sample_id: str,
    span_id: str,
    trace_sentence_count: int,
    pivotal_idx: int,
    global_seed: int,
) -> tuple[int, str]:
    """Pick a deterministic random target sentence, excluding pivotal when possible."""
    candidates = [idx for idx in range(trace_sentence_count) if idx != pivotal_idx]
    if not candidates:
        return pivotal_idx, "no_alternative_sentence"

    row_seed_bytes = f"{global_seed}|{sample_id}|{span_id}".encode("utf-8")
    row_seed = int.from_bytes(sha256(row_seed_bytes).digest()[:8], byteorder="big", signed=False)
    row_rng = random.Random(row_seed)
    random_idx = row_rng.choice(candidates)
    return random_idx, "ok"


def _build_random_target_context(
    source_pivotal_context: str,
    trace_sentences: list[str],
    random_idx: int,
) -> str:
    """Build context prefix ending right before random target sentence."""
    preamble = _extract_pivotal_context_preamble(source_pivotal_context)
    trace_prefix = _join_sentences(trace_sentences[:random_idx]).strip()
    return f"{preamble}{trace_prefix}"


def _compute_sentence_token_ranges(sentences: list[str], tokenizer) -> tuple[list[list[int]], int]:
    ranges: list[list[int]] = []
    cursor = 0
    for sentence in sentences:
        sent_len = len(tokenizer.encode(sentence, add_special_tokens=False))
        ranges.append([cursor, cursor + sent_len])
        cursor += sent_len
    return ranges, cursor


def _compute_trace_metrics(sentences: list[str], tokenizer, pivot_idx: int | None) -> dict:
    token_ranges, trace_token_length = _compute_sentence_token_ranges(sentences, tokenizer)
    metrics: dict[str, int | list[list[int]] | list[int]] = {
        "trace_sentence_indices": list(range(len(sentences))),
        "trace_token_indices": token_ranges,
        "trace_token_length": trace_token_length,
        "trace_sentence_length": len(sentences),
    }
    if pivot_idx is not None:
        pivot_start, pivot_end = token_ranges[pivot_idx]
        metrics.update(
            {
                "pivot_sentence_position": pivot_idx,
                "pivot_token_position": pivot_start,
                "pivot_token_length": pivot_end - pivot_start,
                "pivotal_context_token_length": pivot_start,
                "pivotal_context_sentence_length": pivot_idx,
            }
        )
    return metrics


def _build_metadata(
    orig_sentence_ids: list[int],
    orig_token_ranges: list[list[int]],
    ids_interv: list[int],
    interv_metrics: dict,
    extra: dict | None = None,
) -> dict:
    metadata = {
        "trace_sentence_indices_original": list(orig_sentence_ids),
        "trace_sentence_indices_interv": list(interv_metrics["trace_sentence_indices"]),
        "trace_token_indices_original": [list(token_range) for token_range in orig_token_ranges],
        "trace_token_indices_interv": [
            list(token_range) for token_range in interv_metrics["trace_token_indices"]
        ],
        "trace_sentence_origin_indices_interv": list(ids_interv),
        "trace_token_length_interv": interv_metrics["trace_token_length"],
        "trace_sentence_length_interv": interv_metrics["trace_sentence_length"],
    }
    if "pivot_sentence_position" in interv_metrics:
        metadata.update(
            {
                "pivot_sentence_position_interv": interv_metrics["pivot_sentence_position"],
                "pivot_token_position_interv": interv_metrics["pivot_token_position"],
                "pivot_token_length_interv": interv_metrics["pivot_token_length"],
                "pivotal_context_token_length_interv": interv_metrics["pivotal_context_token_length"],
                "pivotal_context_sentence_length_interv": interv_metrics["pivotal_context_sentence_length"],
            }
        )
    if extra is not None:
        metadata.update(extra)
    return metadata


def _force_shuffle_prefix_pairs(
    prefix_pairs_original: list[tuple[str, int]],
    rng: random.Random,
) -> tuple[list[tuple[str, int]], str]:
    """Shuffle prefix pairs and ensure order changes when possible."""
    if len(prefix_pairs_original) < 2:
        return list(prefix_pairs_original), "not_enough_sentences"

    original_prefix_ids = [pair[1] for pair in prefix_pairs_original]
    shuffled_pairs = list(prefix_pairs_original)
    for _ in range(10):
        rng.shuffle(shuffled_pairs)
        shuffled_prefix_ids = [pair[1] for pair in shuffled_pairs]
        if shuffled_prefix_ids != original_prefix_ids:
            return shuffled_pairs, "ok"

    # Deterministic fallback to force movement.
    shuffled_pairs = list(prefix_pairs_original)
    shuffled_pairs[0], shuffled_pairs[1] = shuffled_pairs[1], shuffled_pairs[0]
    return shuffled_pairs, "ok"


def _is_invalid_replace_candidate(candidate: str, normalized_original_sentence: str) -> bool:
    if any(x in candidate for x in (THINK_CLOSE, "Answer:", f"{THINK_CLOSE} Answer:", f"{THINK_CLOSE}\nAnswer:", f"{THINK_CLOSE}\n\nAnswer:")):
        return True
    if THINK_OPEN in candidate or THINK_CLOSE in candidate:
        return True
    if _normalize_sentence_for_match(candidate) == normalized_original_sentence:
        return True
    return False


def build_base_rows(spans_df: pd.DataFrame, traces_df: pd.DataFrame) -> list[dict]:
    """Join spans with traces and build base row dicts.

    Each base row contains source fields and resolved `pivot_sentence_position_original`.
    """
    traces_lookup = traces_df.set_index("id")
    spans_with_order = spans_df.copy()
    spans_with_order["_source_order"] = list(range(len(spans_with_order)))

    logger = logging.getLogger(__name__)
    base_rows: list[dict] = []
    for sample_id, group in spans_with_order.groupby("sample_id", sort=False):
        trace_row = traces_lookup.loc[str(sample_id)]
        trace_original = trace_row["trace"]
        trace_parts = parse_think_wrapped_trace(trace_original)
        trace_sentences = split_body_sentences(trace_parts["body_text"])

        match_start = 0
        for row in group.sort_values("_source_order").itertuples(index=False):
            span_text = str(row.span_text)
            span_text_normalized = _normalize_sentence_for_match(span_text)
            pivot_idx = None
            for sent_idx in range(match_start, len(trace_sentences)):
                sentence_normalized = _normalize_sentence_for_match(trace_sentences[sent_idx])
                if sentence_normalized == span_text_normalized:
                    pivot_idx = sent_idx
                    match_start = sent_idx + 1
                    break

            if pivot_idx is None:
                message = (
                    "Failed to match span sentence in trace body for "
                    f"sample_id={sample_id}, span_id={row.span_id}, span_text={span_text!r}"
                )
                logger.error(message)
                raise ValueError(message)

            source_pivotal_context = str(row.pivotal_context)
            random_idx, random_selection_status = _select_random_target_sentence(
                sample_id=str(row.sample_id),
                span_id=str(row.span_id),
                trace_sentence_count=len(trace_sentences),
                pivotal_idx=pivot_idx,
                global_seed=GLOBAL_SEED,
            )
            random_target_sentence = trace_sentences[random_idx]
            random_target_context = _build_random_target_context(
                source_pivotal_context=source_pivotal_context,
                trace_sentences=trace_sentences,
                random_idx=random_idx,
            )

            logger.info(
                "Selected random target sentence sample_id=%s span_id=%s random_idx=%s source_pivotal_idx=%s status=%s",
                row.sample_id,
                row.span_id,
                random_idx,
                pivot_idx,
                random_selection_status,
            )

            base_row = {
                "sample_id": str(row.sample_id),
                "span_id": str(row.span_id),
                "query": str(trace_row["query"]),
                "ground_truth": str(trace_row["ground_truth"]),
                "pivotal_context": random_target_context,
                "span_text": random_target_sentence,
                "source_pivotal_span_text": span_text,
                "source_pivotal_context": source_pivotal_context,
                "prob_before": row.prob_before,
                "prob_after": row.prob_after,
                "prob_delta": row.prob_delta,
                "is_pivotal": bool(row.is_pivotal),
                "trace_original": trace_original,
                "trace_sentences_original": trace_sentences,
                "pivot_sentence_position_original": random_idx,
                "source_pivotal_sentence_position_original": pivot_idx,
                "random_sentence_selection_status": random_selection_status,
                "target_sentence_strategy": "random",
                "trace_has_open_tag": trace_parts["has_open_tag"],
                "trace_has_close_tag": trace_parts["has_close_tag"],
            }
            base_rows.append(base_row)

    base_rows = sorted(base_rows, key=lambda r: (r["sample_id"], r["span_id"]))
    return base_rows


def build_common_original_metrics(base_row: dict, tokenizer) -> dict:
    """Compute and attach original trace metrics.

    Computes original sentence/token positions and lengths required as top-level output columns.
    """
    original_metrics = _compute_trace_metrics(
        sentences=base_row["trace_sentences_original"],
        tokenizer=tokenizer,
        pivot_idx=base_row["pivot_sentence_position_original"],
    )
    base_row["pivot_token_position_original"] = original_metrics["pivot_token_position"]
    base_row["pivot_token_length_original"] = original_metrics["pivot_token_length"]
    base_row["pivotal_context_token_length_original"] = original_metrics["pivotal_context_token_length"]
    base_row["pivotal_context_sentence_length_original"] = original_metrics["pivotal_context_sentence_length"]
    base_row["trace_token_length_original"] = original_metrics["trace_token_length"]
    base_row["trace_sentence_length_original"] = original_metrics["trace_sentence_length"]
    base_row["orig_sentence_ids"] = list(original_metrics["trace_sentence_indices"])
    base_row["orig_token_ranges"] = [list(token_range) for token_range in original_metrics["trace_token_indices"]]
    return base_row


def make_delete_row(base_row: dict, tokenizer) -> dict:
    """Create intervention payload for `delete_sentence`."""
    sents = list(base_row["trace_sentences_original"])
    pivot = base_row["pivot_sentence_position_original"]
    orig_sentence_ids = base_row["orig_sentence_ids"]
    orig_token_ranges = base_row["orig_token_ranges"]
    del sents[pivot]
    next_sentence = sents[pivot] if pivot < len(sents) else ""
    ids_interv = [sid for sid in orig_sentence_ids if sid != pivot]

    interv_metrics = _compute_trace_metrics(sents, tokenizer=tokenizer, pivot_idx=None)
    metadata = _build_metadata(
        orig_sentence_ids=orig_sentence_ids,
        orig_token_ranges=orig_token_ranges,
        ids_interv=ids_interv,
        interv_metrics=interv_metrics,
    )

    return {
        "intervention_type": "delete_sentence",
        "trace_interv": _rebuild_trace_from_base(base_row=base_row, trace_sentences=sents),
        "span_text_interv": next_sentence,
        "metadata": metadata,
    }


def make_shuffle_row(base_row: dict, rng: random.Random, tokenizer) -> dict:
    """Create intervention payload for `shuffle_preceding_context`.

    Shuffles only sentences before target index.
    """
    sents = list(base_row["trace_sentences_original"])
    pivot = base_row["pivot_sentence_position_original"]
    orig_sentence_ids = base_row["orig_sentence_ids"]
    orig_token_ranges = base_row["orig_token_ranges"]

    prefix_pairs_original = list(zip(sents[:pivot], orig_sentence_ids[:pivot]))
    prefix_pairs, shuffle_status = _force_shuffle_prefix_pairs(prefix_pairs_original=prefix_pairs_original, rng=rng)
    prefix_sents = [pair[0] for pair in prefix_pairs]
    prefix_ids = [pair[1] for pair in prefix_pairs]
    sents_interv = prefix_sents + sents[pivot:]
    ids_interv = prefix_ids + orig_sentence_ids[pivot:]

    interv_metrics = _compute_trace_metrics(sents_interv, tokenizer=tokenizer, pivot_idx=pivot)
    metadata = _build_metadata(
        orig_sentence_ids=orig_sentence_ids,
        orig_token_ranges=orig_token_ranges,
        ids_interv=ids_interv,
        interv_metrics=interv_metrics,
        extra={"shuffle_status": shuffle_status},
    )

    return {
        "intervention_type": "shuffle_preceding_context",
        "trace_interv": _rebuild_trace_from_base(base_row=base_row, trace_sentences=sents_interv),
        "span_text_interv": base_row["span_text"],
        "metadata": metadata,
    }


def make_replace_probable_row(base_row: dict, model, tokenizer, max_attempts=MAX_PARAPHRASE_ATTEMPTS) -> dict:
    """Create intervention payload for `replace_with_probable`.

    Generates deterministic continuation and uses first generated sentence as replacement.
    """
    sents = list(base_row["trace_sentences_original"])
    pivot = base_row["pivot_sentence_position_original"]
    orig_sentence_ids = base_row["orig_sentence_ids"]
    orig_token_ranges = base_row["orig_token_ranges"]
    pivotal_context = base_row["pivotal_context"]
    original_sentence = sents[pivot]
    normalized_original_sentence = _normalize_sentence_for_match(original_sentence)
    replacement_config = GenerationConfig(
        max_new_tokens=4096,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    replacement_sentence = ""
    for _ in range(max_attempts):
        enc = tokenizer(pivotal_context, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=replacement_config,
        )

        continuation_ids = out[:, input_ids.shape[1]:]
        continuation = tokenizer.decode(continuation_ids[0], skip_special_tokens=True).strip()
        continuation_sentences = split_body_sentences(continuation)

        if not continuation_sentences:
            continue
        candidate = continuation_sentences[0].strip()
        if not candidate:
            continue
        if _is_invalid_replace_candidate(
            candidate=candidate,
            normalized_original_sentence=normalized_original_sentence,
        ):
            continue

        replacement_sentence = candidate
        break

    if replacement_sentence and _normalize_sentence_for_match(replacement_sentence) == normalized_original_sentence:
        replacement_sentence = ""

    if not replacement_sentence:
        ids_interv = list(orig_sentence_ids)
        interv_metrics = _compute_trace_metrics(sents, tokenizer=tokenizer, pivot_idx=pivot)
        metadata = _build_metadata(
            orig_sentence_ids=orig_sentence_ids,
            orig_token_ranges=orig_token_ranges,
            ids_interv=ids_interv,
            interv_metrics=interv_metrics,
            extra={"replace_with_probable_status": "replacement_not_found"},
        )
        return {
            "intervention_type": "replace_with_probable",
            "trace_interv": base_row["trace_original"],
            "span_text_interv": base_row["span_text"],
            "metadata": metadata,
        }

    sents[pivot] = replacement_sentence
    ids_interv = list(orig_sentence_ids)
    interv_metrics = _compute_trace_metrics(sents, tokenizer=tokenizer, pivot_idx=pivot)
    metadata = _build_metadata(
        orig_sentence_ids=orig_sentence_ids,
        orig_token_ranges=orig_token_ranges,
        ids_interv=ids_interv,
        interv_metrics=interv_metrics,
        extra={"replace_with_probable_status": "ok"},
    )

    return {
        "intervention_type": "replace_with_probable",
        "trace_interv": _rebuild_trace_from_base(base_row=base_row, trace_sentences=sents),
        "span_text_interv": replacement_sentence,
        "metadata": metadata,
    }


def make_replace_noise_row(base_row: dict, tokenizer, rng: random.Random) -> dict:
    """Create intervention payload for `replace_with_noise`.

    Builds random-token replacement sentence with similar token length.
    """
    sents = list(base_row["trace_sentences_original"])
    pivot = base_row["pivot_sentence_position_original"]
    orig_sentence_ids = base_row["orig_sentence_ids"]
    orig_token_ranges = base_row["orig_token_ranges"]
    k = base_row["pivot_token_length_original"]

    vocab_size = tokenizer.vocab_size
    special_ids = set(tokenizer.all_special_ids)

    noise_sentence = ""
    while True:
        token_ids = []
        while len(token_ids) < k:
            token_id = rng.randrange(vocab_size)
            if token_id not in special_ids:
                token_ids.append(token_id)

        candidate = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        if not candidate:
            continue
        if candidate[-1] not in ".!?":
            candidate = f"{candidate}."

        noise_sentence = candidate
        break

    sents[pivot] = noise_sentence
    ids_interv = list(orig_sentence_ids)
    interv_metrics = _compute_trace_metrics(sents, tokenizer=tokenizer, pivot_idx=pivot)
    metadata = _build_metadata(
        orig_sentence_ids=orig_sentence_ids,
        orig_token_ranges=orig_token_ranges,
        ids_interv=ids_interv,
        interv_metrics=interv_metrics,
    )

    return {
        "intervention_type": "replace_with_noise",
        "trace_interv": _rebuild_trace_from_base(base_row=base_row, trace_sentences=sents),
        "span_text_interv": noise_sentence,
        "metadata": metadata,
    }


def make_move_rows(base_row: dict, tokenizer) -> list[dict]:
    """Create three payloads for move interventions.

    Returns payloads for beginning, middle, and end variants.
    """
    sents_original = list(base_row["trace_sentences_original"])
    pivot = base_row["pivot_sentence_position_original"]
    orig_sentence_ids = base_row["orig_sentence_ids"]
    orig_token_ranges = base_row["orig_token_ranges"]
    moved_sentence = sents_original[pivot]
    moved_id = orig_sentence_ids[pivot]
    remaining_sentences = [sentence for i, sentence in enumerate(sents_original) if i != pivot]
    remaining_ids = [sid for sid in orig_sentence_ids if sid != pivot]

    rows = []
    variants = [
        ("beginning", "move_sentence_beginning", 0),
        ("middle", "move_sentence_middle", len(remaining_sentences) // 2),
        ("end", "move_sentence_end", len(remaining_sentences)),
    ]
    for variant, intervention_type, destination_idx in variants:
        sents_interv = list(remaining_sentences)
        ids_interv = list(remaining_ids)
        sents_interv.insert(destination_idx, moved_sentence)
        ids_interv.insert(destination_idx, moved_id)

        interv_metrics = _compute_trace_metrics(sents_interv, tokenizer=tokenizer, pivot_idx=destination_idx)
        metadata = _build_metadata(
            orig_sentence_ids=orig_sentence_ids,
            orig_token_ranges=orig_token_ranges,
            ids_interv=ids_interv,
            interv_metrics=interv_metrics,
            extra={"intervention_variant": variant},
        )

        rows.append(
            {
                "intervention_type": intervention_type,
                "trace_interv": _rebuild_trace_from_base(base_row=base_row, trace_sentences=sents_interv),
                "span_text_interv": base_row["span_text"],
                "metadata": metadata,
            }
        )

    return rows


def generate_unsuccessful_trace(
    sample_id: str,
    query: str,
    ground_truth: str,
    system_prompt: str,
    generation_config: GenerationConfig,
    model,
    tokenizer,
    oracle,
    max_attempts: int,
) -> str | None:
    """Generate donor trace whose completion is incorrect for `ground_truth`.

    Implementation contract:
    - Build one `Sample` from the provided fields.
    - Use `generate_batch` to produce completion text per attempt.
    - Use `extract_thinking_trace` to extract trace candidate.
    - Use oracle on the full completion text to decide success/failure.
    - Return first extracted trace from an unsuccessful completion.
    - Return `None` if no unsuccessful trace with extractable thinking trace is found.

    Returns:
      - donor_trace (string) when found, otherwise `None`.
    """
    sample = Sample(id=sample_id, query=query, ground_truth=ground_truth, metadata={})

    for _ in range(max_attempts):
        generated_text = generate_batch(
            samples=[sample],
            system_prompt=system_prompt,
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            enable_thinking=True,
        )[0]

        trace_candidate = extract_thinking_trace(generated_text)
        is_success = oracle.verify(actual=generated_text, expected=[ground_truth])

        if trace_candidate is not None and not is_success:
            return trace_candidate

    return None


def make_transplant_row(
    base_row: dict,
    system_prompt: str,
    generation_config: GenerationConfig,
    model,
    tokenizer,
    oracle,
) -> dict:
    """Create intervention payload for `transplant_sentence`.

    Uses `generate_unsuccessful_trace` and inserts the original pivotal sentence
    into donor trace sentences.
    Falls back to unchanged-trace payload if donor trace is unavailable.
    """
    pivot = base_row["pivot_sentence_position_original"]
    sents = list(base_row["trace_sentences_original"])
    original_sentence = sents[pivot]
    orig_sentence_ids = base_row["orig_sentence_ids"]
    orig_token_ranges = base_row["orig_token_ranges"]
    donor_trace_found = False
    last_donor_trace = None
    last_insert_idx = None
    last_donor_trace_interv = None
    original_len = base_row["trace_sentence_length_original"]

    for _ in range(MAX_TRANSPLANT_ATTEMPTS):
        donor_trace = generate_unsuccessful_trace(
            sample_id=base_row["sample_id"],
            query=base_row["query"],
            ground_truth=base_row["ground_truth"],
            system_prompt=system_prompt,
            generation_config=generation_config,
            model=model,
            tokenizer=tokenizer,
            oracle=oracle,
            max_attempts=1,
        )
        if donor_trace is None:
            continue

        donor_trace_found = True
        donor_parts = parse_think_wrapped_trace(donor_trace)
        donor_sentences = split_body_sentences(donor_parts["body_text"])
        donor_sentence_count = len(donor_sentences)
        rel = pivot / max(original_len - 1, 1)
        insert_idx = round(rel * donor_sentence_count)
        insert_idx = max(0, min(insert_idx, donor_sentence_count))

        trial_sentences = (
            donor_sentences[:insert_idx] + [original_sentence] + donor_sentences[insert_idx:]
        )
        trial_trace_interv = _rebuild_trace(
            trace_sentences=trial_sentences,
            has_open_tag=donor_parts["has_open_tag"],
            has_close_tag=donor_parts["has_close_tag"],
        )

        last_donor_trace = donor_trace
        last_insert_idx = insert_idx
        last_donor_trace_interv = trial_trace_interv

        if trial_trace_interv == base_row["trace_original"]:
            continue

        ids_interv: list[int] = []
        if orig_sentence_ids:
            original_last_idx = len(orig_sentence_ids) - 1
            donor_last_idx = max(donor_sentence_count - 1, 1)
            for pos in range(len(trial_sentences)):
                if pos == insert_idx:
                    ids_interv.append(orig_sentence_ids[pivot])
                    continue
                donor_pos = pos if pos < insert_idx else pos - 1
                mapped_orig_idx = round((donor_pos * original_last_idx) / donor_last_idx)
                mapped_orig_idx = max(0, min(mapped_orig_idx, original_last_idx))
                ids_interv.append(orig_sentence_ids[mapped_orig_idx])

        interv_metrics = _compute_trace_metrics(trial_sentences, tokenizer=tokenizer, pivot_idx=insert_idx)
        metadata = _build_metadata(
            orig_sentence_ids=orig_sentence_ids,
            orig_token_ranges=orig_token_ranges,
            ids_interv=ids_interv,
            interv_metrics=interv_metrics,
            extra={
                "transplant_donor_trace": donor_trace,
                "transplant_insert_position": insert_idx,
                "transplant_inserted_sentence_text": original_sentence,
                "transplant_status": "ok",
            },
        )
        return {
            "intervention_type": "transplant_sentence",
            "trace_interv": trial_trace_interv,
            "span_text_interv": original_sentence,
            "metadata": metadata,
        }

    if not donor_trace_found:
        ids_interv = list(orig_sentence_ids)
        interv_metrics = _compute_trace_metrics(sents, tokenizer=tokenizer, pivot_idx=pivot)
        metadata = _build_metadata(
            orig_sentence_ids=orig_sentence_ids,
            orig_token_ranges=orig_token_ranges,
            ids_interv=ids_interv,
            interv_metrics=interv_metrics,
            extra={"transplant_status": "unsuccessful_trace_not_found"},
        )
        return {
            "intervention_type": "transplant_sentence",
            "trace_interv": base_row["trace_original"],
            "span_text_interv": base_row["span_text"],
            "metadata": metadata,
        }

    ids_interv = list(orig_sentence_ids)
    interv_metrics = _compute_trace_metrics(sents, tokenizer=tokenizer, pivot_idx=pivot)
    metadata = _build_metadata(
        orig_sentence_ids=orig_sentence_ids,
        orig_token_ranges=orig_token_ranges,
        ids_interv=ids_interv,
        interv_metrics=interv_metrics,
        extra={
            "transplant_donor_trace": last_donor_trace,
            "transplant_insert_position": last_insert_idx,
            "transplant_inserted_sentence_text": original_sentence,
            "transplant_status": "unsuccessful_changed_trace_not_found",
        },
    )
    return {
        "intervention_type": "transplant_sentence",
        "trace_interv": last_donor_trace_interv if last_donor_trace_interv is not None else base_row["trace_original"],
        "span_text_interv": original_sentence,
        "metadata": metadata,
    }


def assemble_output_row(base_row: dict, intervention_payload: dict) -> dict:
    """Assemble final CSV row.

    Writes common fields to top-level columns and serializes intervention-specific fields into `metadata`.
    Ensures `span_text_interv` is a top-level output column.
    """
    intervention_type = intervention_payload["intervention_type"]
    target_sentence_strategy = base_row["target_sentence_strategy"]
    source_pivotal_span_text = base_row["source_pivotal_span_text"]
    source_pivotal_sentence_position_original = base_row["source_pivotal_sentence_position_original"]
    source_pivotal_context = base_row["source_pivotal_context"]
    random_sentence_selection_status = base_row["random_sentence_selection_status"]

    metadata = dict(intervention_payload["metadata"])
    metadata.update(
        {
            "target_sentence_strategy": target_sentence_strategy,
            "source_pivotal_span_text": source_pivotal_span_text,
            "source_pivotal_sentence_position_original": source_pivotal_sentence_position_original,
            "source_pivotal_context": source_pivotal_context,
            "random_sentence_selection_status": random_sentence_selection_status,
        }
    )
    pivotal_context_interv = _build_pivotal_context_interv(
        base_row=base_row,
        intervention_payload=intervention_payload,
    )
    return {
        "intervention_id": f"{base_row['sample_id']}:{base_row['span_id']}:{intervention_type}",
        "sample_id": base_row["sample_id"],
        "span_id": base_row["span_id"],
        "span_ids": json.dumps([base_row["span_id"]]),
        "query": base_row["query"],
        "ground_truth": base_row["ground_truth"],
        "target_sentence_strategy": target_sentence_strategy,
        "pivotal_context": base_row["pivotal_context"],
        "pivotal_context_interv": pivotal_context_interv,
        "span_text": base_row["span_text"],
        "span_text_interv": intervention_payload["span_text_interv"],
        "source_pivotal_span_text": source_pivotal_span_text,
        "source_pivotal_sentence_position_original": source_pivotal_sentence_position_original,
        "source_pivotal_context": source_pivotal_context,
        "random_sentence_selection_status": random_sentence_selection_status,
        "prob_before": base_row["prob_before"],
        "prob_after": base_row["prob_after"],
        "prob_delta": base_row["prob_delta"],
        "is_pivotal": base_row["is_pivotal"],
        "trace_original": base_row["trace_original"],
        "trace_interv": intervention_payload["trace_interv"],
        "intervention_type": intervention_type,
        "pivot_token_position_original": base_row["pivot_token_position_original"],
        "pivot_sentence_position_original": base_row["pivot_sentence_position_original"],
        "pivot_token_length_original": base_row["pivot_token_length_original"],
        "pivotal_context_token_length_original": base_row["pivotal_context_token_length_original"],
        "pivotal_context_sentence_length_original": base_row["pivotal_context_sentence_length_original"],
        "trace_token_length_original": base_row["trace_token_length_original"],
        "trace_sentence_length_original": base_row["trace_sentence_length_original"],
        "metadata": json.dumps(metadata),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    rng = random.Random(GLOBAL_SEED)

    traces_df, spans_df = load_inputs(traces_path=TRACES_PATH, spans_path=SPANS_PATH)

    model = load_model(model_id=MODEL_ID, device=DEVICE)
    tokenizer = load_tokenizer(model_id=MODEL_ID)
    model.eval()

    generation_config = GenerationConfig(
        max_new_tokens=4096,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    oracle = RegexOracle(fuzzy_match_threshold=0.7)

    base_rows = build_base_rows(spans_df=spans_df, traces_df=traces_df)

    for row in base_rows:
        build_common_original_metrics(base_row=row, tokenizer=tokenizer)

    output_rows: list[dict] = []
    total_rows = len(base_rows)
    for row_idx, row in enumerate(base_rows, start=1):
        row_start = time.time()
        payloads: list[dict] = []

        logger.info(f"Processing row {row_idx}/{total_rows}")

        logger.info("Running delete")
        start = time.time()
        payloads.append(make_delete_row(base_row=row, tokenizer=tokenizer))
        elapsed = time.time() - start
        logger.info("Finished delete in {:.2f}s".format(elapsed))

        logger.info("Running shuffle")
        start = time.time()
        payloads.append(make_shuffle_row(base_row=row, rng=rng, tokenizer=tokenizer))
        elapsed = time.time() - start
        logger.info("Finished shuffle in {:.2f}s".format(elapsed))

        logger.info("Running replace_probable")
        start = time.time()
        payloads.append(make_replace_probable_row(base_row=row, model=model, tokenizer=tokenizer))
        elapsed = time.time() - start
        logger.info(
            "Finished replace_probable in {:.2f}s".format(elapsed)
        )

        logger.info("Running replace_noise")
        start = time.time()
        payloads.append(make_replace_noise_row(base_row=row, tokenizer=tokenizer, rng=rng))
        elapsed = time.time() - start
        logger.info("Finished replace_noise in {:.2f}s".format(elapsed))

        logger.info("Running move")
        start = time.time()
        payloads.extend(make_move_rows(base_row=row, tokenizer=tokenizer))
        elapsed = time.time() - start
        logger.info("Finished move in {:.2f}s".format(elapsed))

        logger.info("Running transplant")
        start = time.time()
        payloads.append(
            make_transplant_row(
                base_row=row,
                system_prompt=SYSTEM_PROMPT,
                generation_config=generation_config,
                model=model,
                tokenizer=tokenizer,
                oracle=oracle,
            )
        )
        elapsed = time.time() - start
        logger.info("Finished transplant in {:.2f}s".format(elapsed))

        output_rows.extend(assemble_output_row(base_row=row, intervention_payload=payload) for payload in payloads)
        elapsed = time.time() - row_start
        logger.info(
            "Finished row in {:.2f}s (generated {} interventions)".format(
                elapsed, len(payloads)
            )
        )

    output_df = pd.DataFrame(output_rows)
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
