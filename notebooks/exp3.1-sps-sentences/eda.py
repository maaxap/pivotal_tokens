# %% [markdown]
# EDA for thinking traces: sentence stats, formatting quality, and split heuristics.

# %%
import re
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px

from pivotal_tokens.constants import get_artifacts_dir

# %%
DATA_PATH = get_artifacts_dir() / "exp1.1.1-qwen3-1.7b-traces.csv"

# %%
df = pd.read_csv(DATA_PATH)
df.head()

# %% [markdown]
# Basic cleanup: drop `<think>` tags and normalize whitespace. Keep originals for comparison.

# %%
THINK_TAG_RE = re.compile(r"</?think>")
WHITESPACE_RE = re.compile(r"[ \t]+")


def clean_trace(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = THINK_TAG_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


df["trace_raw"] = df["trace"]
df["trace"] = df["trace_raw"].map(clean_trace)
df[["trace_raw", "trace"]].head()

# %% [markdown]
# Sentence splitting (heuristic). This is intentionally simple and conservative.

# %%
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
QUOTE_CLEAN_RE = re.compile(r"^[\"'“”‘’]+|[\"'“”‘’]+$")


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = SENT_SPLIT_RE.split(text)
    sentences = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        part = QUOTE_CLEAN_RE.sub("", part).strip()
        if part:
            sentences.append(part)
    return sentences


df["sentences"] = df["trace"].map(split_sentences)
df["num_sentences"] = df["sentences"].map(len)
df[["num_sentences", "sentences"]].head()

# %% [markdown]
# Sentence count distribution per trace.

# %%
df["num_sentences"].describe()

# %%
df["num_sentences"].value_counts().head(20)

# %%
fig = px.histogram(df, x="num_sentences", nbins=30, title="Sentences per trace")
fig.show()

# %% [markdown]
# Formatting checks: empty traces, traces without sentence-ending punctuation, excessive newlines.

# %%
def count_newlines(text: str) -> int:
    return text.count("\n") if text else 0


def has_sentence_punct(text: str) -> bool:
    return bool(re.search(r"[.!?]", text)) if text else False


df["is_empty"] = df["trace"].eq("")
df["num_newlines"] = df["trace"].map(count_newlines)
df["has_sentence_punct"] = df["trace"].map(has_sentence_punct)

formatting_stats = pd.Series(
    {
        "empty_trace_pct": df["is_empty"].mean(),
        "no_sentence_punct_pct": (~df["has_sentence_punct"]).mean(),
        "avg_newlines": df["num_newlines"].mean(),
        "median_newlines": df["num_newlines"].median(),
    }
)
formatting_stats

# %%
df.loc[~df["has_sentence_punct"], ["id", "trace"]].head()

# %% [markdown]
# Sentence length and complexity heuristics:
# - token length (words)
# - character length
# - clause markers (commas/semicolons/colons/dashes)
# - conjunction count (simple heuristic)

# %%
CONJ_RE = re.compile(
    r"\\b(and|but|or|nor|yet|so|because|although|though|since|while|when|if|that|which|who)\\b",
    re.IGNORECASE,
)
CLAUSE_MARK_RE = re.compile(r"[,:;—-]")


def sentence_stats(sentences: Iterable[str]) -> pd.DataFrame:
    rows = []
    for s in sentences:
        words = re.findall(r"\\b\\w+\\b", s)
        rows.append(
            {
                "sentence": s,
                "num_words": len(words),
                "num_chars": len(s),
                "num_clause_marks": len(CLAUSE_MARK_RE.findall(s)),
                "num_conjunctions": len(CONJ_RE.findall(s)),
            }
        )
    return pd.DataFrame(rows)


sent_df = sentence_stats(s for sentences in df["sentences"] for s in sentences)
sent_df.describe()

# %%
melted_sent_df = sent_df[["num_words", "num_chars"]].melt(value_vars=["num_words", "num_chars"])
melted_sent_df.head()

# %%
fig = px.histogram(melted_sent_df, x="value", color="variable", barmode="group", nbins=40)
fig.show()

# %%
sent_df["num_clause_marks"].value_counts().head(10)

# %%
sent_df["num_conjunctions"].value_counts().head(10)

# %% [markdown]
# Identify sentences that might need further splitting.

# %%
LONG_WORDS = 40
LONG_CHARS = 220
MANY_CLAUSE_MARKS = 4
MANY_CONJUNCTIONS = 3

sent_df["needs_split"] = (
    (sent_df["num_words"] >= LONG_WORDS)
    | (sent_df["num_chars"] >= LONG_CHARS)
    | (sent_df["num_clause_marks"] >= MANY_CLAUSE_MARKS)
    | (sent_df["num_conjunctions"] >= MANY_CONJUNCTIONS)
)

sent_df["needs_split"].mean()

# %%
sent_df.loc[sent_df["needs_split"], "sentence"].head(20).tolist()

# %% [markdown]
# Aggregate: how many "complex" sentences per trace.

# %%
df["num_complex_sentences"] = df["sentences"].map(
    lambda sents: sum(
        (
            (len(re.findall(r"\\b\\w+\\b", s)) >= LONG_WORDS)
            or (len(s) >= LONG_CHARS)
            or (len(CLAUSE_MARK_RE.findall(s)) >= MANY_CLAUSE_MARKS)
            or (len(CONJ_RE.findall(s)) >= MANY_CONJUNCTIONS)
        )
        for s in sents
    )
)
df["num_complex_sentences"].describe()

# %%
fig = px.histogram(df, x="num_complex_sentences", nbins=30, title="Complex sentences per trace")
fig.show()

# %% [markdown]
# Quick qualitative checks: sample traces and long/complex sentences.

# %%
df.sample(5, random_state=7)[["id", "trace"]]

# %%
sent_df.sort_values(["num_words", "num_clause_marks"], ascending=False).head(10)

# %% [markdown]
# If you decide to split further, a simple second-pass splitter can break long sentences on
# clause markers while keeping short fragments grouped.

# %%
def split_long_sentence(s: str, max_words: int = 30) -> list[str]:
    words = re.findall(r"\\b\\w+\\b", s)
    if len(words) <= max_words:
        return [s]
    parts = re.split(r"([,:;—-])", s)
    chunks = []
    buf = ""
    for part in parts:
        buf += part
        if part in {",", ";", ":", "—", "-"}:
            if len(re.findall(r"\\b\\w+\\b", buf)) >= max_words:
                chunks.append(buf.strip())
                buf = ""
    if buf.strip():
        chunks.append(buf.strip())
    return [c for c in chunks if c]


split_examples = sent_df.loc[sent_df["needs_split"], "sentence"].head(20)
for s in split_examples:
    print("ORIG:", s)
    print("SPLIT:", split_long_sentence(s))
    print("---")


# %%
