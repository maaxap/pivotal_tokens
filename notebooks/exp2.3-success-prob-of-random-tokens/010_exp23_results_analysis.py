# %%
import ast
import json

import numpy as np
import pandas as pd
import plotly.express as px
from IPython.display import HTML, display
from transformers import PreTrainedTokenizer

from pivotal_tokens.constants import get_artifacts_dir, get_data_dir
from pivotal_tokens.hf.loading import load_tokenizer


EXP22_DIR = get_data_dir() / "experiments" / "exp2.2-random-tokens-on-pivotal-positions" / "exp2.2.1-qwen3-1.7b-random-tokens-on-pivotal-positions"
EXP22_RESULTS_FILE = EXP22_DIR / "data" / "alternative_tokens.csv"



EXP23_DIR = get_data_dir() / "experiments" / "exp2.3-success-prob-of-random-tokens" / "exp2.3.1-qwen3-1.7b-success-prob-of-random-tokens"
EXP23_RESULTS_FILE = EXP23_DIR / "data" / "random_tokens.csv"

TRACES_FILE = get_artifacts_dir() / "exp1.1.1-qwen3-1.7b-traces.csv"

QWEN3_1_7B_MODEL_ID = "Qwen/Qwen3-1.7B"

# %%

tokenizer = load_tokenizer(QWEN3_1_7B_MODEL_ID)

traces_df = pd.read_csv(TRACES_FILE)
prob_init_df = pd.read_csv(EXP22_RESULTS_FILE)
prob_init_df = prob_init_df[["sample_id", "prob_init"]].rename(columns={"sample_id": "id"}).copy()
traces_df = pd.merge(traces_df, prob_init_df, on="id", how="inner")

tokens_df = pd.read_csv(EXP23_RESULTS_FILE)

tokens_df[:2]

# %%

df = pd.merge(traces_df.drop(columns=["query", "ground_truth"]), tokens_df, left_on="id", right_on="sample_id", how="inner")
df = df.drop_duplicates()

df["token_ids_random"] = df["token_ids_random"].apply(json.loads)
df["token_ids_alt"] = df["token_ids_alt"].apply(json.loads)

# %%

def get_thinking_trace_from_completion(completion: str) -> str:
    prefix_split_str = "<|im_start|>assistant\n"
    completion_with_suffix = completion.split(prefix_split_str)[-1].strip()

    suffix_split_str = "<|im_end|>"
    completion = completion_with_suffix.split(suffix_split_str)[0].strip()
    return completion


def get_token_length(text: str, tokenizer: PreTrainedTokenizer) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return len(token_ids)


df["partial_trace"] = df["pivotal_context"].apply(get_thinking_trace_from_completion)
df["partial_trace_token_length"] = df["partial_trace"].apply(
    lambda x: get_token_length(x, tokenizer)
)
df["trace_token_length"] = df["trace"].apply(lambda x: get_token_length(x, tokenizer))
df["pivotal_token_relative_position"] = (
    (df["partial_trace_token_length"] + 1) / df["trace_token_length"]
)

assert (df["partial_trace_token_length"] < df["trace_token_length"]).mean() == 1.0

# %%

df = df[df["span_text_alt"].notna()]
df["prob_delta_alt"] = df["prob_after_alt"] - df["prob_before"]

df["prob_before_normalized"] = df["prob_before"] / df["prob_init"]
df["prob_after_normalized"] = df["prob_after"] / df["prob_init"]
df["prob_delta_normalized"] = df["prob_delta"] / df["prob_init"]

df["prob_before_alt_normalized"] = df["prob_before"] / df["prob_init"]
df["prob_after_alt_normalized"] = df["prob_after_alt"] / df["prob_init"]
df["prob_delta_alt_normalized"] = df["prob_delta_alt"] / df["prob_init"]

df["is_positive"] = df["prob_delta"] >= 0
df["is_positive_alt"] = df["prob_delta_alt"] >= 0

# %%

unique_samples = df["sample_id"].nunique()
num_pivotal_tokens = len(df)
# num_positive = df["is_positive"].sum()
# num_positive_alt = df["is_positive_alt"].sum()
print(
    "Overall statistics:\n"
    f"- Unique samples: {unique_samples}\n"
    f"- Pivotal tokens: {num_pivotal_tokens}\n"
    # f"- Positive pivotal tokens: {num_positive}\n"
    # f"- Positive pivotal tokens (alt): {num_positive_alt}"
)

# %%

df.columns


# %%

hist_delta = px.histogram(
    df,
    x="prob_delta",
    nbins=60,
    title="ΔP(success) after pivotal token",
)
hist_delta.update_layout(xaxis_title="ΔP(success)", yaxis_title="Count")


# %%


scatter_rel_delta_random = px.scatter(
    df.drop_duplicates(["sample_id", "span_id"]),
    x="pivotal_token_relative_position",
    y="prob_delta_normalized",
    opacity=0.35,
    render_mode="webgl",
    title="Normalized ΔP(success) vs relative position (random token)",
)
scatter_rel_delta_random.update_layout(
    xaxis_title="Relative position in trace",
    yaxis_title="Normalized ΔP(success)",
)



# %%

alt_types = sorted(df["alt_type"].dropna().unique().tolist())
scatter_rel_delta_alt_by_type = {}
for alt_type in alt_types:
    df_alt = df[df["alt_type"] == alt_type]
    fig = px.scatter(
        df_alt,
        x="pivotal_token_relative_position",
        y="prob_delta_alt_normalized",
        opacity=0.35,
        render_mode="webgl",
        title=f"Normalized ΔP(success) vs relative position (alt type={alt_type})",
    )
    fig.update_layout(
        xaxis_title="Relative position in trace",
        yaxis_title="Normalized ΔP(success)",
    )
    fig.show()

# %%

scatter_rel_after_random = px.scatter(
    df.drop_duplicates(["sample_id", "span_id"]),
    x="pivotal_token_relative_position",
    y="prob_after_normalized",
    opacity=0.35,
    render_mode="webgl",
    title="Normalized P(success after) vs relative position (random token)",
)
scatter_rel_after_random.update_layout(
    xaxis_title="Relative position in trace",
    yaxis_title="Normalized P(success after)",
)

# %%

scatter_rel_after_alt_by_type = {}
for alt_type in alt_types:
    df_alt = df[df["alt_type"] == alt_type]
    fig = px.scatter(
        df_alt,
        x="pivotal_token_relative_position",
        y="prob_after_alt_normalized",
        opacity=0.35,
        render_mode="webgl",
        title=f"Normalized P(success after) vs relative position (alt={alt_type})",
    )
    fig.update_layout(
        xaxis_title="Relative position in trace",
        yaxis_title="Normalized P(success after)",
    )
    fig.show()

# %%

quantile_bins = pd.qcut(df["prob_init"], q=5, duplicates="drop")
intervals = sorted(quantile_bins.cat.categories, key=lambda x: x.left)
quantile_bins = quantile_bins.cat.reorder_categories(intervals, ordered=True)
labels = [f"{interval.left:.2f} - {interval.right:.2f}" for interval in intervals]
df["prob_init_quantile_range"] = quantile_bins.cat.rename_categories(labels)
df_box = df.sort_values(by="prob_init_quantile_range")

box_delta_by_q = px.box(
    df_box,
    x="prob_init_quantile_range",
    y="prob_delta",
    title="ΔP(success) by initial probability quantiles (P0)",
)
box_delta_by_q.update_layout(xaxis_title="P0 range", yaxis_title="ΔP(success)")

# %%

top30_mean_prob_delta = (
    df.groupby("span_text_random", dropna=False)["prob_delta"]
    .mean()
    .sort_values(ascending=False)
    .head(30)
    .reset_index()
    .rename(columns={"span_text_random": "token", "prob_delta": "mean_prob_delta"})
)
display(HTML(top30_mean_prob_delta.to_html(index=False)))

# %%

last30_mean_prob_delta = (
    df.groupby("span_text_random", dropna=False)["prob_delta"]
    .mean()
    .sort_values(ascending=True)
    .head(30)
    .reset_index()
    .rename(columns={"span_text_random": "token", "prob_delta": "mean_prob_delta"})
)
display(HTML(last30_mean_prob_delta.to_html(index=False)))



# %%
