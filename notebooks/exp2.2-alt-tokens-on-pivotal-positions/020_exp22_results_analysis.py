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


EXP_DIR = get_data_dir() / "experiments" / "exp2.2-alt-tokens-on-pivotal-positions" / "exp2.2.1-qwen3-1.7b-alt-tokens-on-pivotal-positions"
ALTERNATIVE_TOKENS_FILE = EXP_DIR / "data" / "alternative_tokens.csv"

TRACES_FILE = get_artifacts_dir() / "exp1.1.1-qwen3-1.7b-traces.csv"

QWEN3_1_7B_MODEL_ID = "Qwen/Qwen3-1.7B"

# %%

tokenizer = load_tokenizer(QWEN3_1_7B_MODEL_ID)

traces_df = pd.read_csv(TRACES_FILE)
tokens_df = pd.read_csv(ALTERNATIVE_TOKENS_FILE)


tokens_df["token_ids"] = tokens_df["token_ids"].apply(json.loads)
tokens_df["token_ids_alt"] = tokens_df["token_ids_alt"]

df = pd.merge(traces_df, tokens_df, left_on="id", right_on="sample_id", how="inner")

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

df = df[df["is_pivotal"] == True].copy()

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

span_text_match = df["span_text"].eq(df["span_text_alt"])
span_match_pivot = pd.crosstab(
    index=df["alt_type"],
    columns=span_text_match,
    rownames=["alt_type"],
    colnames=["span_text == span_text_alt"],
)
display(span_match_pivot)

# %%

hist_delta_alt_by_type = px.histogram(
    df,
    x="prob_delta_alt",
    color="alt_type",
    nbins=10,
    barmode="group",
    title="ΔP(success) after replacement token (alternative), by alt type",
    labels={'alt_type': 'Alt. token type'}
)
hist_delta_alt_by_type.update_layout(
    xaxis_title="ΔP(success) after alternative", yaxis_title="Count"
)

# %%


quantile_bins = pd.qcut(df["prob_init"], q=5, duplicates="drop")
intervals = sorted(quantile_bins.cat.categories, key=lambda x: x.left)
quantile_bins = quantile_bins.cat.reorder_categories(intervals, ordered=True)
labels = [f"{interval.left:.2f} - {interval.right:.2f}" for interval in intervals]
df["prob_init_quantile_range"] = quantile_bins.cat.rename_categories(labels)
df = df.sort_values(by="prob_init_quantile_range")

for alt_type in sorted(df["alt_type"].unique()):
    plot_df = df.query(f"alt_type == '{alt_type}'")
    box_delta_by_q = px.box(
        plot_df,
        x="prob_init_quantile_range",
        y="prob_delta_alt",
        title=f"ΔP(success) by initial probability quantiles (P0) (alt type={alt_type})",
    )
    box_delta_by_q.update_layout(xaxis_title="P0 range", yaxis_title="ΔP(success)")
    box_delta_by_q.show()

# %%
df["prob_after_diff"] = df["prob_after"] - df["prob_after_alt"]

# %%


plot_df = df[df["span_text"].eq(df["span_text_alt"])]
scatter_prob_after_match = px.scatter(
    plot_df,
    x="prob_after",
    y="prob_after_alt",
    color="alt_type",
    hover_data=["span_text"],
    opacity=0.4,
    render_mode="webgl",
    title="P(success) after vs P(success) after alt when tokens match",
)
scatter_prob_after_match.update_layout(
    xaxis_title="P(success) after pivotal token",
    yaxis_title="P(success) after alternative token",
    xaxis_range=[0, 1],
    yaxis_range=[0, 1],
)
scatter_prob_after_match.show()

# %%

alt_types_order = {
    "top_1": 0,
    "top_2": 1,
    "top_3": 2,
    "median": 3,
    "lowest": 4
}

box_diff_after_vs_alt = px.box(
    plot_df.assign(alt_type_order=df["alt_type"].apply(lambda x: alt_types_order[x])).sort_values("alt_type_order"),
    x="alt_type",
    y="prob_after_diff",
    points="outliers",
    title="P(success) diff after pivotal token vs alt by alt type when tokens equal",
)
box_diff_after_vs_alt.update_layout(
    xaxis_title="Alternative token type",
    yaxis_title="P(success) after - P(success) after alt",
)
box_diff_after_vs_alt.show()

# %%

scatter_diff_after_vs_orig_delta = px.scatter(
    plot_df,
    x="prob_delta",
    y="prob_after_diff",
    hover_data=["span_text", "span_text_alt", "prob_after", "prob_after_alt"],
    title="Diff of P(success) after pivotal token and alt token vs orig prob delta",
)
scatter_diff_after_vs_orig_delta.update_layout(
    xaxis_title="Original ΔP(succ)",
    yaxis_title="P(success) after - P(success) after alt",
)
scatter_diff_after_vs_orig_delta.show()

# %%

for alt_type in sorted(df["alt_type"].unique()):
    plot_df = df[df["alt_type"] == alt_type]

    scatter_delta = px.scatter(
        plot_df,
        x="prob_after",
        y="prob_after_alt",
        hover_data=["span_text", "span_text_alt"],
        opacity=0.6,
        render_mode="webgl",
        title=f"P(success) after of pivotal vs alt tokens,<br>alt (alt type={alt_type})",
    )
    scatter_delta.update_layout(
        xaxis_title="P(success) after",
        yaxis_title="P(success) after alternative",
        # xaxis_range=[-1, 1],
        # yaxis_range=[-1, 1],
    )
    scatter_delta.show()

# %%
for alt_type in sorted(df["alt_type"].unique()):
    plot_df = df[df["alt_type"] == alt_type]
    scatter_delta_normalized = px.scatter(
        plot_df,
        x="prob_after_normalized",
        y="prob_after_alt_normalized",
        opacity=0.4,
        hover_data=["span_text", "span_text_alt"],
        render_mode="webgl",
        title=f"Normalized P(success) after of pivotal vs alt tokens,<br>alt (alt type={alt_type})",
    )
    scatter_delta_normalized.update_layout(
        xaxis_title="P(success) after normalized",
        yaxis_title="P(success) after alternative normalized",
    )
    scatter_delta_normalized.show()

# %%

for alt_type in sorted(df["alt_type"].unique()):
    plot_df = df[df["alt_type"] == alt_type]
    scatter_rel_pos_alt_norm = px.scatter(
        plot_df,
        x="pivotal_token_relative_position",
        y="prob_after_alt_normalized",
        opacity=0.4,
        hover_data=["span_text", "span_text_alt"],
        render_mode="webgl",
        title=(
            f"Relative token position vs normalized P(success) after<br>alternative token (alt type={alt_type})"
        ),
    )
    scatter_rel_pos_alt_norm.update_layout(
        xaxis_title="Relative token position",
        yaxis_title="Normalized P(success) after alternative token",
    )
    scatter_rel_pos_alt_norm.show()

scatter_rel_pos_orig_norm = px.scatter(
    df[df["alt_type"] == alt_type],
    x="pivotal_token_relative_position",
    y="prob_after_normalized",
    # color="alt_type",
    opacity=0.4,
    hover_data=["span_text", "span_text_alt"],
    render_mode="webgl",
    title="Relative token position vs normalized P(success) after pivotal token",
)
scatter_rel_pos_orig_norm.update_layout(
    xaxis_title="Relative token position",
    yaxis_title="Normalized P(success) after pivotal token",
)
scatter_rel_pos_orig_norm.show()

# %%
df.columns

# %%

for alt_type in sorted(df["alt_type"].unique()):
    plot_df = df[df["alt_type"] == alt_type]
    scatter_rel_pos_alt_norm = px.scatter(
        plot_df,
        x="pivotal_token_relative_position",
        y="prob_delta_alt_normalized",
        opacity=0.4,
        hover_data=["span_text", "span_text_alt"],
        render_mode="webgl",
        title=(
            f"Relative token position vs normalized ΔP(success)<br>alternative token (alt type={alt_type})"
        ),
    )
    scatter_rel_pos_alt_norm.update_layout(
        xaxis_title="Relative token position",
        yaxis_title="Normalized ΔP(success)",
    )
    scatter_rel_pos_alt_norm.show()

scatter_rel_pos_orig_norm = px.scatter(
    df[df["alt_type"] == alt_type],
    x="pivotal_token_relative_position",
    y="prob_delta_normalized",
    # color="alt_type",
    opacity=0.4,
    hover_data=["span_text", "span_text_alt"],
    render_mode="webgl",
    title="Relative token position vs normalized ΔP(success)",
)
scatter_rel_pos_orig_norm.update_layout(
    xaxis_title="Relative token position",
    yaxis_title="Normalized ΔP(success)",
)
scatter_rel_pos_orig_norm.show()
# %%

df.prob_init.describe()
# %%
