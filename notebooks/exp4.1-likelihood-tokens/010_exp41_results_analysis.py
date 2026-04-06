# %%
import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from pivotal_tokens.constants import get_data_dir, get_artifacts_dir
from pivotal_tokens.hf.loading import load_tokenizer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import HTML, display
from transformers import PreTrainedTokenizer

# EXP41_DIR = get_data_dir() / "experiments" / "exp4.1-likelihood-tokens" / "exp4.1.1-qwen3-1.7b-likelihood-tokens"
EXP41_DIR = get_data_dir() / "experiments" / "exp4.1-likelihood-tokens" / "exp4.1.2-qwen3-1.7b-likelihood-tokens-trunc"
EXP41_RESULTS_FILE = EXP41_DIR / "data" / "loglikelihood_pivotal_tokens.csv"

EXP22_DIR = get_data_dir() / "experiments" / "exp2.2-alt-tokens-on-pivotal-positions" / "exp2.2.1-qwen3-1.7b-alt-tokens-on-pivotal-positions"
EXP22_RESULTS_FILE = EXP22_DIR / "data" / "alternative_tokens.csv"

TRACES_FILE = get_artifacts_dir() / 'exp1.1.1-qwen3-1.7b-traces.csv'

QWEN3_1_7B_MODEL_ID = 'Qwen/Qwen3-1.7B'

# %%
# tokenizer = load_tokenizer(QWEN3_1_7B_MODEL_ID)

# prob_init_df = pd.read_csv(EXP22_RESULTS_FILE)
# prob_init_df = prob_init_df[["sample_id", "prob_init"]].rename(columns={"sample_id": "id"}).copy()
# traces_df = pd.read_csv(TRACES_FILE)
# traces_df = pd.merge(traces_df, prob_init_df, on="id", how="inner")


# nnll_df = pd.read_csv(EXP41_RESULTS_FILE)
# nnll_df['token_ids'] = nnll_df['token_ids'].apply(json.loads)

# df = pd.merge(traces_df, nnll_df, left_on='id', right_on='sample_id', how='inner')

# %%

# def get_thinking_trace_from_completion(completion: str) -> str:
#     prefix_split_str = '<|im_start|>assistant\n'
#     completion_with_suffix = completion.split(prefix_split_str)[-1].strip()

#     suffix_split_str = '<|im_end|>'
#     completion = completion_with_suffix.split(suffix_split_str)[0].strip()
#     return completion


# def get_token_length(text: str, tokenizer: PreTrainedTokenizer) -> int:
#     token_ids = tokenizer.encode(text, add_special_tokens=False)
#     return len(token_ids)


# df["partial_trace"] = df["pivotal_context"].apply(get_thinking_trace_from_completion)
# df["partial_trace_token_length"] = df["partial_trace"].apply(
#     lambda x: get_token_length(x, tokenizer)
# )
# df["trace_token_length"] = df["trace"].apply(lambda x: get_token_length(x, tokenizer))
# df["pivotal_token_relative_position"] = (
#     (df["partial_trace_token_length"] + 1) / df["trace_token_length"]
# )

# %%

# df = df.drop(columns=['pivotal_context'])
# df = df.drop(columns=['trace', 'metadata', 'partial_trace'])

# df.to_csv("data/preprocessed-trunc.csv", index=False)

# %%

def parse_token_ids(value: object) -> list[int]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return []


# PREPROCESSED_FILE = Path("data/preprocessed.csv")
PREPROCESSED_FILE = Path("data/preprocessed-trunc.csv")
if PREPROCESSED_FILE.exists():
    prep_df = pd.read_csv(PREPROCESSED_FILE)
    prep_df["token_ids"] = prep_df["token_ids"].apply(parse_token_ids)
else:
    prep_df = df.copy()


# %%
# sample_df = prep_df.groupby("sample_id", as_index=False).last()
# sample_df

# post_df = pd.read_csv('/home/xaparo00/workspace/projects/pivotal_tokens/data/experiments/exp1.1-hotpotqa-baseline/exp1.1.7-qwen3-1.7b-trace-posteriors/data/posteriors.csv')

# %%
# sample_df = pd.merge(sample_df, post_df[["id", "prob_post"]], on="id", how="inner")

# %%
# px.scatter(sample_df, x="nnll", y="prob_post", marginal_y="histogram", marginal_x="histogram")


# %%

# Columns:
# [
#     'id', 'query', 'ground_truth', 'prob_init', 'sample_id', 'span_id',
#     'token_ids', 'span_text', 'logprob', 'nnll', 'norm_coeff',
#     'partial_trace_token_length', 'trace_token_length',
#     'pivotal_token_relative_position'
# ]

# %% 

MAX_PLOT_SAMPLES = 20
rng = np.random.default_rng(42)
unique_sample_ids = pd.unique(prep_df["sample_id"])
n_plot_samples = min(MAX_PLOT_SAMPLES, len(unique_sample_ids))
plot_sample_ids = rng.choice(unique_sample_ids, size=n_plot_samples, replace=False)
prep_df = prep_df[prep_df["sample_id"].isin(plot_sample_ids)]

# %%
plot_df = prep_df.sort_values(["sample_id", "pivotal_token_relative_position"])
fig = px.line(
    plot_df,
    x="pivotal_token_relative_position",
    y="nnll",
    hover_data=["span_text", "prob_init", "ground_truth"],
    color="sample_id",
)
fig.show()

# %%

SMOOTHING_WINDOW = 30

smoothed_plot_df = prep_df.sort_values(["sample_id", "pivotal_token_relative_position"]).copy()
smoothed_plot_df["nnll_smoothed"] = (
    smoothed_plot_df.groupby("sample_id", sort=False)["nnll"]
    .transform(lambda s: s.rolling(window=SMOOTHING_WINDOW, min_periods=1, center=True).mean())
)


sample_plot_df = smoothed_plot_df.sort_values(["sample_id", "pivotal_token_relative_position"])
for sample_id, sample_group in sample_plot_df.groupby("sample_id", sort=False):
    sample_group = sample_group.copy()
    sample_query = sample_group["query"].iloc[0] if "query" in sample_group.columns else ""
    sample_ground_truth = sample_group["ground_truth"].iloc[0] if "ground_truth" in sample_group.columns else ""

    sample_group = (
        sample_group.groupby("pivotal_token_relative_position", as_index=False, sort=False)
        .agg(
            span_text=("span_text", lambda s: "".join(pd.unique(s.astype(str)))),
            nnll=("nnll", "mean"),
            nnll_smoothed=("nnll_smoothed", "mean"),
        )
        .sort_values("pivotal_token_relative_position")
    )

    fig = make_subplots(rows=1, cols=1)
    hover_tokens = sample_group["span_text"].astype(str).to_numpy().reshape(-1, 1)
    fig.add_trace(
        go.Scatter(
            x=sample_group["pivotal_token_relative_position"],
            y=sample_group["nnll"],
            mode="lines",
            name="nnll",
            customdata=hover_tokens,
            hovertemplate=(
                "pivotal_token_relative_position=%{x}<br>"
                "nnll=%{y}<br>"
                "token=%{customdata[0]}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sample_group["pivotal_token_relative_position"],
            y=sample_group["nnll_smoothed"],
            mode="lines",
            name=f"nnll_smoothed (w={SMOOTHING_WINDOW})",
            customdata=hover_tokens,
            hovertemplate=(
                "pivotal_token_relative_position=%{x}<br>"
                "nnll_smoothed=%{y}<br>"
                "token=%{customdata[0]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"Sample {sample_id}",
        xaxis_title="pivotal_token_relative_position",
        yaxis_title="nnll",
        legend_title="series",
    )
    fig.show()

    nnll_values = sample_group["nnll"].to_numpy(dtype=float)
    q10, q90 = np.quantile(nnll_values, [0.10, 0.90])
    nnll_range = max(float(q90 - q10), 1e-12)
    redness = 1.0 - np.clip((nnll_values - q10) / nnll_range, 0.0, 1.0)
    redness = np.power(redness, 1.6)

    token_html = []
    for token, red in zip(sample_group["span_text"], redness):
        alpha = 0.05 + 0.30 * float(red)
        token_html.append(
            f'<span style="background-color: rgba(255, 0, 0, {alpha:.3f}); border-radius: 2px;">{escape(str(token))}</span>'
        )

    display(
        HTML(
            "<div style='margin: 8px 0 20px; line-height: 1.75; white-space: pre-wrap; "
            "background-color: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px;'>"
            + "<div style='margin-bottom: 8px;'>"
            + f"<div><strong>Question:</strong> {escape(str(sample_query))}</div>"
            + f"<div><strong>Ground truth:</strong> {escape(str(sample_ground_truth))}</div>"
            + "</div>"
            + "".join(token_html)
            + "</div>"
        )
    )

# %%
