# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from IPython.display import HTML, display
from transformers import PreTrainedTokenizer

from pivotal_tokens.constants import get_data_dir, get_artifacts_dir
from pivotal_tokens.hf.loading import  load_tokenizer
from pivotal_tokens.repo import SampleRepo, DictRepo


EXP_DIR = get_data_dir() / 'experiments' / 'exp2.1-sps-tokens' / 'exp2.1.4-qwen3-1.7b-sps-tokens-small-prob-threshold'
# EXP_DIR = get_data_dir() / 'experiments' / 'exp2.1-sps-tokens' / 'exp2.1.2-qwen3-1.7b-sps-tokens-small-prob-threshold'
REPO_DIR = EXP_DIR / 'data' / 'repo'
PIVOTAL_TOKENS_FILE = EXP_DIR / 'data' / 'pivotal_tokens.csv'

TRACES_FILE = get_artifacts_dir() / 'exp1.1.1-qwen3-1.7b-traces.csv'

QWEN3_1_7B_MODEL_ID = 'Qwen/Qwen3-1.7B'

# %%

tokenizer = load_tokenizer(QWEN3_1_7B_MODEL_ID)
base_repo = DictRepo(dirpath=REPO_DIR)

traces_df = pd.read_csv(TRACES_FILE)
tokens_df = pd.read_csv(PIVOTAL_TOKENS_FILE)
tokens_df['token_ids'] = tokens_df['token_ids'].apply(json.loads)

df = pd.merge(traces_df, tokens_df, left_on='id', right_on='sample_id', how='inner')

# %%

def get_thinking_trace_from_completion(completion: str) -> str:
    prefix_split_str = '<|im_start|>assistant\n'
    completion_with_suffix = completion.split(prefix_split_str)[-1].strip()

    suffix_split_str = '<|im_end|>'
    completion = completion_with_suffix.split(suffix_split_str)[0].strip()
    return completion


def get_token_length(text: str, tokenizer: PreTrainedTokenizer) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return len(token_ids)


df['partial_trace'] = df['pivotal_context'].apply(get_thinking_trace_from_completion)
df['partial_trace_token_length'] = df['partial_trace'].apply(lambda x: get_token_length(x, tokenizer))
df['trace_token_length'] = df['trace'].apply(lambda x: get_token_length(x, tokenizer))
df['pivotal_token_relative_position'] = (df['partial_trace_token_length'] + 1) / df['trace_token_length']

assert (df['partial_trace_token_length'] < df['trace_token_length']).mean() == 1.0

# %%

def get_init_success_prob(sample_id: str, base_repo: DictRepo) -> float | None:
    repo = SampleRepo(base_repo=base_repo, sample_id=sample_id)
    subdivisions = repo.list(path="subdivisions")

    init_prob = None
    for subdiv in subdivisions:
        data = repo.load(path="subdivisions", key=subdiv)
        if len(data['prefix']) == 0 and \
                data['full_seq'].startswith('<think>') and \
                data['full_seq'].endswith('</think>'):
            init_prob = data['prob_before']
            break
    
    return init_prob

df['prob_init'] = df['sample_id'].apply(lambda x: get_init_success_prob(x, base_repo))

df['prob_before_normalized'] = df['prob_before'] / df['prob_init']
df['prob_after_normalized'] = df['prob_after'] / df['prob_init']
df['prob_delta_normalized'] = df['prob_delta'] / df['prob_init']

df['is_positive'] = df['prob_delta'] >= 0

df = df[df['is_pivotal'] == True].copy()

# %%

unique_samples = df["sample_id"].nunique()
num_pivotal_tokens = len(df)
num_positive = df["is_positive"].sum()
print(
    "Overall statistics:\n"
    f"- Unique samples: {unique_samples}\n"
    f"- Pivotal tokens: {num_pivotal_tokens}\n"
    f"- Positive pivotal tokens: {num_positive}"
)

# %%

df[df['sample_id'] == '5ab6f0c25542995eadef00ea'].head(10)
# %%

end_of_sentence_mask = df["span_text"].str.strip().str.fullmatch(r"[.;?!:]+", na=False)
span_type_counts = (
    pd.Series(
        np.where(end_of_sentence_mask, "End-of-sentence punctuation", "Other tokens"),
        name="token_type",
    )
    .value_counts()
    .rename_axis("token_type")
    .reset_index(name="count")
)
span_type_counts["token_type"] = pd.Categorical(
    span_type_counts["token_type"],
    categories=["End-of-sentence punctuation", "Other tokens"],
    ordered=True,
)
span_type_counts = span_type_counts.sort_values("token_type")

bar_span_type = px.bar(
    span_type_counts,
    x="token_type",
    y="count",
    title="Counts of end-of-sentence punctuation vs other pivotal tokens",
)
bar_span_type.update_layout(xaxis_title="Token type", yaxis_title="Count")

# %%
sample_probs = df[["sample_id", "prob_init"]].drop_duplicates("sample_id")
hist_init = px.histogram(
    sample_probs,
    x="prob_init",
    nbins=60,
    title="Distribution of initial probabilities per sample (P0)",
)
hist_init.update_layout(xaxis_title="P0", yaxis_title="Count")


# %%

hist_delta = px.histogram(
    df,
    x="prob_delta",
    nbins=60,
    title="ΔP(success) after pivotal token",
)
hist_delta.update_layout(xaxis_title="ΔP(success)", yaxis_title="Count")

# %%

quantile_bins = pd.qcut(df["prob_init"], q=5, duplicates="drop")
intervals = sorted(quantile_bins.cat.categories, key=lambda x: x.left)
quantile_bins = quantile_bins.cat.reorder_categories(intervals, ordered=True)
labels = [f"{interval.left:.2f} - {interval.right:.2f}" for interval in intervals]
df["prob_init_quantile_range"] = quantile_bins.cat.rename_categories(labels)
df = df.sort_values(by="prob_init_quantile_range")

box_delta_by_q = px.box(
    df,
    x="prob_init_quantile_range",
    y="prob_delta",
    title="ΔP(success) by initial probability quantiles (P0)",
)
box_delta_by_q.update_layout(xaxis_title="P0 range", yaxis_title="ΔP(success)")

# %%

df["pivotal_token_abs_position"] = df["partial_trace_token_length"] + 1
scatter_abs = px.scatter(
    df,
    x="pivotal_token_abs_position",
    y="prob_delta",
    opacity=0.4,
    render_mode="webgl",
    title="ΔP(success) vs absolute token position",
)
scatter_abs.update_layout(xaxis_title="Absolute position in trace", yaxis_title="ΔP(success)")

# %%

scatter_abs_to_total = px.scatter(
    df,
    x="trace_token_length",
    y="pivotal_token_abs_position",
    opacity=0.4,
    color='prob_init',
    color_continuous_scale="RdBu_r",
    labels={"prob_init": "P0"},
    title="Trace length vs absolute token position"
)
scatter_abs_to_total.update_layout(xaxis_title="Trace length", yaxis_title="Absolute position in trace")

# %%

scatter_rel_to_total = px.scatter(
    df,
    x="trace_token_length",
    y="pivotal_token_relative_position",
    opacity=0.4,
    color='prob_init',
    color_continuous_scale="RdBu_r",
    labels={"prob_init": "P0"},
    title="Trace length vs relative token position"
)
scatter_rel_to_total.update_layout(xaxis_title="Trace length", yaxis_title="Relative position in trace")

# %%

scatter_total_p0 = px.scatter(
    df,
    x="trace_token_length",
    y="prob_init",
    opacity=0.4,
    title="Trace length vs initial probability (P0)"
)
scatter_total_p0.update_layout(xaxis_title="Trace length", yaxis_title="P0")


# %%

scatter_rel = px.scatter(
    df,
    x="pivotal_token_relative_position",
    y="prob_delta",
    opacity=0.4,
    render_mode="webgl",
    title="ΔP(success) vs relative token position",
)
scatter_rel.update_layout(xaxis_title="Relative position in trace", yaxis_title="ΔP(success)")

# %%
token_counts_head = df["span_text"].value_counts().head(30)
token_counts_head_df = token_counts_head.reset_index()
token_counts_head_df.columns = ["token", "count"]
bar_top_tokens = px.bar(
    token_counts_head_df,
    x="token",
    y="count",
    title="Top-30 most frequent pivotal tokens",
)
bar_top_tokens.update_layout(xaxis_title="Token", yaxis_title="Count")

# %%

token_counts_tail = df["span_text"].value_counts().tail(30)
token_counts_tail_df = token_counts_tail.reset_index()
token_counts_tail_df.columns = ["token", "count"]
bar_last_tokens = px.bar(
    token_counts_tail_df,
    x="token",
    y="count",
    title="Top-30 least frequent pivotal tokens",
)
bar_last_tokens.update_layout(xaxis_title="Token", yaxis_title="Count")


# %%
top_tokens = token_counts_head.index
mean_delta_df = (
    df[df["span_text"].isin(top_tokens)]
    .groupby("span_text", sort=False)["prob_delta"]
    .mean()
    .reindex(top_tokens)
    .reset_index()
)
mean_delta_df.columns = ["token", "mean_delta"]
bar_mean_delta = px.bar(
    mean_delta_df,
    x="token",
    y="mean_delta",
    title="Mean ΔP(success) for top-30 tokens",
)
bar_mean_delta.update_layout(xaxis_title="Token", yaxis_title="Mean ΔP(success)")

# %%
last_tokens = token_counts_tail.index
mean_delta_df = (
    df[df["span_text"].isin(last_tokens)]
    .groupby("span_text", sort=False)["prob_delta"]
    .mean()
    .reindex(last_tokens)
    .reset_index()
)
mean_delta_df.columns = ["token", "mean_delta"]
bar_mean_delta = px.bar(
    mean_delta_df,
    x="token",
    y="mean_delta",
    title="Mean ΔP(success) for last-30 tokens",
)
bar_mean_delta.update_layout(xaxis_title="Token", yaxis_title="Mean ΔP(success)")


# %%
median_pos_df = (
    df[df["span_text"].isin(top_tokens)]
    .groupby("span_text", sort=False)["pivotal_token_abs_position"]
    .median()
    .reindex(top_tokens)
    .reset_index()
)
median_pos_df.columns = ["token", "median_position"]
bar_median_pos = px.bar(
    median_pos_df,
    x="token",
    y="median_position",
    title="Median pivotal token position in trace (top-30 tokens)",
)
bar_median_pos.update_layout(xaxis_title="Token", yaxis_title="Median position in trace")


# %%
median_pos_df = (
    df[df["span_text"].isin(last_tokens)]
    .groupby("span_text", sort=False)["pivotal_token_abs_position"]
    .median()
    .reindex(last_tokens)
    .reset_index()
)
median_pos_df.columns = ["token", "median_position"]
bar_median_pos = px.bar(
    median_pos_df,
    x="token",
    y="median_position",
    title="Median pivotal token position in trace (last-30 tokens)",
)
bar_median_pos.update_layout(xaxis_title="Token", yaxis_title="Median position in trace")

# %%

def build_token_scatter(
    token_counts: pd.Series,
    tokens: pd.Index,
    title: str,
    xjittering: bool = False,
):
    counts_df = token_counts.reindex(tokens).reset_index()
    counts_df.columns = ["token", "count"]
    stats_df = (
        df[df["span_text"].isin(tokens)]
        .groupby("span_text", sort=False)
        .agg(
            median_position=("pivotal_token_abs_position", "median"),
            mean_delta=("prob_delta", "mean"),
        )
        .reindex(tokens)
        .reset_index()
    )
    stats_df.columns = ["token", "median_position", "mean_delta"]
    scatter_df = counts_df.merge(stats_df, on="token", how="left")
    x_col = "count"
    xaxis_title = "Count"
    plot_title = title
    if xjittering:
        rng = np.random.default_rng()
        scatter_df["count_jitter"] = scatter_df["count"] + rng.uniform(
            low=-0.4,
            high=0.4,
            size=len(scatter_df),
        )
        x_col = "count_jitter"
        xaxis_title = "Count (jitter)"
        plot_title = f"{title}"
    scatter = px.scatter(
        scatter_df,
        x=x_col,
        y="median_position",
        color="mean_delta",
        text="token",
        render_mode="webgl",
        color_continuous_scale="RdBu_r",
        title=plot_title,
    )
    scatter.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title="Median position in trace",
        coloraxis_colorbar_title="Mean ΔP",
    )
    scatter.update_traces(textposition="top center")
    return scatter

scatter_top_tokens = build_token_scatter(
    token_counts_head,
    top_tokens,
    "Top-30 tokens: count vs median position (color = mean ΔP)",
)
scatter_top_tokens.show()
scatter_last_tokens = build_token_scatter(
    token_counts_tail,
    last_tokens,
    "Last-30 tokens: count vs median position (color = mean ΔP)",
    xjittering=True
)
scatter_last_tokens.show()
# %%
gt_series = df["ground_truth"].fillna("").astype(str)
token_series = df["span_text"].fillna("").astype(str).str.strip()
token_in_gt = np.fromiter(
    (token in gt for token, gt in zip(token_series, gt_series)),
    dtype=bool,
    count=len(df),
)
df = df.assign(token_in_gt=token_in_gt)
gt_counts = (
    df.groupby("sample_id", sort=False)["token_in_gt"]
    .sum()
    .reset_index(name="pivotal_in_gt_count")
)
df_with_gt_counts = df.merge(gt_counts, on="sample_id", how="left")
_rng = np.random.default_rng(0)
_x_jitter = _rng.uniform(-0.25, 0.25, size=len(df_with_gt_counts))
df_with_gt_counts = df_with_gt_counts.assign(
    pivotal_in_gt_count_jitter=df_with_gt_counts["pivotal_in_gt_count"] + _x_jitter
)

scatter_gt_intersections = px.scatter(
    df_with_gt_counts,
    x="pivotal_in_gt_count_jitter",
    y="pivotal_token_relative_position",
    color="prob_delta",
    opacity=0.8,
    render_mode="webgl",
    color_continuous_scale="RdBu_r",
    title="Relative position vs count of pivotal tokens in GT (color = ΔP)",
)
scatter_gt_intersections.update_layout(
    xaxis_title="Pivotal tokens in GT (count)",
    yaxis_title="Relative position in trace",
    coloraxis_colorbar_title="ΔP(success)",
)

# %%

gt_token_table = df.assign(token_in_gt=token_in_gt)[
    ["ground_truth", "span_text", "token_in_gt", "prob_delta"]
]
gt_token_table_sorted = (
    gt_token_table[gt_token_table["token_in_gt"]]
    .sort_values("prob_delta", ascending=False)
)
display(HTML(gt_token_table_sorted.to_html(index=False)))
# print(gt_token_table_sorted)

# %%


mid_pos_high_delta_mask = (
    df["pivotal_token_relative_position"].between(0.2, 0.8)
    & (df["prob_delta"] > 0.3)
)
mid_pos_high_delta_table = df.loc[
    mid_pos_high_delta_mask,
    ["span_text", "pivotal_token_relative_position", "prob_delta"],
].sort_values("prob_delta", ascending=False)
display(HTML(mid_pos_high_delta_table.to_html(index=False)))

# %%

filter_mask = None
filter_mask = df["prob_delta"].between(0.01, 0.2)
df_filtered = df if filter_mask is None else df.loc[filter_mask]

multi_token_sample_ids = df_filtered["sample_id"].value_counts()
multi_token_sample_ids = multi_token_sample_ids[multi_token_sample_ids > 1].index
multi_token_df = df_filtered[df_filtered["sample_id"].isin(multi_token_sample_ids)].copy()

positions = multi_token_df["pivotal_token_relative_position"].to_numpy()
sample_ids = multi_token_df["sample_id"].to_numpy()
prob_before = multi_token_df["prob_before"].to_numpy()
prob_after = multi_token_df["prob_after"].to_numpy()

position_delta = 0#1e-3
base_plot_df = pd.DataFrame(
    {
        "sample_id": np.concatenate([sample_ids, sample_ids]),
        "token_position": np.concatenate([positions - position_delta, positions]),
        "prob": np.concatenate([prob_before, prob_after]),
    }
)

pos_series = multi_token_df["pivotal_token_relative_position"]
sample_series = multi_token_df["sample_id"]
idx_min = pos_series.groupby(sample_series, sort=False).idxmin()
idx_max = pos_series.groupby(sample_series, sort=False).idxmax()

start_df = pd.DataFrame(
    {
        "sample_id": multi_token_df.loc[idx_min, "sample_id"].to_numpy(),
        "token_position": np.zeros(idx_min.size, dtype=float),
        "prob": multi_token_df.loc[idx_min, "prob_before"].to_numpy(),
    }
)
end_df = pd.DataFrame(
    {
        "sample_id": multi_token_df.loc[idx_max, "sample_id"].to_numpy(),
        "token_position": np.ones(idx_max.size, dtype=float),
        "prob": multi_token_df.loc[idx_max, "prob_after"].to_numpy(),
    }
)

plot_df = pd.concat([base_plot_df, start_df, end_df], ignore_index=True).sort_values(
    ["sample_id", "token_position"]
)

line_multi_token = px.line(
    plot_df,
    x="token_position",
    y="prob",
    color="sample_id",
    line_group="sample_id",
    render_mode="webgl",
    title="P(success) before/after by token position (multi-token samples)",
)
line_multi_token.update_layout(
    xaxis_title="Relative token position",
    yaxis_title="P(success)",
    showlegend=True,
    yaxis_range=[0, 1],
)
line_multi_token.update_traces(opacity=1.0)

# %%

prob_init_by_sample = (
    multi_token_df.groupby("sample_id", sort=False)["prob_init"].first().to_numpy()
)
last_prob_after_by_sample = multi_token_df.loc[idx_max, "prob_after"].to_numpy()
init_lt_last = prob_init_by_sample < last_prob_after_by_sample

init_lt_last_df = pd.DataFrame({
    'prob_init_by_sample': prob_init_by_sample,
    'last_prob_after_by_sample': last_prob_after_by_sample,
    'prob_init_lt_last_prob_after': init_lt_last
})

display(HTML(init_lt_last_df.groupby('prob_init_lt_last_prob_after').agg('count').to_html()))

# %%
scatter_after_norm = px.scatter(
    df,
    x="pivotal_token_relative_position",
    y="prob_after_normalized",
    render_mode="webgl",
    title="P_after / P0 vs relative token position",
)
scatter_after_norm.update_layout(
    xaxis_title="Relative token position",
    yaxis_title="P_after / P0",
)
scatter_after_norm.update_traces(opacity=0.7)

# %%

scatter_after = px.scatter(
    df,
    x="pivotal_token_relative_position",
    y="prob_after",
    render_mode="webgl",
    title="P_after vs relative token position",
)
scatter_after.update_layout(
    xaxis_title="Relative token position",
    yaxis_title="P_after",
)
scatter_after.update_traces(opacity=0.7)

# %%
logprob_df = df.assign(token_logprob=np.log(df["prob_after"].clip(lower=1e-12)))
scatter_token_logprob = px.scatter(
    logprob_df,
    x="pivotal_token_relative_position",
    y="token_logprob",
    render_mode="webgl",
    title="Token logprob vs relative token position",
)
scatter_token_logprob.update_layout(
    xaxis_title="Relative token position",
    yaxis_title="Token logprob",
)
scatter_token_logprob.update_traces(opacity=0.7)

# %%

high_ratio_mask = df["prob_after_normalized"] > 2
high_ratio_table = df.loc[
    high_ratio_mask,
    [
        "span_text",
        "prob_init",
        "prob_after",
        "prob_after_normalized",
        # "pivotal_token_relative_position",
        "prob_delta",
        "token_in_gt"
    ],
].sort_values("prob_after_normalized", ascending=False)
display(HTML(high_ratio_table.to_html(index=False)))

# %%
