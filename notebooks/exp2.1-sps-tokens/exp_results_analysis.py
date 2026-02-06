# %%
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
from transformers import PreTrainedTokenizer

from pivotal_tokens.constants import get_data_dir, get_artifacts_dir
from pivotal_tokens.hf.loading import  load_tokenizer
from pivotal_tokens.repo import SampleRepo, DictRepo


EXP_DIR = get_data_dir() / 'experiments' / 'exp2.1-sps-tokens' / 'exp2.1.4-qwen3-1.7b-sps-tokens-small-prob-threshold'
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
token_counts = df["span_text"].value_counts().head(20)
token_counts_df = token_counts.reset_index()
token_counts_df.columns = ["token", "count"]
bar_top_tokens = px.bar(
    token_counts_df,
    x="token",
    y="count",
    title="Top-20 most frequent pivotal tokens",
)
bar_top_tokens.update_layout(xaxis_title="Token", yaxis_title="Count")

# %%
top_tokens = token_counts.index
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
    title="Mean ΔP(success) for top-20 tokens",
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
    title="Median pivotal token position in trace (top-20 tokens)",
)
bar_median_pos.update_layout(xaxis_title="Token", yaxis_title="Median position in trace")

# %%
hist_after = px.histogram(
    df,
    x="prob_after",
    nbins=60,
    title="P_after(success) after pivotal token",
)
hist_after.update_layout(xaxis_title="P_after(success)", yaxis_title="Count")


# %%
def count_pivotal_in_gt(group: pd.DataFrame) -> int:
    gt = group["ground_truth"].iloc[0]
    if not isinstance(gt, str):
        return 0
    tokens = group["span_text"].astype(str).str.strip()
    return sum(token in gt for token in tokens)

gt_counts = (
    df.groupby("sample_id", sort=False)
    .apply(count_pivotal_in_gt)
    .reset_index(name="pivotal_in_gt_count")
)
gt_counts_df = (
    gt_counts["pivotal_in_gt_count"]
    .value_counts()
    .sort_index()
    .reset_index()
)
gt_counts_df.columns = ["pivotal_in_gt_count", "num_samples"]
bar_gt_intersections = px.bar(
    gt_counts_df,
    x="pivotal_in_gt_count",
    y="num_samples",
    title="Pivotal token intersections with ground truth answer",
)
bar_gt_intersections.update_layout(xaxis_title="Count in GT", yaxis_title="Samples")

# %%
