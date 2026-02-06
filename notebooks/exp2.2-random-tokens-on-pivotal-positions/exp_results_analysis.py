# %%
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
from transformers import PreTrainedTokenizer

from pivotal_tokens.constants import get_data_dir, get_artifacts_dir
from pivotal_tokens.hf.loading import  load_tokenizer
from pivotal_tokens.repo import SampleRepo, DictRepo


EXP21_EXP_DIR = get_data_dir() / 'experiments' / 'exp2.1-sps-tokens' / 'exp2.1.4-qwen3-1.7b-sps-tokens-small-prob-threshold'
REPO_DIR = EXP21_EXP_DIR / 'data' / 'repo'

EXP22_EXP_DIR = get_data_dir() / 'experiments' / 'exp2.2-random-tokens-on-pivotal-positions' / 'exp2.2.1-qwen3-1.7b-random-tokens-on-pivotal-positions'
PIVOTAL_TOKENS_FILE = EXP22_EXP_DIR / 'data' / 'alternative_tokens.csv'

TRACES_FILE = get_artifacts_dir() / 'exp1.1.1-qwen3-1.7b-traces.csv'

QWEN3_1_7B_MODEL_ID = 'Qwen/Qwen3-1.7B'
THRESHOLD = 0.1

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

df['prob_delta_alt'] = df['prob_after_alt'] - df['prob_before']
df['prob_before_normalized'] = df['prob_before'] / df['prob_init']
df['prob_after_alt_normalized'] = df['prob_after'] / df['prob_init']
df['prob_delta_alt_normalized'] = df['prob_delta'] / df['prob_init']

df['is_positive'] = df['prob_delta_alt'] >= 0

df = df[df['is_pivotal'] == True].copy()

# %%

df.head(1).T

# %%

alt_type_order = ['pivotal', 'top_1', 'top_2', 'top_3', 'median', 'lowest']
replacement_order = ['top_1', 'top_2', 'top_3', 'median', 'lowest']

pivotal_delta_df = df[['prob_delta']].copy()
pivotal_delta_df['alt_type'] = 'pivotal'
pivotal_delta_df = pivotal_delta_df.rename(columns={'prob_delta': 'delta_prob'})

replacement_delta_df = df[['alt_type', 'prob_delta_alt']].copy()
replacement_delta_df = replacement_delta_df.rename(columns={'prob_delta_alt': 'delta_prob'})

delta_by_type_df = pd.concat([pivotal_delta_df, replacement_delta_df], ignore_index=True)
delta_by_type_df['alt_type'] = pd.Categorical(delta_by_type_df['alt_type'], categories=alt_type_order, ordered=True)

fig_delta_box = px.box(
    delta_by_type_df,
    x='alt_type',
    y='delta_prob',
    title='ΔP(success) by replacement type',
    category_orders={'alt_type': alt_type_order},
)
fig_delta_box.show()

# %%

fig_after_hist = px.histogram(
    df,
    x='prob_after_alt',
    color='alt_type',
    nbins=10,
    barmode='group',
    title='P_after(success) by replacement type',
    category_orders={'alt_type': replacement_order},
)
fig_after_hist.show()

# %%

fig_delta_vs_logprob = px.scatter(
    df,
    x='token_logprob_alt',
    y='prob_delta_alt',
    color='alt_type',
    title='ΔP(success) vs replacement logprob',
    category_orders={'alt_type': replacement_order},
)
fig_delta_vs_logprob.update_traces(opacity=.7)
fig_delta_vs_logprob.show()

# %%

quantile_bins = pd.qcut(df["prob_before"], q=4, duplicates="drop")
intervals = sorted(quantile_bins.cat.categories, key=lambda x: (x.left, x.right))
quantile_bins = quantile_bins.cat.reorder_categories(intervals, ordered=True)
labels = [f"{interval.left:.2f} - {interval.right:.2f}" for interval in intervals]
df["p0_quantile"] = quantile_bins.cat.rename_categories(labels)
fig_delta_by_p0 = px.box(
    df,
    x='p0_quantile',
    y='prob_delta_alt',
    color='alt_type',
    title='ΔP(success) by P0 quantile and replacement type',
    category_orders={'p0_quantile': labels, 'alt_type': replacement_order},
)
fig_delta_by_p0.show()

# %%

fig_delta_vs_position = px.scatter(
    df,
    x='pivotal_token_relative_position',
    y='prob_delta_alt',
    color='alt_type',
    title='ΔP(success) vs pivotal position',
    category_orders={'alt_type': replacement_order},
)
fig_delta_vs_position.update_traces(opacity=.7)
fig_delta_vs_position.show()

# %%

flip_df = df.copy()
before_above = flip_df['prob_delta'].abs() >= THRESHOLD
after_above = flip_df['prob_delta_alt'].abs() >= THRESHOLD
flip_df['flip'] = before_above != after_above
flip_rate_df = flip_df.groupby('alt_type', as_index=False)['flip'].mean()
flip_rate_df['alt_type'] = pd.Categorical(flip_rate_df['alt_type'], categories=replacement_order, ordered=True)
flip_rate_df = flip_rate_df.sort_values('alt_type')
fig_flip_rate = px.bar(
    flip_rate_df,
    x='alt_type',
    y='flip',
    title='Outcome flip rate by replacement type',
    category_orders={'alt_type': replacement_order},
)
fig_flip_rate.show()

fig_flip_scatter = px.scatter(
    flip_df,
    x='prob_delta',
    y='prob_delta_alt',
    color='flip',
    title='Flip vs non-flip relative to threshold',
    labels={'prob_delta': 'ΔP(success) before', 'prob_delta_alt': 'ΔP(success) after'},
    color_discrete_map={True: '#d62728', False: '#1f77b4'},
)
fig_flip_scatter.add_vline(x=THRESHOLD, line_dash='dot', line_color='gray')
fig_flip_scatter.add_vline(x=-THRESHOLD, line_dash='dot', line_color='gray')
fig_flip_scatter.add_hline(y=THRESHOLD, line_dash='dot', line_color='gray')
fig_flip_scatter.add_hline(y=-THRESHOLD, line_dash='dot', line_color='gray')
fig_flip_scatter.update_traces(opacity=.7)
fig_flip_scatter.show()

# %%

topk_df = df[df['alt_type'].isin(['top_1', 'top_2', 'top_3'])].copy()
topk_mean_delta = topk_df.groupby('alt_type', as_index=False)['prob_delta_alt'].mean()
topk_mean_delta['alt_type'] = pd.Categorical(topk_mean_delta['alt_type'], categories=['top_1', 'top_2', 'top_3'], ordered=True)
topk_mean_delta = topk_mean_delta.sort_values('alt_type')
fig_topk_delta = px.bar(
    topk_mean_delta,
    x='alt_type',
    y='prob_delta_alt',
    title='Mean ΔP(success) by top-k rank',
    category_orders={'alt_type': ['top_1', 'top_2', 'top_3']},
)
fig_topk_delta.show()

# %%

df['prob_after_diff'] = df['prob_after'] - df['prob_after_alt']
fig_after_diff_hist = px.histogram(
    df,
    x='prob_after_diff',
    title='P_after(pivotal) − P_after(replace)',
)
fig_after_diff_hist.show()

# %%

fig_after_diff_by_p0 = px.box(
    df,
    x='p0_quantile',
    y='prob_after_diff',
    title='P_after diff by P0 quantile',
)
fig_after_diff_by_p0.show()

# %%
