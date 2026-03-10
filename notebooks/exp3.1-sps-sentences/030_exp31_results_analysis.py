# %%
import json
import ast
import html
from pathlib import Path

import nltk
import pandas as pd
import plotly.express as px
from IPython.display import HTML, display
from transformers import PreTrainedTokenizer

from pivotal_tokens.constants import get_data_dir, get_artifacts_dir
from pivotal_tokens.hf.loading import  load_tokenizer
from pivotal_tokens.repo import SampleRepo, DictRepo



EXP22_DIR = get_data_dir() / "experiments" / "exp2.2-alt-tokens-on-pivotal-positions" / "exp2.2.1-qwen3-1.7b-alt-tokens-on-pivotal-positions"
EXP22_RESULTS_FILE = EXP22_DIR / "data" / "alternative_tokens.csv"

EXP31_DIR = get_data_dir() / 'experiments' / 'exp3.1-sps-sentences' / 'exp3.1.1-qwen3-1.7b-sps-sentences'
REPO_DIR = EXP31_DIR / 'data' / 'repo'
PIVOTAL_TOKENS_FILE = EXP31_DIR / 'data' / 'pivotal_sentences.csv'

TRACES_FILE = get_artifacts_dir() / 'exp1.1.1-qwen3-1.7b-traces.csv'

QWEN3_1_7B_MODEL_ID = 'Qwen/Qwen3-1.7B'

# %%

tokenizer = load_tokenizer(QWEN3_1_7B_MODEL_ID)
base_repo = DictRepo(dirpath=REPO_DIR)

prob_init_df = pd.read_csv(EXP22_RESULTS_FILE)
prob_init_df = prob_init_df[["sample_id", "prob_init"]].rename(columns={"sample_id": "id"}).copy()
traces_df = pd.read_csv(TRACES_FILE)
traces_df = pd.merge(traces_df, prob_init_df, on="id", how="inner")


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
df['sentence_token_length'] = df['token_ids'].str.len()
df['trace_token_length'] = df['trace'].apply(lambda x: get_token_length(x, tokenizer))
df['sentence_start_token_position'] = df['partial_trace_token_length'] + 1
df['sentence_start_relative_position'] = df['sentence_start_token_position'] / df['trace_token_length']

assert (df['partial_trace_token_length'] < df['trace_token_length']).mean() == 1.0

# %%

df['prob_before_normalized'] = df['prob_before'] / df['prob_init']
df['prob_after_normalized'] = df['prob_after'] / df['prob_init']
df['prob_delta_normalized'] = df['prob_delta'] / df['prob_init']

df['is_positive'] = df['prob_delta'] >= 0

df = df[df['is_pivotal'] == True].copy()

# %%

fig = px.histogram(
    df,
    x='prob_delta',
    nbins=50,
    title='Distribution of ΔP(success) after pivotal sentence',
    labels={'prob_delta': 'ΔP(success)', 'count': 'Count'},
)
fig.show()

# %%

fig = px.scatter(
    df,
    x='sentence_token_length',
    y='prob_delta',
    title='ΔP(success) vs Pivotal Sentence Length',
    labels={'sentence_token_length': 'Sentence length (tokens)', 'prob_delta': 'ΔP(success)'},
    opacity=0.6,
)
fig.show()

# %%

fig = px.scatter(
    df,
    x='sentence_start_token_position',
    y='prob_delta',
    title='ΔP(success) vs Sentence Start Position in Trace',
    labels={'sentence_start_token_position': 'Sentence start position in trace (token index)', 'prob_delta': 'ΔP(success)'},
    opacity=0.6,
)
fig.show()

# %%

fig = px.scatter(
    df,
    x='sentence_start_relative_position',
    y='prob_delta_normalized',
    title='Normalzied ΔP(success) vs Sentence Start Position in Trace',
    hover_data=["span_text"],
    labels={'sentence_start_relative_position': 'Sentence start relative position in trace (token index)', 'prob_delta': 'ΔP(success)'},
    opacity=0.6,
)
fig.show()


# %%

fig = px.histogram(
    df,
    x='prob_after',
    nbins=50,
    title='Distribution of P_after(success) after pivotal sentence',
    labels={'prob_after': 'P_after(success)', 'count': 'Count'},
)
fig.show()


# %%

quantile_col = 'p0_quantile_range'
df[quantile_col] = pd.qcut(df['prob_init'], q=5, duplicates='drop')

quantile_order = [str(interval) for interval in df[quantile_col].cat.categories]
df[quantile_col] = pd.Categorical(df[quantile_col].astype(str), categories=quantile_order, ordered=True)

df = df.sort_values(by=quantile_col)
fig = px.box(
    df,
    x=quantile_col,
    y='prob_delta',
    title='ΔP(success) by P0 Quantile Range',
    labels={quantile_col: 'P0 quantile range', 'prob_delta': 'ΔP(success)'},
)
fig.show()

# %%

answer_col = 'answer_text'
contains_answer_col = 'contains_answer'

df[answer_col] = df['metadata'].apply(
    lambda m: ast.literal_eval(m).get('answer', '') if isinstance(m, str) else ''
)
df[contains_answer_col] = [
    answer.strip().lower() in span.lower() if answer else False
    for answer, span in zip(df[answer_col], df['span_text'])
]

fig = px.box(
    df,
    x='contains_answer',
    y='prob_delta_normalized',
    title='Share of pivotal sentences containing the answer string',
    labels={'contains_answer': 'Pivotal sentence class', 'share': 'Share'},
)
fig.show()


# %%

overall_sentences_df = (
    df[['id', 'trace']]
    .drop_duplicates(subset=['id'])
    .assign(
        overall_sentences=lambda x: x['trace'].map(
            lambda text: len(nltk.tokenize.sent_tokenize(text, language='english'))
        )
    )
)

fig = px.histogram(
    overall_sentences_df,
    x='overall_sentences',
    nbins=50,
    title='Distribution of Overall Sentences per Sample',
    labels={'overall_sentences': 'Overall sentences per sample', 'count': 'Count'},
)
fig.show()

# %%
pivotal_sentences_df = (
    df.groupby('sample_id', as_index=False)
    .agg({'span_id': 'nunique'})
    .rename(columns={'span_id': 'pivotal_sentences'})
)

fig = px.histogram(
    pivotal_sentences_df,
    x='pivotal_sentences',
    nbins=50,
    title='Distribution of Pivotal Sentences per Sample',
    labels={'pivotal_sentences': 'Pivotal sentences per sample', 'count': 'Count'},
)
fig.show()

# %%
sample_prob_delta_df = (
    df.groupby('sample_id', as_index=False)
    .agg({'prob_delta': 'mean'})
)

sample_sentences_df = (
    overall_sentences_df.rename(columns={'id': 'sample_id'})
    .merge(pivotal_sentences_df, on='sample_id', how='inner')
    .merge(sample_prob_delta_df, on='sample_id', how='inner')
)

fig = px.scatter(
    sample_sentences_df,
    x='overall_sentences',
    y='pivotal_sentences',
    color='prob_delta',
    # color_continuous_scale='RdBu_r',
    title='Overall vs Pivotal Sentences per Sample',
    labels={
        'overall_sentences': 'Overall sentences per sample',
        'pivotal_sentences': 'Pivotal sentences per sample',
        'prob_delta': 'Mean ΔP(success)',
    },
    opacity=0.7,
)
fig.show()

# %%

def make_sentence_tooltip(position_rel: float, prob_before: float, prob_after: float) -> str:
    return (
        f"position (rel): {position_rel:.4f} | "
        f"prob_before: {prob_before:.6f} | "
        f"prob_after: {prob_after:.6f}"
    )


def highlight_spans_in_trace(trace: str, sentence_rows: list[dict]) -> str:
    if not isinstance(trace, str):
        return '-'
    if not sentence_rows:
        return html.escape(trace)

    trace_lower = trace.lower()
    matches = []
    for row in sentence_rows:
        text = row['text']
        if not isinstance(text, str) or not text:
            continue
        span_lower = text.lower()
        span_len = len(text)
        tooltip = html.escape(
            make_sentence_tooltip(row['position_rel'], row['prob_before'], row['prob_after']),
            quote=True,
        )
        hit_class = 'trace-pos' if row['prob_delta'] >= 0 else 'trace-neg'

        start = 0
        while True:
            idx = trace_lower.find(span_lower, start)
            if idx == -1:
                break
            matches.append((idx, idx + span_len, tooltip, hit_class))
            start = idx + span_len

    if not matches:
        return html.escape(trace)

    matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    resolved = []
    cursor_end = -1
    for start, end, tooltip, hit_class in matches:
        if start < cursor_end:
            continue
        resolved.append((start, end, tooltip, hit_class))
        cursor_end = end

    cursor = 0
    chunks = []
    for start, end, tooltip, hit_class in resolved:
        chunks.append(html.escape(trace[cursor:start]))
        chunks.append(
            f"<b class='trace-hit has-tooltip {hit_class}' data-tip=\"{tooltip}\">{html.escape(trace[start:end])}</b>"
        )
        cursor = end
    chunks.append(html.escape(trace[cursor:]))
    return ''.join(chunks)


display_cols = [
    'question_text',
    'ground_truth_text',
    'span_text',
    'trace',
    'sentence_start_token_position',
    'sentence_start_relative_position',
    'prob_init',
    'prob_before',
    'prob_after',
    'prob_delta',
    'prob_delta_normalized',
]

question_meta_series = (
    df['metadata'].apply(lambda m: ast.literal_eval(m).get('question', '') if isinstance(m, str) else '')
    if 'query' not in df.columns
    else None
)
ground_truth_meta_series = (
    df['metadata'].apply(lambda m: ast.literal_eval(m).get('ground_truth', '') if isinstance(m, str) else '')
    if 'ground_truth' not in df.columns and 'answer_text' not in df.columns
    else None
)
df['question_text'] = df['query'] if 'query' in df.columns else question_meta_series
df['ground_truth_text'] = (
    df['ground_truth']
    if 'ground_truth' in df.columns
    else df['answer_text'] if 'answer_text' in df.columns else ground_truth_meta_series
)

display_df = df[display_cols].rename(
    columns={
        'question_text': 'question',
        'ground_truth_text': 'ground_truth',
        'span_text': 'text',
        'trace': 'trace',
        'sentence_start_token_position': 'position_abs',
        'sentence_start_relative_position': 'position_rel',
        'prob_before': 'prob_before',   
        'prob_after': 'prob_after',
        'prob_delta': 'prob_delta',
        'prob_delta_normalized': 'prob_delta_normalized',
    }
)

display_df = display_df.drop_duplicates().sort_values(by=['question', 'position_rel'])

total_rows = len(display_df)
pos_rows = int((display_df['prob_delta'] >= 0).sum())
neg_rows = total_rows - pos_rows
mean_delta = display_df['prob_delta'].mean()
mean_delta_norm = display_df['prob_delta_normalized'].mean()
total_questions = display_df['question'].nunique()

blocks = []
for (question, ground_truth), q_df in display_df.groupby(['question', 'ground_truth'], sort=False):
    q_df = q_df.sort_values(by='position_rel')
    trace = q_df['trace'].iloc[0]
    p_init = q_df['prob_init'].iloc[0]
    q_mean_delta = q_df['prob_delta'].mean()
    q_mean_delta_norm = q_df['prob_delta_normalized'].mean()
    q_class = 'sentence-pos' if q_mean_delta >= 0 else 'sentence-neg'
    delta_class = 'delta-pos' if q_mean_delta >= 0 else 'delta-neg'

    sentence_rows = [
        {
            'text': text,
            'position_rel': pos_rel,
            'prob_before': p_before,
            'prob_after': p_after,
            'prob_delta': p_delta,
        }
        for text, pos_rel, p_before, p_after, p_delta in q_df[['text', 'position_rel', 'prob_before', 'prob_after', 'prob_delta']].itertuples(index=False, name=None)
    ]
    trace_html = highlight_spans_in_trace(trace, sentence_rows)

    sentence_chips = ''.join(
        (
            f"<span class='sentence-chip has-tooltip {'sentence-pos-chip' if p_delta >= 0 else 'sentence-neg-chip'}' "
            f"data-tip=\"{html.escape(make_sentence_tooltip(pos_rel, p_before, p_after), quote=True)}\">"
            f"{html.escape((text or '').strip() or '-')}"
            "</span>"
        )
        for text, pos_rel, p_before, p_after, p_delta in q_df[['text', 'position_rel', 'prob_before', 'prob_after', 'prob_delta']].itertuples(index=False, name=None)
    )

    blocks.append(
        f"<div class='sentence-block {q_class}'>"
        "<div class='sentence-header'>"
        f"<span class='delta-badge {delta_class}'>{q_mean_delta:+.6f}</span>"
        f"<span class='delta-sub'>mean normalized {q_mean_delta_norm:+.6f}</span>"
        f"<span class='delta-sub'>prob_init {p_init:.6f}</span>"
        "</div>"
        "<div class='qa-row'>"
        f"<div class='qa-item'><span class='qa-label'>Question</span><div class='qa-text'>{html.escape((question or '').strip() or '-')}</div></div>"
        f"<div class='qa-item'><span class='qa-label'>Ground truth</span><div class='qa-text'>{html.escape((ground_truth or '').strip() or '-')}</div></div>"
        "</div>"
        "<div class='qa-item'>"
        "<span class='qa-label'>Pivotal Sentences (hover for metadata)</span>"
        f"<div class='sentence-list'>{sentence_chips}</div>"
        "</div>"
        "<details class='trace-wrap'>"
        "<summary class='trace-summary'>Trace (all pivotal spans highlighted)</summary>"
        f"<div class='trace-text'>{trace_html}</div>"
        "</details>"
        "</div>"
    )

blocks_html = (
    "<style>"
    ".sentence-wrap{max-height:680px;overflow:auto;padding:10px;border:1px solid #e2e8f0;border-radius:14px;background:#f8fafc;"
    "display:flex;flex-direction:column;gap:12px;}"
    ".viz-summary{position:sticky;top:0;z-index:2;display:flex;flex-wrap:wrap;gap:10px;background:rgba(248,250,252,0.95);"
    "backdrop-filter:blur(4px);padding:6px 2px 10px;}"
    ".summary-pill{padding:6px 10px;border-radius:999px;background:white;border:1px solid #dbe3ef;font-size:12px;color:#1e293b;}"
    ".sentence-block{border:1px solid #dbe3ef;border-left:5px solid #94a3b8;border-radius:12px;padding:12px;background:white;"
    "box-shadow:0 1px 2px rgba(15,23,42,.05);}"
    ".sentence-pos{border-left-color:#16a34a;}"
    ".sentence-neg{border-left-color:#dc2626;}"
    ".sentence-header{display:flex;align-items:center;gap:10px;margin-bottom:8px;}"
    ".delta-badge{font-weight:700;font-size:12px;padding:4px 10px;border-radius:999px;border:1px solid transparent;}"
    ".delta-pos{color:#166534;background:#dcfce7;border-color:#86efac;}"
    ".delta-neg{color:#991b1b;background:#fee2e2;border-color:#fca5a5;}"
    ".delta-sub{font-size:12px;color:#475569;}"
    ".qa-row{display:grid;grid-template-columns:1fr;gap:8px;margin-bottom:10px;}"
    ".qa-item{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:8px 10px;}"
    ".qa-label{display:block;font-size:11px;font-weight:600;color:#475569;margin-bottom:3px;text-transform:uppercase;letter-spacing:.02em;}"
    ".qa-text{font-size:13px;line-height:1.35;color:#0f172a;white-space:pre-wrap;}"
    ".sentence-list{display:flex;flex-wrap:wrap;gap:6px;}"
    ".sentence-chip{display:inline-flex;max-width:100%;border:1px solid #cbd5e1;border-radius:999px;padding:4px 8px;"
    "font-size:11px;line-height:1.3;color:#0f172a;white-space:pre-wrap;}"
    ".sentence-pos-chip{background:#dcfce7;border-color:#86efac;}"
    ".sentence-neg-chip{background:#fee2e2;border-color:#fca5a5;}"
    ".trace-wrap{margin-bottom:10px;border:1px solid #e2e8f0;border-radius:10px;background:#f8fafc;}"
    ".trace-summary{cursor:pointer;font-size:12px;font-weight:600;color:#334155;padding:8px 10px;}"
    ".trace-text{border-top:1px solid #e2e8f0;padding:10px;white-space:pre-wrap;line-height:1.4;font-size:12px;color:#0f172a;max-height:280px;overflow:auto;}"
    ".trace-text .trace-hit{padding:0 1px;}"
    ".trace-text .trace-pos{background:#dcfce7;}"
    ".trace-text .trace-neg{background:#fee2e2;}"
    ".has-tooltip{position:relative;cursor:help;}"
    ".has-tooltip:hover::after{content:attr(data-tip);position:absolute;left:0;bottom:calc(100% + 6px);z-index:30;"
    "background:#0f172a;color:#f8fafc;font-size:11px;line-height:1.3;padding:6px 8px;border-radius:6px;white-space:nowrap;"
    "box-shadow:0 6px 18px rgba(2,6,23,.28);}"
    ".sentence-meta{display:flex;flex-wrap:wrap;gap:7px;}"
    ".pill{display:inline-flex;gap:6px;align-items:center;background:#f1f5f9;border:1px solid #dbe3ef;border-radius:999px;"
    "padding:4px 9px;font-size:11px;color:#334155;}"
    ".pill b{font-weight:600;color:#0f172a;}"
    "</style>"
    "<div class='sentence-wrap'>"
    "<div class='viz-summary'>"
    f"<span class='summary-pill'><b>Questions</b> {total_questions}</span>"
    f"<span class='summary-pill'><b>Pivotal sentences</b> {total_rows}</span>"
    f"<span class='summary-pill'><b>Positive</b> {pos_rows}</span>"
    f"<span class='summary-pill'><b>Negative</b> {neg_rows}</span>"
    f"<span class='summary-pill'><b>Mean delta</b> {mean_delta:+.6f}</span>"
    f"<span class='summary-pill'><b>Mean normalized</b> {mean_delta_norm:+.6f}</span>"
    "</div>"
    + "".join(blocks)
    + "</div>"
)

display(HTML(blocks_html))

w# %%

df.groupby(df['span_text'].str.lower().str.contains('answer is'), as_index=True).agg({'prob_delta': ('max', 'min', 'mean', 'median')})
# %%
