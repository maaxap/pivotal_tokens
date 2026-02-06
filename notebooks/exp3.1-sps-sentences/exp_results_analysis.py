# %%
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
from transformers import PreTrainedTokenizer

from pivotal_tokens.constants import get_data_dir, get_artifacts_dir
from pivotal_tokens.hf.loading import  load_tokenizer
from pivotal_tokens.repo import SampleRepo, DictRepo


EXP_DIR = get_data_dir() / 'experiments' / 'exp3.1-sps-sentences' / 'exp3.1.1-qwen3-1.7b-sps-sentences'
REPO_DIR = EXP_DIR / 'data' / 'repo'
PIVOTAL_TOKENS_FILE = EXP_DIR / 'data' / 'pivotal_sentences.csv'

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
df['sentence_token_length'] = df['token_ids'].str.len()
df['trace_token_length'] = df['trace'].apply(lambda x: get_token_length(x, tokenizer))
df['sentence_start_token_position'] = df['partial_trace_token_length'] + 1
df['sentence_start_relative_position'] = df['sentence_start_token_position'] / df['trace_token_length']

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

# TODO: Implement below the following chart. Each chart should be implemented in a separate cell
# 1. Гистограмма ΔP(success) после pivotal sentence — распределение прироста после ключевого предложения.
# 2. Scatter: ΔP(success) vs длина предложения (в токенах) — зависит ли эффект от длины ключевого предложения.
# 3. Scatter: ΔP(success) vs позиция предложения в trace (позиция в токенах, начало предложения) — эффект в зависимости от места в цепочке.
# 4. Histogram: P_after(success) после pivotal sentence — насколько “уверенным” становится успех после предложения.
# 5. Boxplot ΔP(success) по квантилям P0 — как эффект предложений меняется с сложностью вопроса. Квантили должны быть показаны как quantile ranges и должны быть отсортированы от меньшего к большему. 
# 6. Bar: доля pivotal sentences, содержащих ответ — распределение: сколько ключевых предложений включает строку ответа.
