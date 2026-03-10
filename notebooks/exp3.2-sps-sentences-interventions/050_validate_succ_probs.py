# %%
from pathlib import Path
import pandas as pd


SUCC_PROBS_CSV = Path(
    # "/home/xaparo00/workspace/projects/pivotal_tokens/data/experiments/exp3.2-sps-sentences-interventions/exp3.2.1-qwen3-1.7b-sps-sentence-interventions/data/interventions_succ_prob.csv"
    "/home/xaparo00/workspace/projects/pivotal_tokens/data/experiments/exp3.2-sps-sentences-interventions/exp3.2.2-qwen3-1.7b-sps-rand-sentence-interventions/data/interventions_rand_succ_prob.csv"
)

# %%

df = pd.read_csv(SUCC_PROBS_CSV)
# %%
df[['prob_before_original', 'prob_before_interv', 'intervention_type']]

# %%
df[['prob_after_original', 'prob_after_interv', 'intervention_type']]

# %%

df[['prob_delta_original', 'prob_delta_interv', 'intervention_type']]

# %%

print('TRACE ORIGINAL')
print(df.loc[0, 'trace_original_normalized'])

print('TRACE INTERV')
print(df.loc[0, 'trace_interv_normalized'])

# %%

print('TRACE ORIGINAL')
print(df.loc[0, 'pivotal_context'])

print('TRACE INTERV')
print(df.loc[0, 'pivotal_context_interv'])

# %%

df[['pivot_token_position_original', 'pivot_sentence_position_original']]

# %%
df[['trace_token_length_original', 'trace_sentence_length_original']]

# %%

df
# %%
