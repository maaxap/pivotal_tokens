# %%
from pathlib import Path
import pandas as pd


INTERVENTIONS_CSV = Path(
    "/home/xaparo00/workspace/projects/pivotal_tokens/data/experiments/exp3.2-sps-sentences-interventions/exp3.2.1-qwen3-1.7b-sps-sentence-interventions/data/interventions.csv"
    # "/home/xaparo00/workspace/projects/pivotal_tokens/data/experiments/exp3.2-sps-sentences-interventions/exp3.2.2-qwen3-1.7b-sps-rand-sentence-interventions/data/interventions_rand.csv"
)

# %%

interventions_df = pd.read_csv(INTERVENTIONS_CSV)
interventions_df.isna().mean()

# %%

# pivot_last_sentence_filter = interventions_df.span_text_interv.isna()
# interventions_df = interventions_df[~pivot_last_sentence_filter]

# transplant_sentence_filter = (interventions_df.pivotal_context_interv.isna()) & \
#     (interventions_df.intervention_type == 'transplant_sentence')
# interventions_df = interventions_df[~transplant_sentence_filter]

# open_think_tag_absent_filter = (interventions_df.span_text.str.startswith("<think>")) & \
#     (~interventions_df.span_text_interv.str.startswith("<think>"))
# interventions_df.loc[open_think_tag_absent_filter, 'span_text_interv'] = "<think>\n" + interventions_df.loc[open_think_tag_absent_filter, 'span_text_interv']

# shuffle_preceding_context_filter = (interventions_df.pivotal_context_interv.isna()) & \
#     (interventions_df.intervention_type == 'shuffle_preceding_context')
# interventions_df.loc[shuffle_preceding_context_filter, 'pivotal_context_interv'] = interventions_df.loc[shuffle_preceding_context_filter, 'pivotal_context']

# replace_with_probable_filter = (interventions_df.pivotal_context_interv.isna()) & \
#     (interventions_df.intervention_type == 'replace_with_probable')
# interventions_df.loc[replace_with_probable_filter, 'pivotal_context_interv'] = interventions_df.loc[replace_with_probable_filter, 'pivotal_context']

# replace_with_noise_filter = (interventions_df.pivotal_context_interv.isna()) & \
#     (interventions_df.intervention_type == 'replace_with_noise')
# interventions_df.loc[replace_with_noise_filter, 'pivotal_context_interv'] = interventions_df.loc[replace_with_noise_filter, 'pivotal_context']

# move_sentence_beginning_filter = (interventions_df.pivotal_context_interv.isna()) & \
#     (interventions_df.intervention_type == 'move_sentence_beginning')
# interventions_df.loc[move_sentence_beginning_filter, 'pivotal_context_interv'] = interventions_df.loc[move_sentence_beginning_filter, 'pivotal_context']

# incomplete_pivotal_context_filter = (~interventions_df.pivotal_context_interv.str.startswith('<|im_start|>')) & \
#     (interventions_df.span_text.str.startswith("<think>"))
# # interventions_df[incomplete_pivotal_context_filter].loc[5, 'pivotal_context']
# interventions_df.loc[incomplete_pivotal_context_filter, 'pivotal_context_interv'] = interventions_df.loc[incomplete_pivotal_context_filter, 'pivotal_context'] + interventions_df.loc[incomplete_pivotal_context_filter, 'pivotal_context_interv']
# %%


def extract_context(source_pivotal_context):
    splitter_str = "<|im_start|>assistant\n"
    context = source_pivotal_context.split(splitter_str)[0] + splitter_str
    return context

def extract_trace(source_pivotal_context):
    splitter_str = "<|im_start|>assistant\n"
    trace = source_pivotal_context.split(splitter_str)[1]
    return trace

# %%

# interventions_df.loc[:, 'trace_original'] = interventions_df.loc[:, 'trace_original'].apply(extract_trace)
# interventions_df.loc[:, 'trace_interv'] = interventions_df.loc[:, 'trace_interv'].apply(extract_trace)

# %%

# interventions_df.loc[:, 'trace_original'] = interventions_df.loc[:, 'source_pivotal_context'].apply(extract_context) + interventions_df.loc[:, 'trace_original']
# interventions_df.loc[:, 'trace_interv'] = interventions_df.loc[:, 'source_pivotal_context'].apply(extract_context) + interventions_df.loc[:, 'trace_interv']


# %%

RAND = False
if RAND:
    incomplete_pivotal_context_interv_filter = (~interventions_df.pivotal_context_interv.str.startswith('<|im_start|>'))
    interventions_df.loc[incomplete_pivotal_context_interv_filter, 'pivotal_context_interv'] = \
        interventions_df.loc[incomplete_pivotal_context_interv_filter, 'source_pivotal_context'].apply(extract_context) + \
        "<think>\n" + \
        interventions_df.loc[incomplete_pivotal_context_interv_filter, 'pivotal_context_interv']

    incomplete_pivotal_context_filter = (~interventions_df.pivotal_context.str.startswith('<|im_start|>'))
    interventions_df.loc[incomplete_pivotal_context_filter, 'pivotal_context'] = \
        interventions_df.loc[incomplete_pivotal_context_filter, 'source_pivotal_context'].apply(extract_context) + \
        "<think>\n" + \
        interventions_df.loc[incomplete_pivotal_context_filter, 'pivotal_context']

# %%


# %%

# interventions_df.to_csv(INTERVENTIONS_CSV, index=False)

# %%
