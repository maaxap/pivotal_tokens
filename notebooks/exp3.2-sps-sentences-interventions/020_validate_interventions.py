# %%
import html
import json
from pathlib import Path

import pandas as pd
from IPython.display import HTML, display
from nltk.tokenize import sent_tokenize


# %%
INTERVENTIONS_CSV = Path(
    "/home/xaparo00/workspace/projects/pivotal_tokens/data/experiments/exp3.2-sps-sentences-interventions/exp3.2.1-qwen3-1.7b-sps-sentence-interventions/data/interventions.csv"
    # "/home/xaparo00/workspace/projects/pivotal_tokens/data/experiments/exp3.2-sps-sentences-interventions/exp3.2.2-qwen3-1.7b-sps-rand-sentence-interventions/data/interventions_rand.csv"
)

INTERVENTION_DISPLAY_ORDER = [
    "delete_sentence",
    "shuffle_preceding_context",
    "replace_with_probable",
    "replace_with_noise",
    "move_sentence_beginning",
    "move_sentence_middle",
    "move_sentence_end",
    "transplant_sentence",
]

COLUMNS_TO_HIDE_IN_OVERVIEW = [
    "metadata",
    "metadata_dict",
    "trace_original",
    "trace_interv",
    "trace_original_sentences",
    "trace_interv_sentences",
]

pd.set_option("display.max_colwidth", 140)


# %%
def to_sentences(text: str) -> list[str]:
    return sent_tokenize(text, language="english")


def render_sentence_list(sentences: list[str], pivot_index: int | None, title: str) -> str:
    html_parts = [f"<h4 style='margin:0 0 8px 0'>{html.escape(title)}</h4>"]

    for sentence_index, sentence in enumerate(sentences):
        is_pivot_sentence = pivot_index is not None and sentence_index == pivot_index
        if is_pivot_sentence:
            pivot_label = " <b style='color:#a40000'>(pivot)</b>"
            background_color = "#fff6d6"
        else:
            pivot_label = ""
            background_color = "#fafafa"

        html_parts.append(
            "<div style='padding:8px;margin:4px 0;border:1px solid #ddd;border-radius:6px;"
            f"background:{background_color}'>"
            f"<code>{sentence_index}</code>{pivot_label}<br>{html.escape(sentence)}"
            "</div>"
        )

    return "".join(html_parts)


def render_intervention_card(row: pd.Series) -> None:
    metadata = row["metadata_dict"]

    original_pivot_index = int(row["pivot_sentence_position_original"])
    if "pivot_sentence_position_interv" in metadata:
        intervened_pivot_index = int(metadata["pivot_sentence_position_interv"])
    else:
        intervened_pivot_index = None

    title = f"{row['intervention_type']} | sample={row['sample_id']} | span={row['span_id']}"

    details = (
        f"<b>span_text:</b> {html.escape(str(row['span_text']))}<br>"
        f"<b>span_text_interv:</b> {html.escape(str(row['span_text_interv']))}<br>"
        f"<b>intervention_id:</b> {html.escape(str(row['intervention_id']))}"
    )

    original_trace_html = render_sentence_list(
        sentences=row["trace_original_sentences"],
        pivot_index=original_pivot_index,
        title="trace_original",
    )
    intervened_trace_html = render_sentence_list(
        sentences=row["trace_interv_sentences"],
        pivot_index=intervened_pivot_index,
        title="trace_interv",
    )

    card_html = (
        "<div style='border:1px solid #bbb;border-radius:10px;padding:12px;margin:12px 0'>"
        f"<h3 style='margin:0 0 8px 0'>{html.escape(title)}</h3>"
        f"<div style='margin:0 0 10px 0'>{details}</div>"
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px'>"
        f"<div>{original_trace_html}</div>"
        f"<div>{intervened_trace_html}</div>"
        "</div>"
        "</div>"
    )

    display(HTML(card_html))


# %%
# %%
interventions_df = pd.read_csv(INTERVENTIONS_CSV)
interventions_df["metadata_dict"] = interventions_df["metadata"].apply(json.loads)
interventions_df["trace_original_sentences"] = interventions_df["trace_original"].apply(to_sentences)
interventions_df["trace_interv_sentences"] = interventions_df["trace_interv"].apply(to_sentences)

print(f"Loaded {len(interventions_df)} rows from {INTERVENTIONS_CSV}")
print("Intervention counts:")
display(interventions_df["intervention_type"].value_counts().rename("count").to_frame())
display(interventions_df.head(3))


# %%
# Compact table for quick manual scanning.
intervention_order_map = {
    intervention_type: index for index, intervention_type in enumerate(INTERVENTION_DISPLAY_ORDER)
}

overview_df = interventions_df.copy()
overview_df["_order"] = overview_df["intervention_type"].map(intervention_order_map)
overview_df = overview_df.sort_values(["sample_id", "span_id", "_order"])
overview_df = overview_df.drop(columns=["_order", *COLUMNS_TO_HIDE_IN_OVERVIEW])

display(overview_df)


# %%
# Full visualization: every intervention card for every (sample_id, span_id).
intervention_order_map = {
    intervention_type: index for index, intervention_type in enumerate(INTERVENTION_DISPLAY_ORDER)
}

visualization_df = interventions_df.copy()
visualization_df["_order"] = visualization_df["intervention_type"].map(intervention_order_map)
visualization_df = visualization_df.sort_values(["sample_id", "span_id", "_order"])

for (sample_id, span_id), group_df in visualization_df.groupby(["sample_id", "span_id"], sort=False):
    section_title = f"sample={sample_id} | span={span_id}"
    display(HTML("<hr style='margin:22px 0'>" f"<h2 style='margin:0'>{html.escape(section_title)}</h2>"))

    for intervention_type in INTERVENTION_DISPLAY_ORDER:
        row_df = group_df[group_df["intervention_type"] == intervention_type]
        row = row_df.iloc[0]
        render_intervention_card(row)

# %%
def render_side_by_side_cells(
    left_title: str,
    left_text: str,
    right_title: str,
    right_text: str,
) -> str:
    return (
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px'>"
        "<div style='padding:10px;border:1px solid #ddd;border-radius:8px;background:#fafafa'>"
        f"<h4 style='margin:0 0 8px 0'>{html.escape(left_title)}</h4>"
        f"<div style='white-space:pre-wrap'>{html.escape(left_text)}</div>"
        "</div>"
        "<div style='padding:10px;border:1px solid #ddd;border-radius:8px;background:#fafafa'>"
        f"<h4 style='margin:0 0 8px 0'>{html.escape(right_title)}</h4>"
        f"<div style='white-space:pre-wrap'>{html.escape(right_text)}</div>"
        "</div>"
        "</div>"
    )


for (sample_id, span_id), group_df in visualization_df.groupby(["sample_id", "span_id"], sort=False):
    section_title = f"sample={sample_id} | span={span_id}"
    display(HTML("<hr style='margin:22px 0'>" f"<h2 style='margin:0'>{html.escape(section_title)}</h2>"))

    for intervention_type in INTERVENTION_DISPLAY_ORDER:
        row_df = group_df[group_df["intervention_type"] == intervention_type]
        row = row_df.iloc[0]
        metadata = row["metadata_dict"]

        pivotal_prefix_original = str(row["pivotal_context"])
        pivotal_prefix_interv = str(row["pivotal_context_interv"])
        block_html = (
            "<div style='border:1px solid #bbb;border-radius:10px;padding:12px;margin:12px 0'>"
            f"<h3 style='margin:0 0 10px 0'>{html.escape(str(intervention_type))}</h3>"
            f"{render_side_by_side_cells('pivotal_prefix_original', pivotal_prefix_original, 'pivotal_prefix_interv', pivotal_prefix_interv)}"
            "<div style='height:10px'></div>"
            f"{render_side_by_side_cells('span_text', str(row['span_text']), 'span_text_interv', str(row['span_text_interv']))}"
            "</div>"
        )
        display(HTML(block_html))
