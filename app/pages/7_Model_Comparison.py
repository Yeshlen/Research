import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from utils import MODEL_NAMES, load_resources, model_label, page_setup, stage_control

page_setup("Model Comparison")
stage = stage_control()
_, results, _ = load_resources()
st.title("Model Comparison")
st.write("Saved test-period results using settings and thresholds selected on validation data.")
table = results[results.stage == stage].set_index("model").loc[MODEL_NAMES]
display = table[["precision", "recall", "f1", "fpr"]].copy()
display.columns = ["Precision", "Recall", "F1", "False positive rate"]
display.index = display.index.map(model_label)
st.dataframe(display.style.format({"Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.3f}",
                                    "False positive rate": "{:.2%}"}), width="stretch")
fig, ax = plt.subplots(figsize=(8, 3.5))
x = np.arange(len(table))
for offset, metric, colour, label in [(-.25, "precision", "#315A7D", "Precision"),
                                      (0, "recall", "#B86E32", "Recall"),
                                      (.25, "f1", "#63856C", "F1")]:
    ax.bar(x + offset, table[metric], width=.25, label=label, color=colour)
ax.set(xticks=x, xticklabels=[model_label(name) for name in MODEL_NAMES], ylim=(0, 1), ylabel="Score",
       title=f"Test performance at {'creation' if stage == 'A' else 'redemption'}")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
st.pyplot(fig, width="content")
plt.close(fig)
st.caption("Generated demonstrations are excluded. Higher detection can come with more false alerts.")
