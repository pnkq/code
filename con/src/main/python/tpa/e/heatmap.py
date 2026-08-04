import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoModel, AutoTokenizer


def plot_attention_head(attention, tokens, layer_idx=0, head_idx=0):
    """Plots a 2D heatmap of attention weights for a given layer and head."""
    # Extract attention matrix for the chosen layer and head
    # Shape: (seq_len, seq_len)
    attn_matrix = attention[layer_idx][0, head_idx].cpu().numpy()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        attn_matrix,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        cbar=True,
    )
    plt.title(f"RoBERTa Attention: Layer {layer_idx + 1}, Head {head_idx + 1}")
    plt.xlabel("Key Tokens (Attended To)")
    plt.ylabel("Query Tokens (Attending From)")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


# --- Execution ---
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModel.from_pretrained("roberta-base", output_attentions=True)

text = "The dog didn't cross the street because it was too tired."
inputs = tokenizer(text, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

with torch.no_grad():
    outputs = model(**inputs)

# Plot Layer 6, Head 4
plot_attention_head(outputs.attentions, tokens, layer_idx=5, head_idx=3)

