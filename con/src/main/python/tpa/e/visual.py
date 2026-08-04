import torch
from bertviz import head_view, model_view
from transformers import AutoModel, AutoTokenizer

# 1. Load RoBERTa-base and enable output_attentions
model_name = "roberta-base"  # or your custom/fine-tuned model checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)

# 2. Prepare sample input text
text = "The dog didn't cross the street because it was too tired."
inputs = tokenizer(text, return_tensors="pt")

# Convert token IDs back to human-readable subword tokens
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# 3. Model forward pass
with torch.no_grad():
    outputs = model(**inputs)

# outputs.attentions is a tuple of 12 tensors (one per layer)
# Each tensor has shape: (batch_size, num_heads, seq_len, seq_len)
attention = outputs.attentions

# 4. Render Interactive Head View
# Shows attention lines between tokens for specific layers/heads
head_view(attention, tokens)

# 5. Render Interactive Model View
# Shows a bird's-eye view across all 12 layers and 12 heads simultaneously
model_view(attention, tokens)
