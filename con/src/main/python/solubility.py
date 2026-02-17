import pandas as pd
import torch
from transformers import AutoTokenizer, EsmModel
import pyarrow as pa
import pyarrow.parquet as pq


# 1. Load the model and tokenizer
# Smaller models: 'esm2_t6_8M_UR50D', 'esm2_t12_35M_UR50D'
# Larger models: 'esm2_t30_150M_UR50D', 'esm2_t33_650M_UR50D'
#model_name = "facebook/esm2_t6_8M_UR50D"
model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = EsmModel.from_pretrained(model_name)

# 2. Load protein sequences from a CSV file

#file_name = "S.cerevisiae_test"
#file_name = "eSol_train"
file_name = "eSol_test"
df = pd.read_csv(f"/Users/phuonglh/code/con/dat/sol/{file_name}.csv")

# Basic Data Cleaning
# It's a good idea to remove any rows with missing sequences 
# or extra whitespace that might break the model.
df = df.dropna(subset=['sequence'])
df['sequence'] = df['sequence'].str.strip().str.upper()

# Extract the sequences as a list
sequences = df['sequence'].tolist()

# Optional: Map genes to sequences (to keep track of metadata)
gene_to_seq = dict(zip(df['gene'], df['sequence']))

print(f"Loaded {len(sequences)} sequences.")
print(f"First gene: {df['gene'].iloc[0]}")
print(f"First sequence: {sequences[0][:20]}...")


print("Compute gene sequence embeddings...")
# Processing in batches
batch_size = 16
all_embeddings = []

for i in range(0, len(sequences), batch_size):
    batch_seqs = sequences[i : i + batch_size]
    
    # Tokenize
    inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True)
    
    # Generate Embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        # Mean pooling to get one vector per protein
        batch_embeddings = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(batch_embeddings)

# Combine all batches into one large tensor
final_embeddings = torch.cat(all_embeddings, dim=0)

print(f"Final embedding matrix shape: {final_embeddings.shape}")


# 1. Convert the PyTorch tensor to a NumPy array
# Ensure it's on CPU and detached from any computation graphs
embedding_array = final_embeddings.cpu().numpy()

# 2. Create a DataFrame for the embeddings
# Each dimension of the ESM embedding becomes a column (0, 1, 2...)
embedding_df = pd.DataFrame(
    embedding_array, 
    columns=[f"dim_{i}" for i in range(embedding_array.shape[1])]
)

# 3. Combine with your original metadata (gene and solubility)
# Use reset_index to ensure the row alignment is perfect
metadata_df = df[['gene', 'solubility']].reset_index(drop=True)
final_df = pd.concat([metadata_df, embedding_df], axis=1)

# 4. Save to Parquet
final_df.to_parquet(f"/Users/phuonglh/code/con/dat/sol/{file_name}_embeddings_650M.parquet", engine='pyarrow', compression='snappy')

print(f"Successfully saved to Parquet! File size is roughly {final_df.memory_usage().sum() / 1e6:.2f} MB")

