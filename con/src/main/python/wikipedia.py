# Convert all arrow files to a single Spark DataFrame
# This utility is used to read all *.arrow of the Vietnamse Wikipedia data and convert to a single *.parquet file.
# The *.arrow data files are downloaed from the HF: https://huggingface.co/datasets/wikimedia/wikipedia
# phuonglh@gmail.com 
# 
import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path
import pyarrow.parquet as pq


DIR = "/home/phuonglh/.cache/huggingface/datasets/wikimedia___wikipedia/20231101.vi/0.0.0/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/"
arrow_files = Path(DIR).glob("*.arrow")

tables = []

for f in sorted(arrow_files):
    with pa.memory_map(str(f), "r") as source:
        tables.append(ipc.open_stream(source).read_all())

combined = pa.concat_tables(tables)

# Write to a single parquet file
pq.write_table(combined, "20231101.parquet")





