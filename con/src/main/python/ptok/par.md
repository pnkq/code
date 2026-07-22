
DatasetBuilderV2
        │
        ▼
BytePartitioner
        │
        ▼
16 workers
        │
        ├── CorpusReader(start,end)
        ├── HybridTokenizer
        ├── SequencePacker
        └── MemMapWriter(part_i.bin)
        │
        ▼
DatasetMerger
        │
        ▼
training.bin
