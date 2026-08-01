1. Parallel dataset builder


        DatasetBuilderPar
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

