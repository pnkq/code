from t.corpus import CorpusReader, CorpusShard
from t.packer import SequencePacker
from tqdm import tqdm

from t.partitioner import BytePartitioner
from t.worker import WorkerBuilder
from t.merger import DatasetMerger
from t.stats import BuildStats

from multiprocessing import Process
from pathlib import Path
import tempfile
import os

class DatasetBuilderPar:
    """
    Parallel version of DatasetBuilder.
    """

    def __init__(self, sequence_length, drop_last=True, num_workers=8):
        self.sequence_length = sequence_length
        self.drop_last = drop_last
        self.num_workers = num_workers

    def build(self, corpus_file, output_file):
        temp_dir = Path(tempfile.mkdtemp(prefix="dataset_builder_"))
        partitioner = BytePartitioner()
        partitions = partitioner.partition(corpus_file, self.num_workers)

        processes = []
        part_files = []

        for worker_id, (start, end) in enumerate(partitions):
            part_file = temp_dir / f"part_{worker_id:03d}.bin"
            part_files.append(part_file)
            process = Process(target=self._worker, args=(worker_id, corpus_file, start, end, part_file))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()
        
        # for i, process in enumerate(processes):
        #     print(f"Worker {i}: exit code = {process.exitcode}")

        # for part in part_files:
        #     print(part)
        #     print("exists:", os.path.exists(part))

        DatasetMerger().merge(part_files, output_file)

    def _worker(self, worker_id, corpus_file, start, end, output_file):
        try:
            shard = CorpusShard(corpus_file, start, end)
            builder = WorkerBuilder(
                id=worker_id, 
                shard=shard, 
                output_file=output_file, 
                sequence_length=self.sequence_length, 
                drop_last=self.drop_last
            )
            builder.build()
        except Exception:
            import traceback
            traceback.print_exc()
            raise

class DatasetBuilder:

    def __init__(self, tokenizer, corpus_dir, sequence_length=512, drop_last=True):
        self.tokenizer = tokenizer
        self.reader = CorpusReader(corpus_dir)
        self.packer = SequencePacker(sequence_length)
        self.drop_last = drop_last
        self.stats = BuildStats()

    def build(self):
        progress = tqdm(unit=" tokens")
        pending = 0
        for line in self.reader.documents():
            before = self.packer.tokens_processed
            ids = self.tokenizer.encode_text(line)
            for seq in self.packer.add(ids):
                yield self._finalize(seq)
            count = self.packer.tokens_processed - before
            self.stats.lines += 1
            self.stats.pieces += count
            pending += count
            if pending >= 10000:
                progress.update(pending)
                pending = 0

        if pending > 0:
            progress.update(pending)

        for seq in self.packer.flush(drop_last=self.drop_last, pad_id=self.tokenizer.pad_token_id):
            yield self._finalize(seq)

        progress.close()

    def _finalize(self, seq):
        self.stats.sequences += 1
        return self.tokenizer.build_inputs_with_special_tokens(seq)
    

