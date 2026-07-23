from t.corpus import CorpusReader, CorpusShard
from t.packer import SequencePacker
from tqdm import tqdm

from multiprocessing import RLock
tqdm.set_lock(RLock())

from t.partitioner import BytePartitioner
from t.worker import WorkerBuilder
from t.merger import DatasetMerger
from t.stats import BuildStats

from multiprocessing import Process, Value
from pathlib import Path
import tempfile
from ctypes import c_longlong
import threading
import time


class DatasetBuilderPar:
    """
    Parallel version of DatasetBuilder.
    """
    def __init__(self, sequence_length, drop_last=True, num_workers=8):
        self.sequence_length = sequence_length
        self.drop_last = drop_last
        self.num_workers = num_workers

    def monitor(self, progress, processes):
        total = 0
        # fmt = "{desc}: {percentage:3.0f}% |{bar}| {n:,d}/{total:,d} [{elapsed}<{remaining}]"
        fmt="{desc}: {n:,d} {unit} [{elapsed}, {rate_fmt}]"
        bar = tqdm(unit=" tokens", desc="Tokenizing", bar_format=fmt)

        while True:
            current = sum(counter.value for counter in progress)
            bar.update(current - total)
            total = current
            if all(not p.is_alive() for p in processes):
                break
            time.sleep(0.5)

        current = sum(counter.value for counter in progress)
        bar.update(current - total)

        bar.close()

    def build(self, corpus_file, output_file):
        temp_dir = Path(tempfile.mkdtemp(prefix="dataset_builder_"))
        partitioner = BytePartitioner()
        partitions = partitioner.partition(corpus_file, self.num_workers)

        processes = []
        part_files = []
        # shared progress counters, each counter stores the number of tokens processed by one worker.
        progress = [Value(c_longlong, 0) for _ in range(self.num_workers)]

        for worker_id, (start, end) in enumerate(partitions):
            part_file = temp_dir / f"part_{worker_id:03d}.bin"
            part_files.append(part_file)
            process = Process(target=self._worker, args=(worker_id, corpus_file, start, end, part_file, progress[worker_id]))
            process.start()
            processes.append(process)

        monitor_thread = threading.Thread(target=self.monitor, args=(progress, processes))
        monitor_thread.start()

        for process in processes:
            process.join()

        monitor_thread.join()
        
        # for i, process in enumerate(processes):
        #     print(f"Worker {i}: exit code = {process.exitcode}")

        # for part in part_files:
        #     print(part)
        #     print("exists:", os.path.exists(part))

        DatasetMerger().merge(part_files, output_file)

    def _worker(self, worker_id, corpus_file, start, end, output_file, counter):
        try:
            shard = CorpusShard(corpus_file, start, end)
            builder = WorkerBuilder(
                id=worker_id, 
                shard=shard, 
                output_file=output_file, 
                sequence_length=self.sequence_length, 
                drop_last=self.drop_last,
                progress_counter=counter
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
    

