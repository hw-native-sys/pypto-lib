# Metadata artifact ownership

Only worker_metadata/** is writable. All product files/Main/oldbundles remain read-only.
The script is prepared only; execution waits Main exact frozen source snapshot.
No checkpoint headers/index/tokenizer/payload reads, fixture, prepare, compile or NPU.
