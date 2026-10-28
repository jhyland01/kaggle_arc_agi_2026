import os
import threading
import time

from concurrent.futures import as_completed, ThreadPoolExecutor

def precache(*paths, times=1, delay=0, threads=16) -> None:
    time.sleep(delay)
    print(f'Precaching: Started...')
    start_time = time.time()

    CHUNK_SIZE = 32 * 1024 * 1024  # 128 MB

    chunks = []  # list of (path, offset, length)
    total_size = 0
    file_count = 0

    def _add_file(i: int, path: str) -> None:
        nonlocal total_size, file_count
        size = os.path.getsize(path)
        total_size += size
        file_count += 1
        for offset in range(0, max(size, 1), CHUNK_SIZE):
            length = min(CHUNK_SIZE, size - offset)
            chunks.append((i, path, offset, length))

    for i, path in enumerate(paths):
        try:
            if os.path.isfile(path):
                _add_file(i, path)
                continue
            for root, _, files in os.walk(path):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    if os.path.isfile(file_path):
                        _add_file(i, file_path)
        except: pass

    chunks.sort()

    _local = threading.local()

    def _read_chunk(chunk: tuple) -> None:
        try:
            buf = _local.buf
        except AttributeError:
            _local.buf = buf = bytearray(CHUNK_SIZE)
        i, path, offset, _ = chunk
        try:
            with open(path, 'rb') as f:
                f.seek(offset)
                remaining = CHUNK_SIZE
                for i in range(1024):
                    n = f.readinto(buf)
                    if not n:
                        break
                    remaining -= n
                    if remaining <= 0: break
        except: pass

    with ThreadPoolExecutor(max_workers=threads) as pool:
        for _ in range(times):
            list(pool.map(_read_chunk, chunks))

    elapsed = time.time() - start_time
    print(f'Precaching: loaded {file_count} files ({total_size / 1e9:.2f} GB) in {elapsed:.2f} seconds.\n')

precache_wheels_thread = threading.Thread(target=precache, args=(WHEELHOUSE_DIR,))
precache_wheels_thread.start()
precache_wheels_thread.join()

precache_model_thread = threading.Thread(target=precache, args=(MODEL_DIR, DRAFT_MODEL_DIR), kwargs={'delay': 60, 'threads': 3})