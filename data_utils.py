import os
import random
import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from config import GPTConfig
from tokenizer import make_tokenizer


@dataclass
class DatasetBundle:
    name: str
    train_tokens: np.ndarray
    val_tokens: np.ndarray
    test_tokens: np.ndarray
    tokenizer: object
    vocab_size: int


def _load_enwik8_bytes(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "enwik8.zip")
    if not os.path.exists(zip_path):
        enwik8_url = "http://mattmahoney.net/dc/enwik8.zip"
        print(f"downloading enwik8 to {zip_path} ...")
        urllib.request.urlretrieve(enwik8_url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        raw = zf.read("enwik8")

    assert len(raw) >= 100_000_000, "enwik8 must be at least 100M bytes"
    train_raw = raw[:90_000_000]
    val_raw = raw[90_000_000:95_000_000]
    test_raw = raw[95_000_000:100_000_000]
    return train_raw, val_raw, test_raw


def _load_tinyshakespeare_text(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    txt_path = os.environ.get("TINY_SHAKESPEARE_PATH", os.path.join(data_dir, "tinyshakespeare.txt"))
    if not os.path.exists(txt_path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"downloading tinyshakespeare to {txt_path} ...")
        try:
            urllib.request.urlretrieve(url, txt_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download tinyshakespeare from {url}. "
                f"Place a local file at {txt_path} or set TINY_SHAKESPEARE_PATH. "
                f"Original error: {e}"
            ) from e

    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    n = len(text)
    n_train = int(0.9 * n)
    n_val = int(0.05 * n)
    train_text = text[:n_train]
    val_text = text[n_train:n_train + n_val]
    test_text = text[n_train + n_val:]
    return train_text, val_text, test_text


def _build_stream_and_doc_starts(docs_list, tokenizer):
    bos_token = tokenizer.bos_id
    tokens = [bos_token]
    doc_starts = [0]
    for d in docs_list:
        tokens.extend(tokenizer.encode_text(d))
        tokens.append(bos_token)
        doc_starts.append(len(tokens) - 1)
    return np.array(tokens, dtype=np.int32), np.array(doc_starts, dtype=np.int32)


def load_dataset(dataset_name: str, data_dir: str, tokenizer_mode: str = "auto") -> DatasetBundle:
    dataset_name = dataset_name.strip().lower()
    mode = tokenizer_mode.strip().lower()
    resolved_mode = ("byte" if dataset_name == "enwik8" else "char") if mode == "auto" else mode

    if dataset_name == "enwik8":
        train_raw, val_raw, test_raw = _load_enwik8_bytes(data_dir)
        docs_for_char = [train_raw.decode("latin-1"), val_raw.decode("latin-1"), test_raw.decode("latin-1")]
        tokenizer = make_tokenizer(resolved_mode, dataset_name, docs_all=docs_for_char)

        train_tokens = np.frombuffer(train_raw, dtype=np.uint8).astype(np.int32)
        val_tokens = np.frombuffer(val_raw, dtype=np.uint8).astype(np.int32)
        test_tokens = np.frombuffer(test_raw, dtype=np.uint8).astype(np.int32)

        if resolved_mode == "char":
            train_tokens = np.array(tokenizer.encode_text(docs_for_char[0]), dtype=np.int32)
            val_tokens = np.array(tokenizer.encode_text(docs_for_char[1]), dtype=np.int32)
            test_tokens = np.array(tokenizer.encode_text(docs_for_char[2]), dtype=np.int32)

        print("dataset: enwik8")
        print(f"split bytes/tokens | train={len(train_tokens):,} val={len(val_tokens):,} test={len(test_tokens):,}")
        print(f"tokenizer: {resolved_mode}")
        print(f"vocab size: {tokenizer.vocab_size}")
        return DatasetBundle(dataset_name, train_tokens, val_tokens, test_tokens, tokenizer, tokenizer.vocab_size)

    if dataset_name == "tinyshakespeare":
        train_text, val_text, test_text = _load_tinyshakespeare_text(data_dir)
        docs_all = [train_text, val_text, test_text]
        tokenizer = make_tokenizer(resolved_mode, dataset_name, docs_all=docs_all)

        train_tokens = np.array(tokenizer.encode_text(train_text), dtype=np.int32)
        val_tokens = np.array(tokenizer.encode_text(val_text), dtype=np.int32)
        test_tokens = np.array(tokenizer.encode_text(test_text), dtype=np.int32)

        print("dataset: tinyshakespeare")
        print(f"split chars/tokens | train={len(train_tokens):,} val={len(val_tokens):,} test={len(test_tokens):,}")
        print(f"tokenizer: {resolved_mode}")
        print(f"vocab size: {tokenizer.vocab_size}")
        return DatasetBundle(dataset_name, train_tokens, val_tokens, test_tokens, tokenizer, tokenizer.vocab_size)

    if dataset_name == "names":
        if not os.path.exists("input.txt"):
            names_url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
            urllib.request.urlretrieve(names_url, "input.txt")

        docs_all = [line.strip() for line in open("input.txt") if line.strip()]
        random.shuffle(docs_all)
        n = len(docs_all)
        n_train = int(0.9 * n)
        n_val = int(0.05 * n)
        docs_train = docs_all[:n_train]
        docs_val = docs_all[n_train:n_train + n_val]
        docs_test = docs_all[n_train + n_val:]

        tokenizer = make_tokenizer(resolved_mode, dataset_name, docs_all=docs_all)
        train_tokens, _ = _build_stream_and_doc_starts(docs_train, tokenizer)
        val_tokens, _ = _build_stream_and_doc_starts(docs_val, tokenizer)
        test_tokens, _ = _build_stream_and_doc_starts(docs_test, tokenizer)

        print("dataset: names")
        print(f"split docs | train={len(docs_train):,} val={len(docs_val):,} test={len(docs_test):,}")
        print(f"tokenizer: {resolved_mode}")
        print(f"vocab size: {tokenizer.vocab_size}")
        return DatasetBundle(dataset_name, train_tokens, val_tokens, test_tokens, tokenizer, tokenizer.vocab_size)

    raise ValueError("Unsupported DATASET. Use one of: enwik8, tinyshakespeare, names")


def make_random_window_dataset(tokens_np: np.ndarray, cfg: GPTConfig):
    block_len = cfg.seq_len + 1
    max_start = len(tokens_np) - block_len

    def gen():
        while True:
            s = np.random.randint(0, max_start + 1)
            seg = tokens_np[s:s + block_len]
            yield seg[:-1], seg[1:]

    sig = (
        tf.TensorSpec(shape=(cfg.seq_len,), dtype=tf.int32),
        tf.TensorSpec(shape=(cfg.seq_len,), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=sig)
    ds = ds.batch(cfg.batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def iter_eval_batches(tokens_np: np.ndarray, cfg: GPTConfig, max_tokens: int):
    block_len = cfg.seq_len + 1
    max_tokens = max(cfg.seq_len, int(max_tokens))
    starts = np.arange(0, len(tokens_np) - block_len + 1, cfg.seq_len, dtype=np.int64)
    if len(starts) == 0:
        raise ValueError("Eval split too short for current seq_len")
    max_seq = max(1, max_tokens // cfg.seq_len)
    starts = starts[:max_seq]

    for i in range(0, len(starts), cfg.batch_size):
        batch_starts = starts[i:i + cfg.batch_size]
        x = np.stack([tokens_np[s:s + cfg.seq_len] for s in batch_starts], axis=0)
        y = np.stack([tokens_np[s + 1:s + 1 + cfg.seq_len] for s in batch_starts], axis=0)
        yield x, y
