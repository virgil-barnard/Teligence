import unittest

import numpy as np
import tensorflow as tf

from teligence.config import GPTConfig, validate_config
from teligence.data_utils import iter_eval_batches, make_random_window_dataset
from teligence.modeling import ExplicitGPT, set_precision
from teligence.tokenizer import ByteTokenizer, CharTokenizer
from teligence.train_utils import build_train_micro_step, build_train_state, evaluate_model, lr_schedule


class SmokeTests(unittest.TestCase):
    def _tiny_cfg(self):
        return GPTConfig(
            vocab_size=32,
            n_layer=1,
            n_embd=32,
            n_head=4,
            n_kv_head=2,
            mlp_mult=2,
            seq_len=16,
            attn_window=8,
            batch_size=2,
            num_updates=2,
            log_every=1,
            eval_every=1,
            eval_tokens=64,
            save_every=0,
            use_bf16=False,
            use_flash_attn=True,
            flash_q_block=8,
            flash_k_block=8,
            dropout=0.0,
            runs_dir="./runs_test",
            run_name="",
            keep_last_ckpts=1,
            keep_best_ckpts=1,
        )

    def test_model_forward_shape(self):
        cfg = self._tiny_cfg()
        validate_config(cfg)
        set_precision(cfg.use_bf16)
        model = ExplicitGPT(cfg)

        x = tf.zeros([cfg.batch_size, cfg.seq_len], dtype=tf.int32)
        logits = model(x, training=False)
        self.assertEqual(tuple(logits.shape), (cfg.batch_size, cfg.seq_len, cfg.vocab_size))

    def test_data_iterators(self):
        cfg = self._tiny_cfg()
        tokens = np.arange(512, dtype=np.int32) % cfg.vocab_size

        ds = make_random_window_dataset(tokens, cfg)
        x, y = next(iter(ds))
        self.assertEqual(tuple(x.shape), (cfg.batch_size, cfg.seq_len))
        self.assertEqual(tuple(y.shape), (cfg.batch_size, cfg.seq_len))

        batches = list(iter_eval_batches(tokens, cfg, max_tokens=64))
        self.assertGreater(len(batches), 0)
        bx, by = batches[0]
        self.assertEqual(bx.shape[1], cfg.seq_len)
        self.assertEqual(by.shape[1], cfg.seq_len)

    def test_one_train_step_and_eval(self):
        cfg = self._tiny_cfg()
        validate_config(cfg)
        set_precision(cfg.use_bf16)

        model = ExplicitGPT(cfg)
        state = build_train_state(cfg, model, run_dir="./runs_test/smoke")
        train_micro_step = build_train_micro_step(cfg, model, state)

        x = tf.zeros([cfg.batch_size, cfg.seq_len], dtype=tf.int32)
        y = tf.zeros([cfg.batch_size, cfg.seq_len], dtype=tf.int32)
        loss, gnorm, lr, updated = train_micro_step(x, y)

        self.assertTrue(np.isfinite(float(loss.numpy())))
        self.assertTrue(np.isfinite(float(gnorm.numpy())))
        self.assertTrue(np.isfinite(float(lr.numpy())))
        self.assertGreater(float(lr.numpy()), 0.0)
        self.assertTrue(bool(updated.numpy()))

        tokens = np.arange(512, dtype=np.int32) % cfg.vocab_size
        nll, bpc, ppl = evaluate_model(model, tokens, cfg, max_tokens=64)
        self.assertTrue(np.isfinite(nll))
        self.assertTrue(np.isfinite(bpc))
        self.assertTrue(np.isfinite(ppl))

    def test_tokenizer_roundtrip(self):
        char_tok = CharTokenizer.from_texts(["alice", "bob"])
        ids = char_tok.encode_text("alice")
        self.assertGreater(len(ids), 0)
        self.assertEqual(char_tok.decode_ids(ids), "alice")

        byte_tok = ByteTokenizer()
        b_ids = byte_tok.encode_text("hello")
        self.assertEqual(byte_tok.decode_ids(b_ids), "hello")

    def test_lr_schedule_caps_warmup_for_short_runs(self):
        cfg = self._tiny_cfg()
        cfg.num_updates = 20
        cfg.warmup_steps = 200

        lr0 = float(lr_schedule(cfg, tf.constant(0, dtype=tf.int64)).numpy())
        lr10 = float(lr_schedule(cfg, tf.constant(10, dtype=tf.int64)).numpy())
        self.assertGreater(lr0, 0.0)
        self.assertLess(lr0, cfg.base_lr)
        self.assertLessEqual(lr10, cfg.base_lr)

if __name__ == "__main__":
    unittest.main()
