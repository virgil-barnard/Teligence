import os
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int

    # shape
    n_layer: int = int(os.environ.get("N_LAYER", "6"))
    n_embd: int = int(os.environ.get("N_EMBD", "384"))
    n_head: int = int(os.environ.get("N_HEAD", "6"))
    n_kv_head: int = int(os.environ.get("N_KV_HEAD", "2"))
    mlp_mult: int = int(os.environ.get("MLP_MULT", "4"))

    # training length (T) and local window (W)
    seq_len: int = int(os.environ.get("SEQ_LEN", "512"))
    attn_window: int = int(os.environ.get("ATTN_WINDOW", "256"))

    # positional encoding
    pos_encoding: str = "rope"

    # RoPE
    rope_base: float = 10000.0
    rope_scale_factor: float = 1.0

    # norms + MLP
    rmsnorm_eps: float = 1e-5
    rmsnorm_scale: bool = True
    use_swiglu: bool = True

    # fused qkv
    fuse_qkv: bool = True

    # regularization
    dropout: float = float(os.environ.get("DROPOUT", "0.1"))

    # output
    tie_embeddings: bool = True

    # compute
    use_bf16: bool = True

    # flash attention
    use_flash_attn: bool = True
    flash_q_block: int = 128
    flash_k_block: int = 128

    # while_loop tuning
    flash_parallel_iterations: int = 1
    flash_swap_memory: bool = False

    # activation checkpointing for flash
    flash_recompute_grad: bool = True

    # graph compilation
    jit_compile: bool = False

    # optimization
    base_lr: float = float(os.environ.get("BASE_LR", "3e-4"))
    min_lr: float = float(os.environ.get("MIN_LR", "3e-5"))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", "200"))
    weight_decay: float = float(os.environ.get("WEIGHT_DECAY", "0.1"))
    grad_clip_norm: float = 1.0

    # gradient accumulation
    grad_accum_steps: int = 1

    # run
    batch_size: int = int(os.environ.get("BATCH_SIZE", "32"))
    num_updates: int = int(os.environ.get("NUM_UPDATES", "3000"))
    log_every: int = int(os.environ.get("LOG_EVERY", "50"))
    eval_every: int = int(os.environ.get("EVAL_EVERY", "50"))
    eval_tokens: int = int(os.environ.get("EVAL_TOKENS", "262144"))
    preview_prompt: str = os.environ.get("PREVIEW_PROMPT", "The ")
    preview_max_new_tokens: int = int(os.environ.get("PREVIEW_MAX_NEW_TOKENS", "64"))
    save_every: int = int(os.environ.get("SAVE_EVERY", "500"))
    tokenizer: str = os.environ.get("TOKENIZER", "auto")

    # live visualization (TensorBoard)
    visualize: bool = os.environ.get("VISUALIZE", "0") == "1"
    tb_logdir: str = os.environ.get("TB_LOGDIR", "")
    tb_hist_every: int = int(os.environ.get("TB_HIST_EVERY", "200"))
    attn_viz_every: int = int(os.environ.get("ATTN_VIZ_EVERY", "50"))
    attn_viz_max_new_tokens: int = int(os.environ.get("ATTN_VIZ_MAX_NEW_TOKENS", "24"))
    attn_viz_max_heads: int = int(os.environ.get("ATTN_VIZ_MAX_HEADS", "4"))
    attn_viz_max_layers: int = int(os.environ.get("ATTN_VIZ_MAX_LAYERS", "3"))

    # experiment tracking
    run_name: str = os.environ.get("RUN_NAME", "")
    runs_dir: str = os.environ.get("RUNS_DIR", "./runs")
    keep_last_ckpts: int = int(os.environ.get("KEEP_LAST_CKPTS", "3"))
    keep_best_ckpts: int = int(os.environ.get("KEEP_BEST_CKPTS", "1"))

    # resume control
    resume_from: str = os.environ.get("RESUME_FROM", "")
    extra_updates: int = int(os.environ.get("EXTRA_UPDATES", "0"))


def validate_config(cfg: GPTConfig):
    assert cfg.n_embd % cfg.n_head == 0
    assert cfg.n_head % cfg.n_kv_head == 0
    assert cfg.attn_window <= cfg.seq_len
    assert cfg.pos_encoding in ("rope", "alibi")

    head_dim = cfg.n_embd // cfg.n_head
    assert head_dim % 2 == 0, "RoPE requires even head_dim (n_embd//n_head)."

    if cfg.use_flash_attn:
        assert cfg.seq_len % cfg.flash_q_block == 0, "seq_len must be divisible by flash_q_block"
        assert cfg.seq_len % cfg.flash_k_block == 0, "seq_len must be divisible by flash_k_block"
