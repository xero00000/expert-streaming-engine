CREATE TABLE benchmark_submissions (
  event_id TEXT PRIMARY KEY,
  installation_hash TEXT NOT NULL,
  recorded_at_epoch INTEGER NOT NULL,
  received_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  ese_version TEXT NOT NULL,
  platform TEXT NOT NULL,
  cpu_model TEXT,
  logical_cpus INTEGER NOT NULL,
  ram_gib INTEGER,
  gpus_json TEXT NOT NULL,
  model_signature TEXT NOT NULL,
  model_size_bytes INTEGER,
  architecture TEXT,
  quantization TEXT,
  preset TEXT NOT NULL,
  objective TEXT NOT NULL,
  safe_margin REAL NOT NULL,
  verified_max_context INTEGER NOT NULL,
  promoted_context INTEGER NOT NULL,
  best_kv_type TEXT NOT NULL,
  best_batch_size INTEGER NOT NULL,
  best_tokens_per_second REAL NOT NULL,
  private_payload_json TEXT NOT NULL
);

CREATE INDEX benchmark_received_at ON benchmark_submissions(received_at);
CREATE INDEX benchmark_grouping ON benchmark_submissions(
  architecture, quantization, gpus_json, promoted_context, best_kv_type, best_batch_size
);
