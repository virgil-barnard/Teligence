import json
import os
import time


class RunTracker:
    def __init__(self, dataset_name: str, run_name: str, runs_dir: str):
        run_stamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_id = run_name.strip() if run_name.strip() else f"{dataset_name}_{run_stamp}"
        self.dataset_name = dataset_name
        self.run_dir = os.path.join(runs_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.metrics_jsonl = os.path.join(self.run_dir, "metrics.jsonl")
        self.summary_json = os.path.join(self.run_dir, "summary.json")

    def log_event(self, event):
        rec = dict(event)
        rec["dataset"] = self.dataset_name
        rec["run_id"] = self.run_id
        rec["time"] = time.time()
        with open(self.metrics_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    def write_summary(self, summary):
        with open(self.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
