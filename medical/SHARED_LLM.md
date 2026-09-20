# Shared medical LLM server (pod port 8001). Do not start your own.

One vLLM server serves every medical agent. Workers call it at
`http://127.0.0.1:8001/v1` (the pipeline's default `MED_LLM_URL`). Do not restart
it, do not start a second vLLM, and do not run `medical/start_llm.sh` or
`pod_bench.sh llm`/`e2e`: both call `start_llm.sh`, which kills whatever is on :8001. A cold
start takes 8 to 15 minutes (weights come over FUSE and torch.compile is slow), and
everyone else is blocked while it runs.

## Model

`Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` (MoE, 3B active). It was chosen on 2026-09-18 by medical-bringup-1.

| model (think=0, bias 1.2, turbo transcripts, 31 convs) | dev | test | all | acc | tIoU | QA s/conv |
|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 (chosen) | 0.756 | 0.786 | 0.770 | 0.990 | 0.623 | 1.4 mean / 2.6 max |
| cyankiwi/Qwen3.8-27B-AWQ-INT4 | 0.744 | 0.795 | 0.768 | 0.997 | 0.616 | 6.6 mean / 10.2 max |

The scores are tied within noise, and the MoE is 4.7 times faster. That leaves time
inside the 60 s budget for thinking, multi-pass prompts or self-consistency.

## Exact command (as started)

```bash
# on the pod
MED_SERVED_NAME=Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
  bash /workspace/medical-bringup-1/medical/start_llm.sh /root/qwen35-a3b-gptq 0.6
```

This expands to the following (see `start_llm.sh`; the env has PATH=/workspace/venv-vllm/bin:$PATH,
OMP_NUM_THREADS=8, VLLM_USE_FLASHINFER_SAMPLER=0, HF_HUB_OFFLINE=1, HF_HOME=/workspace/hf):

```bash
nohup /workspace/venv-vllm/bin/vllm serve /root/qwen35-a3b-gptq --port 8001 --host 127.0.0.1 \
  --max-model-len 8192 --gpu-memory-utilization 0.6 --max-num-seqs 16 \
  --enable-prefix-caching --limit-mm-per-prompt '{"image":0,"video":0}' \
  --served-model-name Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 > /workspace/logs/vllm_med.log 2>&1 &
```

- The weights are on the container disk at `/root/qwen35-a3b-gptq`, not on /workspace, because
  the /workspace volume quota is 60 GB and was full. They are lost if the pod restarts. To
  re-download them (about 4 min, ~25 GB):
  `HF_HUB_DISABLE_XET=1 /workspace/venv-vllm/bin/python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3.5-35B-A3B-GPTQ-Int4', local_dir='/root/qwen35-a3b-gptq', max_workers=4)"`
  (xet downloads fail with "Background writer channel closed").
- The fallback model `cyankiwi/Qwen3.8-27B-AWQ-INT4` is still complete in /workspace/hf.
- Check that it is up with `curl -s localhost:8001/v1/models`.

## GPU budget (A40, 48 GB, shared)

vLLM takes about 28.9 GB. The medical /predict server (whisper turbo fp16) takes about 2.4 GB.
**Drone jobs must stay at or below ~15 GB.** When the GPU was full, whisper hit CUDA OOM,
which gave all-yes/no-span answers (score 0.32). `pipeline.transcribe` now reloads the model
and retries on a CUDA error, then falls back to CPU small.en int8 (10.5 s for 100 s of audio).

## Pod pitfalls found during bring-up

- The pid/thread cap is 5120 for the whole pod. Orphaned `multiprocessing` workers from timed-out
  survival evaluator runs (127 threads each) used all of it, and vLLM then died with
  "Resource temporarily unavailable (src/thread.cpp:241)". Set OMP_NUM_THREADS=1 in
  multiprocessing jobs and kill orphans with `ps -eo pid,ppid,args | awk '$2==1 && /spawn_main/'`.
- vLLM needs `ninja` on PATH, which `start_llm.sh` now handles.
- /workspace has a 60 GB quota. Watch it with `du -sh /workspace`; writes fail with "Disk quota exceeded".

## Serving /predict

`bash /workspace/medical-bringup-1/medical/start_server.sh` starts the server on :9054 from the
bring-up checkout, and `pkill`s any other `medical/server.py`. Workers should run their own copy
on a different port: `MED_PORT=90xx $PY medical/server.py`. Leave 9054 alone, because it is the
URL registered with the portal. `MED_SAVE_SEEN` is now off by default, because portal requests
can contain hidden-holdout conversations (verify sent conversation_sample_3, which is not in
the worker set).
