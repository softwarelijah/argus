# Running the GPU pipeline on RunPod

A first-timer walkthrough: rent a GPU by the hour, run one script, copy the
results back. Budget about $3-6 and an evening (most of which is the GPU
training while you do something else).

The whole GPU job is automated by
[`scripts/cloud/runpod_bootstrap.sh`](../scripts/cloud/runpod_bootstrap.sh).
You mostly just start a machine and run it.

## 0. Before you start: put the code on GitHub

The pod needs to fetch your code. If you have not already:

```bash
# from your local argus/ folder
gh repo create argus --private --source=. --push
# or create an empty repo on github.com and:
#   git remote add origin https://github.com/<you>/argus.git && git push -u origin main
```

Note the clone URL. You will also want a GitHub Personal Access Token if the
repo is private (github.com -> Settings -> Developer settings -> Tokens).

## 1. Create a RunPod account and add credit

1. Sign up at runpod.io and add ~$10 of credit (Billing page).
2. You will not spend most of it; a full run is a few dollars.

## 2. Deploy a pod

1. Go to **Pods -> Deploy**.
2. **GPU:** pick **RTX 4090** (Community Cloud is cheapest, ~$0.40/hr). An
   A40 / L40 / A100 also work and finish faster.
3. **Template:** choose an official **RunPod PyTorch** template (it ships CUDA
   and PyTorch preinstalled, which saves setup).
4. **Disk:** set the container/volume disk to **at least 40 GB** (VisDrone plus
   weights need room; the default 10-20 GB is too small).
5. Click **Deploy On-Demand**. Wait ~1 minute for it to start.

## 3. Connect to the pod

On the pod card click **Connect**, then either:
- **Start Web Terminal** (easiest, runs in the browser), or
- **SSH** using the command shown (needs your SSH key added in RunPod settings).

You now have a shell on the GPU machine.

## 4. Get the code and run

```bash
cd /workspace
git clone https://github.com/<you>/argus.git
cd argus

# 1) smoke run first: ~20 min, proves the whole pipeline end to end
SMOKE=1 bash scripts/cloud/runpod_bootstrap.sh

# 2) if the smoke run finished and printed numbers, do the real run
bash scripts/cloud/runpod_bootstrap.sh
```

On a 16 GB GPU add `BATCH=8`. To shorten the real run, `EPOCHS=60` still gets
close to target accuracy.

The script installs deps, downloads VisDrone, trains, evaluates, exports the
INT8 engine, benchmarks, and prints a **RESULTS SUMMARY** with mAP and FPS.
Everything is also saved under `results/`.

## 5. Bring the results home

Easiest is to commit them back to GitHub from the pod:

```bash
git add results/ && git commit -m "results: GPU run on RTX 4090" && git push
# best.pt and the .engine are git-ignored (large); download those separately:
```

To download the weights/engine, use `runpodctl` (printed in the Connect panel)
or the file browser in the pod's Jupyter view. The engine is GPU-specific, so
keep it only if you deploy on the same GPU type.

## 6. STOP THE POD (this is the part people forget)

Billing runs until you stop. On the pod card:
- **Stop** pauses it (you still pay a little for disk), or
- **Terminate** deletes it completely (no more charges).

Terminate once you have your `results/` and `best.pt`.

## 7. Update the README locally

Back on your machine, pull the results and fill in the real numbers in the
README tables (the detector mAP table and the latency/FPS table), then commit.
That converts the last aspirational numbers in the project into measured ones.

## Troubleshooting

- **`no GPU visible`**: you deployed a CPU pod. Terminate and redeploy a GPU one.
- **CUDA out of memory**: lower `BATCH` (try 8, then 4) or `IMGSZ` (try 960).
- **`tensorrt` install fails**: the script falls back to Ultralytics' native
  engine export automatically; you still get an engine and numbers.
- **Disk full**: redeploy with a larger volume (40 GB+).
- **Download is slow**: VisDrone is ~2 GB; it is a one-time cost per pod.
