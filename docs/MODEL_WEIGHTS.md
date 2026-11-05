# Model Weights Policy

The NNDS pipeline uses two pre-trained models:

| Model | Size | Source |
|---|---|---|
| UVH-26 (vehicle detector) | ~80 MB | Hugging Face: iisc-aim/UVH-26 |
| YOLO11n (person / COCO fallback) | ~5 MB | GitHub: ultralytics/assets |

## They are not stored in this repository

Neither model is committed. This is deliberate:

1. Repo size. 85 MB of binary in git would grow the packfile on every clone
   and slow down CI checkout for every PR, for artifacts that never change.
2. Licensing. UVH-26 has its own terms. Bundling it would pull those terms
   into NNDS's LICENSE.
3. Security. A torch.load on a weight file is arbitrary code execution (see
   src/utils/checkpoint.py). We want weights to come from a known URL the
   downloader can verify, not from an opaque commit.

## How to get them

    bash scripts/download_models.sh

That fetches data/models/uvh26.pt and data/models/yolo11n.pt from their
public URLs. It is idempotent and safe to re-run.

## What is allowed

- Small fixtures (< 1 MiB) that tests need, added to ALLOWLIST in
  tests/test_no_model_weights_committed.py with a comment explaining why.
- Pointers produced by Git LFS. If a weight file is genuinely
  version-controlled in the future, route it through LFS in .gitattributes.
  The extensions .pt .pth .onnx .engine .safetensors .ckpt .gguf are already
  routed. The git blob is then ~130 bytes, not the full model.

## What is not allowed

- Any weight file committed as a raw binary (blob size > 1 MiB). CI fails
  with test_no_model_weights_committed and a message pointing here.
- Extensions: .pt .pth .onnx .engine .safetensors .ckpt .gguf .msgpack
  .tflite .mlmodel .h5 .bin .pkl, if larger than 1 MiB in the tree.

## How the guard works

tests/test_no_model_weights_committed.py runs "git ls-tree -r -l HEAD" and
checks the stored blob size of every file whose extension matches the weight
list. LFS pointers show up as ~130 bytes and pass; direct commits show up as
many megabytes and fail.

The test runs in the fast CI lane, takes ~10 ms, and has no dependencies
beyond the standard library.

## What if CI fails

The failure message names the offending files. Do one of:

1. If it is a local artifact:

       git rm --cached data/models/uvh26.pt

2. If it is needed for tests, shrink it below 1 MiB or download it at test
   time from a public URL.

3. If it genuinely belongs in the repo, route it through LFS and add it to
   ALLOWLIST in tests/test_no_model_weights_committed.py.
