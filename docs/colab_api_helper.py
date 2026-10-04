"""Reusable helper for working with GitHub from Colab.

Copy this cell into any Colab notebook. It provides:
    put_file(path, content, message) -> commit SHA
    trigger_workflow(workflow_file, inputs) -> run id
    wait_for_run(commit_sha, workflow_name) -> final run dict
    list_artifacts(run_id) -> list of artifact dicts
    download_artifact(artifact_id) -> bytes
    fetch_log(run_id, name_filter) -> decoded log text

No local filesystem writes. No git commands. Everything through the API.
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.request
import zipfile

REPO = "Sharqascc/nnds"
BRANCH = "feature/pipeline-to-metric-schemas"


def _api(token: str, method: str, path: str, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        data=data, method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "colab",
        },
    )
    try:
        with urllib.request.urlopen(req) as r:
            return json.load(r) if r.status != 204 else {}
    except urllib.error.HTTPError as e:
        return {"_http_error": e.code, "_body": e.read().decode()[:400]}


def put_file(token: str, path: str, content: str, message: str, branch: str = BRANCH) -> dict:
    existing = _api(token, "GET", f"/repos/{REPO}/contents/{path}?ref={branch}")
    body = {
        "message": message,
        "content": base64.b64encode(content.encode()).decode(),
        "branch": branch,
    }
    if isinstance(existing, dict) and "sha" in existing:
        body["sha"] = existing["sha"]
    return _api(token, "PUT", f"/repos/{REPO}/contents/{path}", body)


def trigger_workflow(token: str, workflow_file: str, inputs: dict, branch: str = BRANCH) -> dict:
    return _api(
        token, "POST",
        f"/repos/{REPO}/actions/workflows/{workflow_file}/dispatches",
        {"ref": branch, "inputs": inputs},
    )


def wait_for_run(token: str, commit_sha: str, workflow_name: str, timeout_s: int = 900) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        runs = _api(token, "GET", f"/repos/{REPO}/actions/runs?branch={BRANCH}&per_page=20")
        for r in runs.get("workflow_runs", []):
            if r["head_sha"] == commit_sha and r["name"] == workflow_name:
                if r["status"] == "completed":
                    return r
        time.sleep(15)
    return {"_timeout": True}


def list_artifacts(token: str, run_id: int) -> list:
    r = _api(token, "GET", f"/repos/{REPO}/actions/runs/{run_id}/artifacts")
    return r.get("artifacts", [])


def download_artifact(token: str, artifact_id: int) -> bytes:
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/actions/artifacts/{artifact_id}/zip",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "colab",
        },
    )
    with urllib.request.urlopen(req) as r:
        return r.read()


def fetch_log(token: str, run_id: int, name_filter: str = "") -> str:
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/actions/runs/{run_id}/logs",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "colab",
        },
    )
    with urllib.request.urlopen(req) as r:
        blob = r.read()
    zf = zipfile.ZipFile(io.BytesIO(blob))
    parts = []
    for name in zf.namelist():
        if name_filter and name_filter not in name:
            continue
        parts.append(f"===== {name} =====")
        parts.append(zf.read(name).decode("utf-8", errors="replace"))
    return "\n".join(parts)
