"""Launch a RunPod pod, run the experiment sweep over SSH, then terminate.

Requires these environment variables (put them in ~/.env or export them):
    RUNPOD_API_KEY      — RunPod API key (from runpod.io/console/user/settings)
    RUNPOD_VOLUME_ID    — Network volume ID (from runpod.io/console/storage)
    SSH_KEY_PATH        — Path to SSH private key (default: ~/.ssh/id_rsa)

Usage:
    python scripts/launch_and_run.py [--dry-run] [--no-terminate]

    --dry-run       Print the pod config and exit without launching
    --no-terminate  Leave the pod running after the sweep (for debugging)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Pod configuration — edit these if you need a different GPU or image
# ---------------------------------------------------------------------------
GPU_TYPE_ID = "NVIDIA GeForce RTX 4090"
CONTAINER_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
CONTAINER_DISK_GB = 20
VOLUME_MOUNT_PATH = "/workspace"
DATACENTER_ID = "EU-RO-1"
# ---------------------------------------------------------------------------

RUNPOD_GRAPHQL = "https://api.runpod.io/graphql"


def gql(api_key: str, query: str, variables: dict | None = None) -> dict:
    resp = requests.post(
        RUNPOD_GRAPHQL,
        json={"query": query, "variables": variables or {}},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL error: {data['errors']}")
    return data["data"]


def create_pod(api_key: str, volume_id: str) -> str:
    mutation = """
    mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
        podFindAndDeployOnDemand(input: $input) {
            id
            desiredStatus
            machineId
        }
    }
    """
    variables = {
        "input": {
            "gpuTypeId": GPU_TYPE_ID,
            "gpuCount": 1,
            "imageName": CONTAINER_IMAGE,
            "containerDiskInGb": CONTAINER_DISK_GB,
            "networkVolumeId": volume_id,
            "volumeMountPath": VOLUME_MOUNT_PATH,
            "startSsh": True,
            "dataCenterId": DATACENTER_ID,
            "ports": "22/tcp",
        }
    }
    data = gql(api_key, mutation, variables)
    pod_id = data["podFindAndDeployOnDemand"]["id"]
    print(f"Pod created: {pod_id}")
    return pod_id


def get_pod_ssh(api_key: str, pod_id: str) -> tuple[str, int] | None:
    """Return (host, port) when the pod is ready, else None."""
    query = """
    query GetPod($podId: String!) {
        pod(input: { podId: $podId }) {
            id
            desiredStatus
            runtime {
                ports {
                    ip
                    isIpPublic
                    privatePort
                    publicPort
                    type
                }
            }
        }
    }
    """
    data = gql(api_key, query, {"podId": pod_id})
    pod = data["pod"]
    if pod["desiredStatus"] != "RUNNING":
        return None
    runtime = pod.get("runtime")
    if not runtime:
        return None
    for port in runtime.get("ports", []):
        if port["privatePort"] == 22 and port["isIpPublic"]:
            return port["ip"], port["publicPort"]
    return None


def terminate_pod(api_key: str, pod_id: str):
    mutation = """
    mutation TerminatePod($podId: String!) {
        podTerminate(input: { podId: $podId })
    }
    """
    gql(api_key, mutation, {"podId": pod_id})
    print(f"Pod {pod_id} terminated.")


def wait_for_pod(api_key: str, pod_id: str, timeout: int = 300) -> tuple[str, int]:
    print("Waiting for pod to become ready", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        result = get_pod_ssh(api_key, pod_id)
        if result:
            print(" ready.")
            return result
        print(".", end="", flush=True)
        time.sleep(10)
    raise TimeoutError(f"Pod {pod_id} did not become ready within {timeout}s")


def ssh_run(host: str, port: int, key_path: str, command: str):
    """Run a command over SSH, streaming output live. Raises on non-zero exit."""
    cmd = [
        "ssh",
        "-i", key_path,
        "-p", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "LogLevel=ERROR",
        f"root@{host}",
        command,
    ]
    print(f"\n$ {command}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"SSH command failed with exit code {result.returncode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    parser.add_argument("--no-terminate", action="store_true", help="Leave pod running after sweep")
    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    volume_id = os.environ.get("RUNPOD_VOLUME_ID")
    key_path = os.environ.get("SSH_KEY_PATH", str(Path.home() / ".ssh" / "id_rsa"))

    if not api_key:
        sys.exit("ERROR: RUNPOD_API_KEY not set")
    if not volume_id:
        sys.exit("ERROR: RUNPOD_VOLUME_ID not set")
    if not Path(key_path).exists():
        sys.exit(f"ERROR: SSH key not found at {key_path} — set SSH_KEY_PATH")

    print(f"GPU:         {GPU_TYPE_ID}")
    print(f"Image:       {CONTAINER_IMAGE}")
    print(f"Volume:      {volume_id}  →  {VOLUME_MOUNT_PATH}")
    print(f"Datacenter:  {DATACENTER_ID}")
    print(f"SSH key:     {key_path}")

    if args.dry_run:
        print("\n--dry-run: exiting without launching.")
        return

    pod_id = create_pod(api_key, volume_id)

    try:
        host, port = wait_for_pod(api_key, pod_id)
        print(f"SSH: root@{host}:{port}")

        # Give the SSH daemon a moment to fully start
        time.sleep(5)

        ssh_run(host, port, key_path, "bash /workspace/unlearning-compression/scripts/session_start.sh")
        ssh_run(host, port, key_path, "bash /workspace/unlearning-compression/scripts/run_sweep.sh")

    finally:
        if args.no_terminate:
            print(f"\nPod {pod_id} left running (--no-terminate). SSH: root@{host}:{port}")
        else:
            print("\nTerminating pod...")
            terminate_pod(api_key, pod_id)


if __name__ == "__main__":
    main()
