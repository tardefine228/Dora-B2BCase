"""Small async smoke/load test for the classification API.

Example: python scripts/load_test.py --base-url http://127.0.0.1:8000 --pid 12345
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx


def process_rss_mb(pid: int) -> float | None:
    """Read Linux process RSS if /proc is available; return None otherwise."""
    status_file = Path(f"/proc/{pid}/status")
    try:
        for line in status_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    except (OSError, ValueError, IndexError):
        return None
    return None


async def run(args: argparse.Namespace) -> None:
    timeout = httpx.Timeout(args.request_timeout)
    latencies: list[float] = []
    maximum_rss: float | None = None
    stop_sampler = asyncio.Event()

    async def sample_rss() -> None:
        nonlocal maximum_rss
        if args.pid is None:
            return
        while not stop_sampler.is_set():
            rss = process_rss_mb(args.pid)
            if rss is not None:
                maximum_rss = max(maximum_rss or 0, rss)
            try:
                await asyncio.wait_for(stop_sampler.wait(), timeout=0.25)
            except TimeoutError:
                pass

    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/"), timeout=timeout) as client:
        sampler = asyncio.create_task(sample_rss())
        started = time.perf_counter()
        try:

            async def submit_batch(batch_number: int) -> dict:
                payload = {
                    "items": [
                        f"Нагрузочный тест {batch_number}: товар {item_number}"
                        for item_number in range(args.items)
                    ]
                }
                request_started = time.perf_counter()
                response = await client.post("/classify/batch", json=payload)
                latencies.append((time.perf_counter() - request_started) * 1000)
                response.raise_for_status()
                return response.json()

            accepted = await asyncio.gather(
                *(submit_batch(batch_number) for batch_number in range(args.concurrency))
            )
            jobs = {job["job_id"]: job["status_url"] for job in accepted}
            deadline = time.perf_counter() + args.completion_timeout
            final_states: dict[str, dict] = {}

            while len(final_states) < len(jobs):
                if time.perf_counter() >= deadline:
                    raise TimeoutError("Timed out waiting for batch jobs to finish")

                async def poll_job(job_id: str, status_url: str) -> tuple[str, dict]:
                    response = await client.get(status_url)
                    response.raise_for_status()
                    return job_id, response.json()

                states = await asyncio.gather(
                    *(poll_job(job_id, status_url) for job_id, status_url in jobs.items())
                )
                for job_id, state in states:
                    if state["status"] in {"completed", "failed"}:
                        final_states[job_id] = state
                if len(final_states) < len(jobs):
                    await asyncio.sleep(args.poll_interval)

            failed = [state for state in final_states.values() if state["status"] != "completed"]
            elapsed = time.perf_counter() - started
            summary = {
                "jobs": len(jobs),
                "items_per_job": args.items,
                "total_items": len(jobs) * args.items,
                "completed_items": sum(state["completed"] for state in final_states.values()),
                "failed_jobs": len(failed),
                "request_latency_ms": {
                    "min": round(min(latencies), 2),
                    "average": round(sum(latencies) / len(latencies), 2),
                    "max": round(max(latencies), 2),
                },
                "total_elapsed_seconds": round(elapsed, 2),
                "api_metrics": (await client.get("/metrics")).json(),
                "max_server_rss_mb": maximum_rss,
            }
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            if failed:
                raise RuntimeError(f"{len(failed)} batch job(s) failed")
        finally:
            stop_sampler.set()
            await sampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=10, help="number of parallel batches")
    parser.add_argument("--items", type=int, default=100, help="items in each batch (max 100)")
    parser.add_argument("--pid", type=int, help="optional API process PID for Linux RSS sampling")
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--completion-timeout", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.concurrency < 1 or not 1 <= args.items <= 100:
        parser.error("concurrency must be >= 1 and items must be between 1 and 100")
    return args


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
