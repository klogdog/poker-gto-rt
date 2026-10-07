"""Exercise the running API and save literal backend/strategy evidence."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("artifacts/api-smoke.json"))
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[1]
    with urlopen(args.base_url + "/health", timeout=10) as response:
        health = json.load(response)
    evidence = {"health": health, "source_sha256": {}, "runs": []}
    for path in sorted((base / "src").rglob("*.py")):
        evidence["source_sha256"][str(path.relative_to(base))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("heads-up-river", "multiplayer-river", "six-player-river"):
        payload = json.loads((base / "examples" / f"{name}.json").read_text())
        request = Request(args.base_url + "/v1/solve", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        start = perf_counter()
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
        assert result["backend"]["actual"] == payload["backend"]
        if payload["backend"] == "metal":
            assert result["backend"]["metal_execution"] is True
        wall = perf_counter() - start
        evidence["runs"].append({"example": name, "http_wall_seconds": wall, "response": result})
        print(f"{name}: {len(payload['players'])} players, {payload['backend']}, {wall:.3f}s")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
