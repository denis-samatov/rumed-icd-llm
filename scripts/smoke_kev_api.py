"""Check choice/noul/score through the real TypeSafe SDK against a local Kev server.

This synthetic contract probe is separate from the RuMed quality benchmark.
Run in the independent Kev environment, not the CPU verification environment.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8008")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    parsed = urlparse(args.base_url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        ap.error("this probe only sends synthetic inputs to a loopback HTTP server")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite API evidence")
    import msgspec
    from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

    root = Path(__file__).resolve().parents[1]
    labels = json.loads((root / "data/icd_labels_ru.json").read_text(encoding="utf-8"))["labels"]
    criteria = {r["code"]: r["description"] for r in labels}
    with urllib.request.urlopen(args.base_url + "/v1/models", timeout=30) as response:
        model = json.loads(response.read())["models"][0]
    with TypeSafeClient(api_key="local", base_url=args.base_url, model="kev-latest") as client:
        response = client.system_one(
            state="I was charged twice for order 1182. Please refund one of the charges.",
            questions={
                "team": Choice(instructions="Which team should handle this?", criteria={
                    "billing": "Charges and refunds", "shipping": "Deliveries",
                    "returns": "Product exchanges"}),
                "duplicate": Noul(instructions="Was the customer charged twice?"),
                "urgency": Score(instructions="How urgent is this request?", criteria=[
                    "General question, no issue", "A billing problem needs a reply",
                    "Immediate physical danger"]),
            },
        )
        wide = client.system_one(
            state="Synthetic API check: headache, fatigue and high blood pressure.",
            questions={"icd": Choice(instructions="Which ICD-10 category fits these complaints?",
                                     criteria=criteria)},
        ).choices["icd"]
    answers = msgspec.to_builtins(response.answers)
    assert set(answers) == {"team", "duplicate", "urgency"}
    assert answers["team"]["choice"] in {"billing", "shipping", "returns"}
    assert 0 <= answers["duplicate"]["noul"] <= 1
    assert 0 <= answers["urgency"]["score"] <= 2
    for key in ("team", "urgency"):
        values = list(answers[key]["probabilities"].values())
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in values)
        assert abs(sum(values) - 1) < .02
    assert set(wide.probabilities) == set(criteria) and len(wide.probabilities) == 105
    assert wide.choice in criteria
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in wide.probabilities.values())
    assert abs(sum(wide.probabilities.values()) - 1) < .02
    result = {"scope": "synthetic local API contract; not a quality or load benchmark",
              "sdk": "typesafe-sdk", "sdk_version": importlib.metadata.version("typesafe-sdk"),
              "base_url": args.base_url,
              "backend": model["backend"], "dtype": model["dtype"],
              "temperature": model["temperature"], "answers": answers,
              "wide_choice_options": len(wide.probabilities),
              "checks": "SDK parsed three primitives and 105-option choice; "
                        "normalized probabilities"}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
