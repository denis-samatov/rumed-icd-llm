import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("changed", [False, True])
def test_download_preserves_manifest_and_rejects_drift(tmp_path, changed):
    root = tmp_path / "project"
    raw = root / "data/raw"
    scripts = root / "scripts"
    raw.mkdir(parents=True)
    scripts.mkdir()
    source = Path(__file__).resolve().parents[1] / "scripts/download_data.sh"
    shutil.copy2(source, scripts / source.name)
    content = b'{"idx":"a","symptoms":"example","code":"J06"}\n'
    listing = ""
    for split in ("train", "dev", "test"):
        name = f"{split}_v1.jsonl"
        (raw / name).write_bytes(content)
        listing += f"{hashlib.sha256(content).hexdigest()}  ./{name}\n"
    (raw / "SHA256SUMS").write_text(listing)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    # Emulate curl without any network access; deliberately drift only the test file.
    curl = fake_bin / "curl"
    curl.write_text('#!/usr/bin/env bash\nset -eu\n'
                    'cp "$FAKE_RAW/$(basename "$2")" "$4"\n'
                    'if [ "$DRIFT" = 1 ] && [[ "$2" = */test_v1.jsonl ]]; then\n'
                    '  echo drift >> "$4"\nfi\n')
    curl.chmod(0o755)
    env = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
           "FAKE_RAW": str(raw), "DRIFT": str(int(changed))}
    result = subprocess.run(["bash", str(scripts / source.name)], env=env,
                            capture_output=True, text=True)
    assert (result.returncode != 0) is changed, result.stdout + result.stderr
    assert (raw / "SHA256SUMS").read_text() == listing
    assert all((raw / f"{s}_v1.jsonl").read_bytes() == content for s in ("train", "dev", "test"))
    assert not list(raw.glob(".download.*"))
