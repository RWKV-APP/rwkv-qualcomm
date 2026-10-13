#!/usr/bin/env python3
"""Validate RWKV mobile RMPack container metadata without loading model payloads."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


MAGIC = b"RWKVMBLE"
ALIGNMENT = 4096


def read_exact(handle, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"unexpected EOF: wanted {size} bytes, got {len(data)}")
    return data


def read_u32(handle) -> int:
    return struct.unpack("<I", read_exact(handle, 4))[0]


def read_u64(handle) -> int:
    return struct.unpack("<Q", read_exact(handle, 8))[0]


def validate(path: Path) -> dict:
    total_size = path.stat().st_size
    with path.open("rb") as handle:
        if read_exact(handle, len(MAGIC)) != MAGIC:
            raise ValueError("bad magic header")

        config_size = read_u32(handle)
        if config_size <= 0 or config_size > 1_048_576:
            raise ValueError(f"invalid config length: {config_size}")
        config = json.loads(read_exact(handle, config_size).decode("utf-8"))

        file_count = read_u32(handle)
        entries = []
        for _ in range(file_count):
            name_size = read_u32(handle)
            if name_size <= 0 or name_size > 65_536:
                raise ValueError(f"invalid entry name length: {name_size}")
            name = read_exact(handle, name_size).decode("utf-8")
            size = read_u64(handle)
            offset = read_u64(handle)
            entries.append({"name": name, "size": size, "offset": offset})

    expected_names = [f"model_{index}" for index in range(file_count)]
    actual_names = [entry["name"] for entry in entries]
    if actual_names != expected_names:
        raise ValueError(f"unexpected entry names: {actual_names}")
    if config.get("backend") != "qnn":
        raise ValueError(f"unexpected backend: {config.get('backend')!r}")
    if config.get("n_chunks") != file_count or file_count != 4:
        raise ValueError(
            f"chunk mismatch: config={config.get('n_chunks')}, entries={file_count}"
        )

    expected = (
        {"hidden_size": 2048, "quant_type": "a16w8"}
        if "1.5b" in path.name.lower()
        else {"hidden_size": 2560, "quant_type": "a16w8" if "a16w8" in path.name.lower() else "a16w4"}
    )
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"unexpected {key}: {config.get(key)!r}, expected {value!r}")

    previous_end = 0
    for entry in entries:
        offset = entry["offset"]
        end = offset + entry["size"]
        if offset % ALIGNMENT:
            raise ValueError(f"unaligned entry offset: {offset}")
        if offset < previous_end:
            raise ValueError(f"overlapping entry at offset {offset}")
        if end > total_size:
            raise ValueError(f"entry exceeds container: end={end}, size={total_size}")
        previous_end = end
    if entries and previous_end != total_size:
        raise ValueError(f"unexpected trailing data: payload_end={previous_end}, size={total_size}")

    return {
        "path": str(path),
        "bytes": total_size,
        "target_platform": config.get("target_platform"),
        "hidden_size": config.get("hidden_size"),
        "quant_type": config.get("quant_type"),
        "chunks": file_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    files: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.rmpack")))
        else:
            files.append(path)

    results = []
    failures = []
    for path in files:
        try:
            results.append(validate(path))
        except Exception as error:  # keep validating the rest of the cohort
            failures.append({"path": str(path), "error": str(error)})

    print(json.dumps({"valid": results, "failures": failures}, indent=2))
    return 1 if failures or not results else 0


if __name__ == "__main__":
    raise SystemExit(main())
