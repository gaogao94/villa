#!/usr/bin/env python3
"""Check a published OME-Zarr store for the metadata failures a reader cannot see.

Every check here comes from a defect found in the published catalog: stores whose name states a
physical pitch while the OME metadata carries no unit, a store that declares six levels and holds no
chunks, unitless prediction stores, headers that would mislead a consumer about shape or dtype. A
reader that trusts the metadata gets a wrong number or blank data with no error, so the time to
notice is before publishing.

    python lasagna/scripts/check_omezarr_metadata.py --root samples/PHerc0814/volumes/xxx.zarr
    python lasagna/scripts/check_omezarr_metadata.py --root ./local/store.zarr --name-states-pitch 2.4
    python lasagna/scripts/check_omezarr_metadata.py --root <url> --json

Reads metadata only; no chunk bytes are downloaded. Exits 1 if any ERROR was reported, 0 otherwise.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import urllib.parse
import urllib.request

DEFAULT_BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
USER_AGENT = {"User-Agent": "vesuvius-metadata-check/1.0"}

ERROR = "ERROR"
WARN = "WARN"


class Finding:
    __slots__ = ("level", "code", "message")

    def __init__(self, level: str, code: str, message: str):
        self.level = level
        self.code = code
        self.message = message

    def as_dict(self) -> dict:
        return {"level": self.level, "code": self.code, "message": self.message}

    def __str__(self) -> str:
        return f"[{self.level}] {self.code}: {self.message}"

    __repr__ = __str__


def _is_all_one(scale) -> bool:
    return bool(scale) and all(abs(float(v) - 1.0) < 1e-9 for v in scale)


def check_multiscales(ms: dict, pitch_in_name: float | None) -> list[Finding]:
    """Checks that need only the multiscales block."""
    out: list[Finding] = []
    if not isinstance(ms, dict):
        return [Finding(ERROR, "NO_MULTISCALES", "no multiscales block, so nothing declares the pyramid")]
    axes = ms.get("axes") or []
    units = [a.get("unit") for a in axes if isinstance(a, dict)]
    datasets = ms.get("datasets") or []
    if not datasets:
        out.append(Finding(ERROR, "NO_DATASETS", "multiscales declares no datasets"))
    if not any(units):
        if pitch_in_name is not None:
            out.append(Finding(
                ERROR, "UNITS_MISSING_PITCH_IN_NAME",
                f"name states a {pitch_in_name} um pitch but no axis carries a unit, so a reader "
                f"measures in voxels and believes they are micrometres",
            ))
        else:
            out.append(Finding(
                WARN, "UNITS_MISSING",
                "no axis carries a unit; acceptable only if physical size is stated elsewhere",
            ))
    else:
        missing = [a.get("name") for a in axes if isinstance(a, dict) and not a.get("unit")]
        if missing and pitch_in_name is not None:
            out.append(Finding(
                WARN, "UNITS_PARTIAL",
                f"axes {missing} carry no unit while the name states a {pitch_in_name} um pitch",
            ))
    paths = [str(d.get("path")) for d in datasets if isinstance(d, dict)]
    try:
        nums = [int(p) for p in paths]
        if nums != list(range(len(nums))):
            out.append(Finding(ERROR, "LEVELS_NOT_CONTIGUOUS", f"declared level paths {paths} are not 0..n-1"))
    except ValueError:
        out.append(Finding(
            WARN, "LEVELS_NON_NUMERIC",
            f"declared level paths {paths} are not integers, so level ordering is not obvious",
        ))
    return out


def check_level0_scale(scale, pitch_in_name: float | None) -> list[Finding]:
    """A level-0 scale of [1,1,1] is either a genuine unit voxel or a missing physical scale."""
    if scale is None:
        return [Finding(ERROR, "NO_SCALE", "level 0 declares no scale")]
    if _is_all_one(scale):
        if pitch_in_name is not None:
            return [Finding(
                ERROR, "SCALE_IS_UNIT",
                f"level-0 scale is {list(scale)} while the name states a {pitch_in_name} um pitch, so "
                f"every consumer computes sizes off by that factor",
            )]
        return [Finding(
            WARN, "SCALE_UNITLESS",
            "level-0 scale is [1,1,1]; correct only for a genuinely unit-voxel store",
        )]
    if pitch_in_name is not None and abs(float(scale[-1]) - pitch_in_name) > 1e-6:
        return [Finding(
            WARN, "SCALE_DISAGREES_WITH_NAME",
            f"level-0 scale {list(scale)} disagrees with the {pitch_in_name} um in the name",
        )]
    return []


def check_array_header(header: dict) -> list[Finding]:
    """Checks that need one level's .zarray / zarr.json."""
    out: list[Finding] = []
    shape, chunks = header.get("shape"), header.get("chunks")
    if shape and chunks and any(c > s for c, s in zip(chunks, shape)):
        out.append(Finding(ERROR, "CHUNK_LARGER_THAN_SHAPE", f"chunks {chunks} exceed shape {shape}"))
    fill = header.get("fill_value")
    if fill not in (0, 0.0, None):
        out.append(Finding(
            WARN, "NONZERO_FILL_VALUE",
            f"fill_value is {fill}; blank regions will not read as zero, which an ink detector can "
            f"mistake for signal",
        ))
    return out


def check_level_headers(declared: list[str], header_for) -> list[Finding]:
    """Every declared level must exist; the last one especially, since it is the coarsest."""
    out: list[Finding] = []
    for path in declared:
        if header_for(path) is None:
            out.append(Finding(ERROR, "LEVEL_MISSING", f"declared level {path} has no array header"))
    return out


def pitch_from_name(name: str):
    """The micrometre pitch a store name states, if it states one.

    Names of the form `1.129um-...-L1.zarr` state the pitch of the *source scan* and the level they
    were taken from, so the expected level-0 pitch is that value doubled per level. Comparing the
    raw token against level 0 flagged healthy stores.
    """
    m = re.search(r"(\d+(?:\.\d+)?)um", name)
    if not m:
        return None
    pitch = float(m.group(1))
    lvl = re.search(r"[-_]L(\d+)(?=[-_.]|$)", name)
    if lvl:
        pitch *= 2 ** int(lvl.group(1))
    return pitch


CHUNK_META_SUFFIXES = (".zarray", ".zattrs", "zarr.json", ".zmetadata", ".zgroup")


def is_chunk_key(key: str) -> bool:
    """A level directory holds metadata files next to the actual chunk objects."""
    if key.endswith(CHUNK_META_SUFFIXES) or key.endswith("/"):
        return False
    return True


def check_has_chunks(keys: list[str] | None) -> list[Finding]:
    """A store whose level 0 holds no chunks reads back as fill_value everywhere, silently."""
    if keys is None:
        return [Finding(WARN, "CHUNKS_UNKNOWN", "could not list the store, so chunk presence is unverified")]
    chunks = [k for k in keys if is_chunk_key(k)]
    if not chunks:
        return [Finding(
            ERROR, "NO_CHUNKS",
            f"level 0 exists but holds no chunks ({len(keys)} metadata key(s) only); readers get "
            f"fill_value only",
        )]
    return []


class Store:
    """Fetches store metadata from a URL or a local directory."""

    def __init__(self, root: str, bucket: str = DEFAULT_BUCKET):
        import os
        self.raw = root
        self.base = bucket.rstrip("/")
        # A relative path is a catalog path served by the bucket unless it actually exists on disk.
        # Treating it as local unconditionally made every relative --root report "no metadata".
        self.local = root if os.path.isdir(root) else None
        self.is_url = root.startswith(("http://", "https://"))
        self.rel = root
        if self.is_url:
            p = urllib.parse.urlparse(root)
            self.base = f"{p.scheme}://{p.netloc}"
            self.rel = p.path.lstrip("/")

    def _get(self, rel: str):
        url = f"{self.base}/{urllib.parse.quote(rel)}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=USER_AGENT), timeout=45) as r:
                b = r.read()
        except Exception:
            return None
        if b[:2] == b"\x1f\x8b":
            b = gzip.decompress(b)
        try:
            return json.loads(b.decode("utf-8"))
        except Exception:
            return None

    def json_at(self, rel: str):
        if self.local:
            import os
            p = os.path.join(self.local, rel)
            if not os.path.exists(p):
                return None
            raw = open(p, "rb").read()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception:
                return None
        return self._get(f"{self.rel.rstrip('/')}/{rel}" if self.rel else rel)

    def multiscales(self):
        attrs = self.json_at(".zattrs")
        if isinstance(attrs, dict):
            ms = (attrs.get("multiscales") or [None])[0]
            if isinstance(ms, dict):
                return ms, "v2"
        z = self.json_at("zarr.json")
        if isinstance(z, dict):
            ms = ((z.get("attributes") or {}).get("multiscales") or [None])[0]
            if isinstance(ms, dict):
                return ms, "v3"
        return None, None

    def level_header(self, level: str, fmt: str | None):
        for name in (("zarr.json",) if fmt == "v3" else (".zarray", "zarr.json")):
            h = self.json_at(f"{level}/{name}")
            if isinstance(h, dict):
                return h
        return None

    def level0_keys(self):
        """First page of a level-0 listing, or None when listing is unavailable."""
        if self.local:
            import os
            d = os.path.join(self.local, "0")
            return sorted(os.listdir(d)) if os.path.isdir(d) else None
        if not self.base.endswith("amazonaws.com"):
            return None
        prefix = f"{self.rel.rstrip('/')}/0/"
        q = urllib.parse.urlencode({"list-type": "2", "prefix": prefix, "max-keys": "1000"})
        try:
            with urllib.request.urlopen(
                urllib.request.Request(f"{self.base}/?{q}", headers=USER_AGENT), timeout=45
            ) as r:
                body = r.read().decode("utf-8", "replace")
        except Exception:
            return None
        return re.findall(r"<Key>([^<]+)</Key>", body)


def scale_of(dataset, level0_header=None):
    """The level's scale, from either OME-NGFF shape.

    v0.4 and v3 both nest it as datasets[i].coordinateTransformations[0].scale; a few producers put it
    directly on the dataset. Reading only the flat form reported "no scale" for every real store.
    """
    if isinstance(dataset, dict):
        if isinstance(dataset.get("scale"), (list, tuple)):
            return list(dataset["scale"])
        for t in dataset.get("coordinateTransformations") or []:
            if isinstance(t, dict) and t.get("type") == "scale" and isinstance(t.get("scale"), (list, tuple)):
                return list(t["scale"])
    if isinstance(level0_header, dict):
        for key in ("scale", "_scale"):
            if isinstance(level0_header.get(key), (list, tuple)):
                return list(level0_header[key])
    return None


def run_checks(store: Store, pitch_in_name: float | None, check_chunks: bool) -> list[Finding]:
    findings: list[Finding] = []
    ms, fmt = store.multiscales()
    pitch = pitch_in_name if pitch_in_name is not None else pitch_from_name(store.raw)
    findings += check_multiscales(ms, pitch)
    if not isinstance(ms, dict):
        return findings
    datasets = [str(d.get("path")) for d in (ms.get("datasets") or []) if isinstance(d, dict)]
    level0 = store.level_header(datasets[0], fmt) if datasets else None
    if level0 is None:
        findings.append(Finding(ERROR, "LEVEL0_MISSING", "the first declared level has no array header"))
    else:
        findings += check_array_header(level0)
        findings += check_level0_scale(scale_of((ms.get("datasets") or [{}])[0], level0), pitch)
    findings += check_level_headers(datasets, lambda lvl: store.level_header(lvl, fmt))
    if check_chunks:
        findings += check_has_chunks(store.level0_keys())
    return findings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="store path, local directory, or full URL")
    ap.add_argument("--bucket", default=DEFAULT_BUCKET, help="base for a relative --root")
    ap.add_argument("--name-states-pitch", type=float, default=None,
                    help="micrometre pitch the store name states; inferred from the name when omitted")
    ap.add_argument("--no-chunks", action="store_true", help="skip the level-0 listing")
    ap.add_argument("--json", action="store_true", help="emit findings as JSON")
    args = ap.parse_args(argv)

    store = Store(args.root, args.bucket)
    findings = run_checks(store, args.name_states_pitch, not args.no_chunks)

    if args.json:
        print(json.dumps([f.as_dict() for f in findings], indent=2))
    else:
        print(f"store: {args.root}")
        if not findings:
            print("  no findings")
        for f in findings:
            print(f"  {f}")
    return 1 if any(f.level == ERROR for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
