from __future__ import annotations

import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import check_omezarr_metadata as check


def multiscales(units, scale, levels=("0", "1"), axes=("z", "y", "x")):
    return {
        "axes": [{"name": n, "type": "space", "unit": u} for n, u in zip(axes, units)],
        "datasets": [{"path": p, "scale": list(scale)} for p in levels],
    }


class MultiscalesTest(unittest.TestCase):
    def test_missing_block_is_an_error(self) -> None:
        findings = check.check_multiscales(None, None)
        self.assertEqual([f.code for f in findings], ["NO_MULTISCALES"])

    def test_unitless_with_pitch_in_name_is_an_error(self) -> None:
        ms = multiscales((None, None, None), (1.0, 1.0, 1.0))
        findings = check.check_multiscales(ms, 2.4)
        self.assertIn("UNITS_MISSING_PITCH_IN_NAME", [f.code for f in findings])
        self.assertTrue(all(f.level == check.ERROR for f in findings))

    def test_unitless_without_pitch_in_name_is_only_a_warning(self) -> None:
        ms = multiscales((None, None, None), (1.0, 1.0, 1.0))
        findings = check.check_multiscales(ms, None)
        self.assertEqual([f.level for f in findings], [check.WARN])

    def test_units_present_is_clean(self) -> None:
        ms = multiscales(("micrometer",) * 3, (2.399, 2.399, 2.399))
        self.assertEqual(check.check_multiscales(ms, 2.399), [])

    def test_partial_units_are_reported_when_a_pitch_is_named(self) -> None:
        ms = multiscales(("micrometer", None, "micrometer"), (2.4, 2.4, 2.4))
        findings = check.check_multiscales(ms, 2.4)
        self.assertEqual([f.code for f in findings], ["UNITS_PARTIAL"])

    def test_levels_must_be_contiguous(self) -> None:
        ms = multiscales(("micrometer",) * 3, (2.4, 2.4, 2.4), levels=("0", "1", "3"))
        findings = check.check_multiscales(ms, 2.4)
        self.assertEqual([f.code for f in findings], ["LEVELS_NOT_CONTIGUOUS"])

    def test_no_datasets_is_an_error(self) -> None:
        ms = multiscales(("micrometer",) * 3, (2.4, 2.4, 2.4), levels=())
        findings = check.check_multiscales(ms, 2.4)
        self.assertIn("NO_DATASETS", [f.code for f in findings])


class Level0ScaleTest(unittest.TestCase):
    def test_unit_scale_with_pitch_in_name_is_an_error(self) -> None:
        findings = check.check_level0_scale([1.0, 1.0, 1.0], 8.64)
        self.assertEqual([f.code for f in findings], ["SCALE_IS_UNIT"])
        self.assertEqual(findings[0].level, check.ERROR)

    def test_unit_scale_without_a_named_pitch_is_a_warning(self) -> None:
        findings = check.check_level0_scale([1.0, 1.0, 1.0], None)
        self.assertEqual([f.code for f in findings], ["SCALE_UNITLESS"])

    def test_matching_scale_is_clean(self) -> None:
        self.assertEqual(check.check_level0_scale([2.399, 2.399, 2.399], 2.399), [])

    def test_scale_disagreeing_with_the_name_is_flagged(self) -> None:
        findings = check.check_level0_scale([2.4, 2.4, 2.4], 8.64)
        self.assertEqual([f.code for f in findings], ["SCALE_DISAGREES_WITH_NAME"])

    def test_missing_scale_is_an_error(self) -> None:
        self.assertEqual(check.check_level0_scale(None, 2.4)[0].code, "NO_SCALE")


class ArrayHeaderTest(unittest.TestCase):
    def test_chunk_larger_than_shape_is_an_error(self) -> None:
        findings = check.check_array_header({"shape": [10, 10, 10], "chunks": [128, 128, 128]})
        self.assertEqual([f.code for f in findings], ["CHUNK_LARGER_THAN_SHAPE"])

    def test_nonzero_fill_value_is_reported(self) -> None:
        findings = check.check_array_header({"shape": [10], "chunks": [10], "fill_value": 255})
        self.assertEqual([f.code for f in findings], ["NONZERO_FILL_VALUE"])

    def test_zero_fill_value_is_clean(self) -> None:
        self.assertEqual(check.check_array_header({"shape": [10], "chunks": [10], "fill_value": 0}), [])


class LevelHeaderTest(unittest.TestCase):
    def test_a_declared_level_without_a_header_is_an_error(self) -> None:
        findings = check.check_level_headers(["0", "1"], lambda lvl: None if lvl == "1" else {})
        self.assertEqual([f.code for f in findings], ["LEVEL_MISSING"])

    def test_all_levels_present_is_clean(self) -> None:
        self.assertEqual(check.check_level_headers(["0", "1"], lambda lvl: {}), [])


class ChunkPresenceTest(unittest.TestCase):
    def test_no_chunks_is_an_error(self) -> None:
        self.assertEqual(check.check_has_chunks([])[0].code, "NO_CHUNKS")

    def test_unlistable_is_a_warning_not_an_error(self) -> None:
        findings = check.check_has_chunks(None)
        self.assertEqual(findings[0].level, check.WARN)

    def test_chunks_present_is_clean(self) -> None:
        self.assertEqual(check.check_has_chunks(["0/0/0/0"]), [])


class PitchFromNameTest(unittest.TestCase):
    def test_reads_the_pitch(self) -> None:
        self.assertEqual(check.pitch_from_name("PHerc0814/.../8.64um-1.2m-116keV-volume-2025.zarr"), 8.64)

    def test_no_pitch_in_name(self) -> None:
        self.assertIsNone(check.pitch_from_name(".../20260319104112-surface-20260413222639.zarr"))


class ScaleOfTest(unittest.TestCase):
    """The scale is nested in OME-NGFF; reading only the flat form reported "no scale" everywhere."""

    def test_reads_coordinate_transformations(self) -> None:
        ds = {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [2.4, 2.4, 2.4]}]}
        self.assertEqual(check.scale_of(ds), [2.4, 2.4, 2.4])

    def test_reads_flat_scale_too(self) -> None:
        self.assertEqual(check.scale_of({"path": "0", "scale": [1.0, 1.0, 1.0]}), [1.0, 1.0, 1.0])

    def test_falls_back_to_the_header(self) -> None:
        self.assertEqual(check.scale_of({}, {"scale": [3.0, 3.0, 3.0]}), [3.0, 3.0, 3.0])

    def test_none_when_absent(self) -> None:
        self.assertIsNone(check.scale_of({"path": "0"}, {}))

    def test_the_1951_store_shape_is_read(self) -> None:
        ds = {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0]}]}
        self.assertEqual([f.code for f in check.check_level0_scale(check.scale_of(ds), 8.64)], ["SCALE_IS_UNIT"])


class FakeStore(check.Store):
    """A store served from dicts, so the end-to-end path is tested without a network."""

    def __init__(self, attrs=None, zarr_json=None, headers=None, keys=("0/0/0/0",), listed=True):
        self.raw = "PHerc0814/segments/x/surface-volumes/1.129um-0.22m-59keV-volume-20260521123630-L1.zarr"
        self.base = "https://example.invalid"
        self.local = None
        self.rel = self.raw
        self._attrs, self._zarr, self._headers = attrs, zarr_json, headers or {}
        self._keys = list(keys) if listed else None

    def json_at(self, rel):
        if rel == ".zattrs":
            return self._attrs
        if rel == "zarr.json":
            return self._zarr
        return self._headers.get(rel)

    def level0_keys(self):
        return self._keys


class EndToEndTest(unittest.TestCase):
    def test_the_1951_store_reports_scale_is_unit(self) -> None:
        ms = multiscales((None, None, None), (1.0, 1.0, 1.0))
        store = FakeStore(attrs={"multiscales": [ms]}, headers={"0/.zarray": {"shape": [10, 10, 10], "chunks": [128, 128, 128], "fill_value": 0}})
        codes = [f.code for f in check.run_checks(store, None, True)]
        self.assertIn("SCALE_IS_UNIT", codes)
        self.assertIn("UNITS_MISSING_PITCH_IN_NAME", codes)

    def test_the_1892_store_reports_no_chunks(self) -> None:
        ms = multiscales(("micrometer",) * 3, (1.129, 1.129, 1.129))
        store = FakeStore(attrs={"multiscales": [ms]}, keys=(), listed=True)
        findings = check.run_checks(store, None, True)
        self.assertIn("NO_CHUNKS", [f.code for f in findings])

    def test_a_healthy_v3_store_has_no_findings(self) -> None:
        # The fake store's name ends in -L1 and states 1.129 um, so its level-0 pitch is 2.258 and the
        # fixture has to say so. Both mistakes here were caught by the check itself, which is the point.
        ms = multiscales(("micrometer",) * 2, (2.258, 2.258), levels=("0",), axes=("y", "x"))
        store = FakeStore(zarr_json={"attributes": {"multiscales": [ms]}}, headers={"0/zarr.json": {"shape": [10], "chunks": [10], "fill_value": 0}})
        self.assertEqual(check.run_checks(store, None, True), [])

    def test_an_L_name_states_the_source_scan_not_level_zero(self) -> None:
        self.assertEqual(check.pitch_from_name(".../8.64um-1.2m-116keV-volume-20250521151220.zarr"), 8.64)
        self.assertEqual(check.pitch_from_name(".../1.129um-0.22m-59keV-volume-20260521123630-L1.zarr"), 2.258)

    def test_metadata_keys_are_not_chunks(self) -> None:
        keys = ["x/0/.zarray", "x/0/.zattrs", "x/0/0/", "x/0/0/0/0"]
        self.assertEqual([k for k in keys if check.is_chunk_key(k)], ["x/0/0/0/0"])

    def test_a_level_with_only_metadata_reports_no_chunks(self) -> None:
        ms = multiscales(("micrometer",) * 3, (2.258, 2.258, 2.258))
        store = FakeStore(attrs={"multiscales": [ms]}, keys=("x/0/.zarray", "x/0/.zattrs"))
        self.assertIn("NO_CHUNKS", [f.code for f in check.run_checks(store, None, True)])

    def test_v3_multiscales_are_read_from_attributes(self) -> None:
        ms = multiscales(("micrometer",) * 2, (2.4, 2.4), levels=("0",), axes=("y", "x"))
        store = FakeStore(zarr_json={"attributes": {"multiscales": [ms]}})
        found, fmt = store.multiscales()
        self.assertEqual(fmt, "v3")
        self.assertIs(found, ms)

    def test_missing_everything_exits_with_an_error(self) -> None:
        store = FakeStore()
        self.assertTrue(any(f.level == check.ERROR for f in check.run_checks(store, None, False)))


if __name__ == "__main__":
    unittest.main()
