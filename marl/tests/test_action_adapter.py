from __future__ import annotations

import unittest

import numpy as np

from marl.adapters import FlatActionAdapter


class FlatActionAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = FlatActionAdapter(23)

    def test_round_trip_and_bess_value_is_unchanged(self) -> None:
        idc = np.linspace(0.0, 1.0, 22, dtype=np.float32)
        bess = np.asarray([0.25], dtype=np.float32)
        flat = self.adapter.compose_action(idc, bess)
        restored_idc, restored_bess = self.adapter.split_action(flat)
        self.assertEqual(flat.shape, (23,))
        self.assertEqual(flat.dtype, np.float32)
        np.testing.assert_array_equal(restored_idc, idc)
        np.testing.assert_array_equal(restored_bess, bess)
        self.assertEqual(float(flat[22]), 0.25)

    def test_invalid_values_are_rejected_without_clipping(self) -> None:
        valid_idc = np.full(22, 0.5, dtype=np.float32)
        valid_bess = np.asarray([0.5], dtype=np.float32)
        invalid_cases = (
            ("not-an-array", valid_bess, TypeError),
            (np.full(21, 0.5), valid_bess, ValueError),
            (np.append(valid_idc[:-1], np.nan), valid_bess, ValueError),
            (np.append(valid_idc[:-1], np.inf), valid_bess, ValueError),
            (np.append(valid_idc[:-1], -0.01), valid_bess, ValueError),
            (valid_idc, np.asarray([1.01]), ValueError),
            (valid_idc, np.asarray([[0.5]]), ValueError),
        )
        for idc, bess, error_type in invalid_cases:
            with self.subTest(idc_shape=np.shape(idc), bess_shape=np.shape(bess)):
                with self.assertRaises(error_type):
                    self.adapter.compose_action(idc, bess)


if __name__ == "__main__":
    unittest.main()
