"""Monitor boundaries and conservative handling of unusable batches."""
import unittest
from part3_machine_learning.monitoring import check_hours


class MonitoringTests(unittest.TestCase):
    def test_same_and_shifted(self):
        reference = list(range(24))*10
        self.assertEqual(check_hours(reference, reference)['status'], 'PASS')
        shifted = check_hours(reference, [8]*100)
        self.assertEqual(shifted['status'], 'ALERT')
        self.assertAlmostEqual(shifted['total_variation_distance'], 23/24)

    def test_bad_and_small(self):
        self.assertIn('insufficient_batch', check_hours([8]*100, [8]*5)['reasons'])
        for hours in ([], [24], [-1], [True], [8.0]):
            with self.assertRaises(ValueError):
                check_hours([8]*100, hours)
