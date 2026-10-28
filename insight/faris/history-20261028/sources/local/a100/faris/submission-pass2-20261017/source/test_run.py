import unittest
import numpy as np
from run import combine, describe


class EnsembleTests(unittest.TestCase):
    def test_trim_and_order(self):
        arrays = {'a': np.array([.1, .8]), 'b': np.array([.5, .4]), 'c': np.array([.9, .1])}
        for parents in [['a', 'b', 'c'], ['c', 'a', 'b']]:
            np.testing.assert_allclose(combine({'parents': parents, 'method': 'trimmed'}, arrays), [.5, .4])

    def test_identical_predictions_and_probability_bounds(self):
        p = np.array([1e-8, .2, .5, .99])
        arrays = {key: p for key in ['a', 'b', 'c']}
        for method in ['trimmed', 'mean', 'geometric', 'logit']:
            np.testing.assert_allclose(combine({'parents': list(arrays), 'method': method}, arrays), p, atol=1e-12)

    def test_invalid_prediction_rejected(self):
        with self.assertRaises(AssertionError):
            combine({'parents': ['a'], 'method': 'mean'}, {'a': np.array([np.nan])})

    def test_fold_gate_rejects_a_damaged_fold(self):
        y = np.tile([0, 0, 1, 1], 3)
        folds = np.repeat(np.arange(3), 4)
        baseline = np.tile([.1, .2, .8, .9], 3)
        self.assertTrue(describe(y, baseline, folds, baseline)['gate'])
        damaged = baseline.copy()
        damaged[folds == 1] = .5
        self.assertFalse(describe(y, damaged, folds, baseline)['gate'])


if __name__ == '__main__':
    unittest.main()
