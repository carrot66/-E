"""Run: python -m unittest -v test_alignment (no network/GPU required)."""
import itertools
import unittest
import numpy as np
from alignment import ctc_viterbi, interval_coverage, overlap_pool, overlap_scalar, time_grid


class AlignmentTests(unittest.TestCase):
    def test_ctc_matches_exhaustive_optimum(self):
        rng = np.random.default_rng(7)
        for tokens in ([1, 2], [1, 1], [2], [2, 1, 2]):
            logits = rng.normal(size=(5, 3))
            logp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
            best, best_path = -np.inf, None
            for path in itertools.product(range(3), repeat=5):
                collapsed = [k for j,k in enumerate(path) if (j==0 or k!=path[j-1]) and k!=0]
                if collapsed == tokens:
                    score = sum(logp[j,k] for j,k in enumerate(path))
                    if score > best:
                        best, best_path = score, path
            spans = ctc_viterbi(logp, tokens, blank=0)
            recovered = np.zeros(5, int)
            for token, (a,b,_) in zip(tokens, spans):
                recovered[a:b] = token
            np.testing.assert_array_equal(recovered, best_path)

    def test_ctc_repeated_token_requires_blank(self):
        with self.assertRaises(ValueError):
            ctc_viterbi(np.zeros((2, 3)), [1, 1], 0)

    def test_overlap_moments_missing_and_boundary(self):
        grid = time_grid(1.2, .5)
        x, mask, w = overlap_pool([[0,.25],[.25,.5],[.5,1.]],
                                 [[2],[6],[np.nan]], grid, std=True)
        np.testing.assert_allclose(x, [[4,2],[0,0],[0,0]])
        np.testing.assert_array_equal(mask, [True,False,False])
        np.testing.assert_allclose(grid[-1], [1.,1.2])
        self.assertEqual(w[1,0], 0)

    def test_empty_modality(self):
        x, mask, w = overlap_pool([], np.empty((0,25)), time_grid(1,.5), std=True)
        self.assertEqual(x.shape, (2,50))
        self.assertFalse(mask.any())
        self.assertEqual(w.shape, (2,0))

    def test_coverage_is_union_and_quality_is_weighted(self):
        grid = time_grid(1., .5)
        intervals = np.array([[0., .4], [.2, .6]])
        coverage = interval_coverage(intervals, grid)
        np.testing.assert_allclose(coverage, [1., .2])
        quality, mask, _ = overlap_scalar(intervals, [0.2, 0.8], grid)
        np.testing.assert_allclose(quality, [(0.4*.2 + 0.3*.8)/.7, .8])
        np.testing.assert_array_equal(mask, [True, True])

    def test_invalid_grid(self):
        for d in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                time_grid(d, .5)


if __name__ == '__main__':
    unittest.main()
