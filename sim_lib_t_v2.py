temporal_relationship = [
    'precedes', 'is_preceded_by', 'meets', 'is_met_by',\
    'overlaps_with', 'is_overlapped_by', 'starts', 'is_started_by',\
    'during', 'contains', 'finishes', 'finished_by',\
    'is_equal_to'
]

import numpy as np
from allen_interval_algebra import *
import pdb 

# ------------------------------------------------------------------#
# --------------------   interval helpers   ------------------------#
# ------------------------------------------------------------------#

def _random_interval(min_start=0.0, max_start=80.0,
                     min_len=1.0,  max_len=20.0):
    """Uniform random closed interval [start, end] on the real line (≥0)."""
    start  = np.random.uniform(min_start, max_start)
    length = np.random.uniform(min_len,  max_len)
    return (start, start + length)

def _inner_interval(outer, shrink=0.3):
    """Return an interval strictly inside `outer` (same centre)."""
    s, e = outer
    mid  = 0.5 * (s + e)
    half = 0.5 * (e - s) * (1 - shrink)
    return (mid - half, mid + half)

def _shift_interval(iv, dx):
    """Translate interval by `dx`."""
    s, e = iv
    return (s + dx, e + dx)

def _shift_pair_non_negative(iv1, iv2, margin=1e-6):
    """If either interval has a negative coord, shift *both* rightwards."""
    min_val = min(iv1[0], iv1[1], iv2[0], iv2[1])
    if min_val < 0:
        iv1 = _shift_interval(iv1, -min_val + margin)
        iv2 = _shift_interval(iv2, -min_val + margin)
    return iv1, iv2

# ------------------------------------------------------------------#
# ---------------  generic generator “skeleton”  -------------------#
# ------------------------------------------------------------------#

def _make_pair(interval1, satisfy_fn, make_pos, make_neg):
    """
    Build (interval1, interval2, label) where label==1  ⇔  satisfy_fn(...)
    Positive / negative chosen with 50 % probability.
    """
    want_positive = np.random.rand() < 0.5
    builder       = make_pos if want_positive else make_neg

    while True:
        interval2 = builder()
        truth     = satisfy_fn(interval1, interval2)
        if truth == want_positive:
            # final hygiene step → no negatives anywhere
            interval1, interval2 = _shift_pair_non_negative(interval1, interval2)
            return interval1, interval2, int(truth)

# ------------------------------------------------------------------#
# --------------  generators for all 13 relations  -----------------#
# ------------------------------------------------------------------#

def gen_precedes(interval1=None):
    if interval1 is None: interval1 = _random_interval()

    def pos():
        gap   = np.random.uniform(0.1, 10)
        len2  = np.random.uniform(1, 20)
        s2    = interval1[1] + gap
        return (s2, s2 + len2)

    def neg(): return _random_interval()

    return _make_pair(interval1, precedes, pos, neg)

def gen_is_preceded_by(interval1=None):
    if interval1 is None: interval1 = _random_interval()

    def pos():
        # Ensure end2 ≥ 0 by making interval1 start far enough from 0
        len2 = np.random.uniform(1, 20)
        max_gap = interval1[0] - len2 - 0.1
        gap  = np.random.uniform(0.1, max(0.2, max_gap))
        e2   = interval1[0] - gap
        return (e2 - len2, e2)

    def neg(): return _random_interval()

    return _make_pair(interval1, is_preceded_by, pos, neg)

def gen_meets(interval1=None):
    if interval1 is None: interval1 = _random_interval()

    def pos():
        len2 = np.random.uniform(1, 20)
        s2   = interval1[1]
        return (s2, s2 + len2)

    def neg(): return _random_interval()

    return _make_pair(interval1, meets, pos, neg)

def gen_is_met_by(interval1=None):
    if interval1 is None: interval1 = _random_interval()

    def pos():
        len2 = np.random.uniform(1, 20)
        e2   = interval1[0]
        return (e2 - len2, e2)

    def neg(): return _random_interval()

    return _make_pair(interval1, is_met_by, pos, neg)

def gen_overlaps_with(interval1=None):
    if interval1 is None: interval1 = _random_interval()

    def pos():
        s2   = np.random.uniform(interval1[0] + 0.1, interval1[1] - 0.1)
        len2 = np.random.uniform(interval1[1] - s2 + 0.1, 20)
        return (s2, s2 + len2)

    def neg(): return _random_interval()

    return _make_pair(interval1, overlaps_with, pos, neg)

def gen_is_overlapped_by(interval1=None):
    if interval1 is None: interval1 = _random_interval()

    def pos():
        e2 = np.random.uniform(interval1[0] + 0.1, interval1[1] - 0.1)
        len2 = np.random.uniform(1, e2 - 0.1)
        s2 = e2 - len2
        # guarantee s2 < start1
        if s2 >= interval1[0]:
            s2 = interval1[0] - np.random.uniform(0.1, 2)
        return (s2, e2)

    def neg(): return _random_interval()

    return _make_pair(interval1, is_overlapped_by, pos, neg)

def gen_starts(interval1=None):
    if interval1 is None: interval1 = _random_interval()

    def pos():
        len2 = np.random.uniform(interval1[1] - interval1[0] + 0.1, 20)
        return (interval1[0], interval1[0] + len2)

    def neg(): return _random_interval()

    return _make_pair(interval1, starts, pos, neg)

def gen_is_started_by(interval1=None):
    if interval1 is None: interval1 = _random_interval()

    def pos():
        inner = _inner_interval(interval1, shrink=0.5)
        return (interval1[0], inner[1])

    def neg(): return _random_interval()

    return _make_pair(interval1, is_started_by, pos, neg)

def gen_during(interval1=None):
    if interval1 is None: interval1 = _random_interval()

    def pos():
        len_outer = (interval1[1] - interval1[0]) * np.random.uniform(1.2, 3.0)
        left_pad  = np.random.uniform(0.1, len_outer - (interval1[1]-interval1[0]) - 0.1)
        s2        = interval1[0] - left_pad
        return (s2, s2 + len_outer)

    def neg(): return _random_interval()
    return _make_pair(interval1, during, pos, neg)

def gen_contains(interval1=None):
    if interval1 is None: interval1 = _random_interval()

    def pos():
        return _inner_interval(interval1, shrink=np.random.uniform(0.2, 0.8))

    def neg(): return _random_interval()
    return _make_pair(interval1, contains, pos, neg)

def gen_finishes(interval1=None):
    if interval1 is None: interval1 = _random_interval()

    def pos():
        extra = np.random.uniform(0.1, interval1[1] - interval1[0] - 0.1)
        return (interval1[0] - extra, interval1[1])

    def neg(): return _random_interval()
    return _make_pair(interval1, finishes, pos, neg)

def gen_finished_by(interval1=None):
    if interval1 is None: interval1 = _random_interval()

    def pos():
        inner = _inner_interval(interval1, shrink=0.5)
        return (inner[0], interval1[1])

    def neg(): return _random_interval()
    return _make_pair(interval1, finished_by, pos, neg)

def gen_is_equal_to(interval1=None):
    if interval1 is None: interval1 = _random_interval()

    def pos(): return interval1
    def neg():
        while True:
            iv = _random_interval()
            if not is_equal_to(interval1, iv):
                return iv

    return _make_pair(interval1, is_equal_to, pos, neg)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for gen in [gen_precedes, gen_is_preceded_by, gen_meets, gen_is_met_by,
                gen_overlaps_with, gen_is_overlapped_by, gen_starts,
                gen_is_started_by, gen_during, gen_contains,
                gen_finishes, gen_finished_by, gen_is_equal_to]:
        for _ in range(20):
            interval =  _random_interval(min_start=0, max_start=1,
                     min_len=0.5, max_len=1)
            i1, i2, y = gen(interval)
            func = globals()[gen.__name__[4:]]
            sanity_check = func(i1, i2)
            print(f"{gen.__name__:<20} i1=({i1[0]:.4f}, {i1[1]:.4f}) i2=({i2[0]:.4f}, {i2[1]:.4f})  label={y}")
            if y != sanity_check:
                pdb.set_trace()