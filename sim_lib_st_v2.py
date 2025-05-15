import pdb 
from tqdm import tqdm 

def coords_to_string(coords):
    return "["+", ".join(f"({x:.4f}, {y:.4f})" for x, y in coords)+"]"

def format_geometry_object(geo, geo_type, include_geo_type=True):
    if geo_type in ('Point', 'Linestring'):
        if not include_geo_type:
            geo_type = ''
        return geo_type + ' ' + coords_to_string(geo.coords)
    elif geo_type in ('Polygon'):
        if not include_geo_type:
            geo_type = ''
        return geo_type + ' ' + coords_to_string(geo.exterior.coords)
    
sub_spatial_relationship = ['Linestring_Point_intersects', 'Linestring_Linestring_equals', 'Linestring_Linestring_intersects', \
'Linestring_Linestring_contains', 'Linestring_Linestring_crosses', \
'Linestring_Linestring_touches', 'Linestring_Linestring_overlaps', 'Linestring_Polygon_intersects', \
'Linestring_Polygon_within', 'Linestring_Polygon_crosses', 'Linestring_Polygon_touches']
# ]

sub_temporal_relationship = [
    'precedes', 'meets',\
    'overlaps_with', 'starts',\
    'during', 'finishes',\
    'is_equal_to'
]

# sub_spatial_relationship = ['Linestring_Polygon_within']

# sub_temporal_relationship = [
#     'finishes'
# ]

template = [
    'Does the interval when {event_1} {temporal_relationship} the interval {interval_2}?'
]

event = 'the {geo1} {spatial_relationship} {geo2}'

import numpy as np
from shapely.geometry import Point, LineString, Polygon
from shapely.affinity import translate, scale
from shapely.ops import snap 

def _rand_idx(n, forbid=None):
    """Return a random index in range(n) different from `forbid` (if given)."""
    while True:
        i = np.random.randint(n)
        if i != forbid:
            return i

def _two_distinct_indices(n):
    i = np.random.randint(n)
    j = _rand_idx(n, forbid=i)
    return i, j

def _random_point(loc):
    """Return shapely Point sampled uniformly from the given ndarray."""
    idx = np.random.randint(len(loc))
    return Point(loc[idx])

def _point_on_segment(loc, t=None):
    pts = loc 
    """
    Return a point on a segment defined by a (10, 2) array, snapped to ensure topological intersection.
    The function selects a random pair of adjacent points to define the segment.
    """
    assert pts.shape == (10, 2), "Input must be a (10, 2) numpy array."
    
    i = np.random.randint(0, 9)  # Choose a segment between pts[i] and pts[i+1]
    a = tuple(pts[i])
    b = tuple(pts[i + 1])
    
    if t is None:
        t = np.random.uniform(0.15, 0.85)  # Avoid endpoints

    ls = LineString([a, b])
    p = ls.interpolate(t, normalized=True)

    # Snap both ways
    p = snap(p, ls, tolerance=1e-8)
    ls = snap(ls, p, tolerance=1e-8)

    # Replace pts[i] and pts[i+1] with snapped coordinates from `ls`
    snapped_coords = list(ls.coords)
    pts[i] = snapped_coords[0]
    pts[i + 1] = snapped_coords[1]

    return p, pts, i

def _subsegment_of(ls):
    """Return a proper sub-segment of a LineString (strictly inside), with both geometries snapped for topological alignment."""
    t1, t2 = sorted(np.random.uniform(0.15, 0.85, size=2))
    p1 = ls.interpolate(t1, normalized=True)
    p2 = ls.interpolate(t2, normalized=True)
    ls_small = LineString([p1, p2])

    # Snap both geometries to each other to ensure topological consistency
    ls_small = snap(ls_small, ls, tolerance=1e-8)
    ls = snap(ls, ls_small, tolerance=1e-8)

    return ls_small, ls

def _orthogonal_cross(center, length=2.0):
    """Return two orthogonal LineStrings that cross at `center`."""
    cx, cy = center
    half = length / 2
    ls1 = LineString([(cx - half, cy), (cx + half, cy)])        # horizontal
    ls2 = LineString([(cx, cy - half), (cx, cy + half)])        # vertical
    return ls1, ls2

def _random_linestring(loc):
    """Two‑point LineString with distinct vertices from loc array."""
    i, j = _two_distinct_indices(len(loc))
    return LineString([tuple(loc[i]), tuple(loc[j])])

from shapely.affinity import rotate

def _make_regular_polygon(centre, radius=1.0, num_sides=6):
    """Return a regular polygon centered at `centre` with given radius and number of sides."""
    x, y = centre
    angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
    coords = [(x + radius * np.cos(a), y + radius * np.sin(a)) for a in angles]
    return Polygon(coords)

def _random_polygon(loc, sides=None):
    """Generate a random polygon: could be a square or a more complex regular polygon (5–8 sides)."""
    centre = loc[np.random.randint(len(loc))]
    if sides is None:
        num_sides = np.random.choice([4, 5, 6, 7, 8])  # square or more complex
        angle = np.random.uniform(0, 360)  # random rotation
    else:
        num_sides = sides
        angle = 0
    radius = np.random.uniform(0.5, 1.5)
    poly = _make_regular_polygon(centre, radius, num_sides)
    
    # return rotate(poly, angle, origin=centre)
    return rotate(poly, angle, origin=tuple(centre))

def gen_Linestring_Point_intersects(object_location, timestamp):
    # p  = ls.interpolate(np.random.uniform(0.05, 0.95), normalized=True)
    p, updated_object_location, i = _point_on_segment(object_location)
    ls = LineString(updated_object_location)
    interval = (timestamp[i], timestamp[i+1])
    return ls, Point(p), interval

def gen_Linestring_Linestring_equals(object_location, timestamp):
    """
    geo2 is geometrically identical to geo1.
    """
    ls1 = LineString(object_location)
    ls2 = LineString(object_location.copy())        # identical, independent object
    interval = (timestamp[0], timestamp[-1])        # entire trajectory
    return ls1, ls2, interval

def gen_Linestring_Linestring_contains(object_location, timestamp):
    """
    geo1 strictly contains geo2 (proper sub-segment).
    """
    ls1 = LineString(object_location)
    # convert the parametrised t-values (0–1) to discrete time–stamps
    start_idx = np.random.choice(len(object_location)-1)
    end_idx = start_idx + 1
    ls2 = LineString(
        [object_location[start_idx], object_location[end_idx]]
    )
    interval = (timestamp[start_idx], timestamp[end_idx])
    return ls1, ls2, interval

# def gen_Linestring_Linestring_overlaps(object_location, timestamp):
#     """
#     geo1 and geo2 partially overlap: they share some but not all points,
#     and neither contains the other.
#     """
#     ls_full = LineString(object_location)

#     # Build two partially overlapping sub-segments of the same carrier line
#     t = sorted(np.random.uniform(0.15, 0.85, size=4))   # t0 < t1 < t2 < t3
#     p0, p1 = ls_full.interpolate(t[0], normalized=True), ls_full.interpolate(t[2], normalized=True)
#     p2, p3 = ls_full.interpolate(t[1], normalized=True), ls_full.interpolate(t[3], normalized=True)

#     ls1 = LineString([p0, p1])      # longer segment
#     ls2 = LineString([p2, p3])      # shifted segment → overlaps with ls1 but
#                                     # neither contains the other

#     # snap for perfect topology
#     ls1 = snap(ls1, ls_full, 1e-8)
#     ls2 = snap(ls2, ls_full, 1e-8)

#     idx0, idx3 = int(t[0]*(len(timestamp)-1)), int(t[3]*(len(timestamp)-1))
#     interval = (timestamp[idx0], timestamp[idx3])
#     return ls1, ls2, interval

def gen_Linestring_Linestring_overlaps(object_location, timestamp):
    """
    ls1 is the full trajectory from object_location.
    ls2 is formed by:
      - taking a random-length core segment from ls1,
      - prepending and appending a random-length noisy segment 
        to make ls2 longer than the core (ensuring partial overlap only).
    """
    ls1 = LineString(object_location)
    object_location = np.array(object_location)
    n_points = len(object_location)

    # Choose a random core segment
    max_core_len = n_points - 4  # leave room for prefix and suffix
    core_len = np.random.randint(2, max_core_len)
    start_idx = np.random.randint(1, n_points - core_len - 1)
    end_idx = start_idx + core_len
    core_segment = object_location[start_idx:end_idx]

    # Create random-length prefix and suffix with perturbed points
    def generate_noisy_segment(anchor_point, num_points, scale=0.01):
        offsets = np.random.uniform(-scale, scale, size=(num_points, 2))
        return anchor_point + offsets

    prefix_len = np.random.randint(1, 4)
    suffix_len = np.random.randint(1, 4)
    prefix_segment = generate_noisy_segment(core_segment[0], prefix_len)
    suffix_segment = generate_noisy_segment(core_segment[-1], suffix_len)

    # Assemble ls2
    ls2_coords = np.vstack([prefix_segment, core_segment, suffix_segment])
    ls2 = LineString(ls2_coords)
    ls2 = snap(ls2, ls1, tolerance=1e-8)

    interval = (timestamp[start_idx], timestamp[end_idx - 1])
    return ls1, ls2, interval

def gen_Linestring_Linestring_intersects(object_location, timestamp):
    """
    geo2 intersects geo1 but does NOT satisfy equals / contains / overlaps / touches.
    We do that by letting the second LS start outside, cross the first one only once,
    and then leave.
    """
    ls1 = LineString(object_location)

    # Choose a random interior point on ls1 as crossing point
    p, _, i = _point_on_segment(object_location)

    # Build an oblique line through that point that is NOT colinear with the carrier
    angle = np.random.uniform(20, 70)            # not 0 / 90  → avoid 'crosses'
    dx = np.cos(np.deg2rad(angle))
    dy = np.sin(np.deg2rad(angle))
    length = np.random.uniform(1.0, 2.0)
    p_start = (p.x - length*dx, p.y - length*dy)
    p_end   = (p.x + length*dx, p.y + length*dy)
    ls2 = LineString([p_start, p_end])

    interval = (timestamp[i], timestamp[i+1])
    return ls1, ls2, interval

def gen_Linestring_Linestring_crosses(object_location, timestamp):
    """
    geo2 crosses geo1 at one interior point with the two interiors intersecting.
    """
    ls1 = LineString(object_location)

    # choose centre point
    p, _, i = _point_on_segment(object_location)
    # Build orthogonal cross, pick vertical member as geo2
    _, ls2 = _orthogonal_cross((p.x, p.y))
    interval = (timestamp[i], timestamp[i+1])
    return ls1, ls2, interval

# def gen_Linestring_Linestring_touches(object_location, timestamp):
#     """
#     geo2 (ls2) touches geo1 (ls1) at a single point, which can be any point on ls1.
#     """
#     ls1 = LineString(object_location)

#     # Randomly interpolate a point along ls1 (not restricted to endpoints)
#     t = np.random.uniform(0.15, 0.85)  # avoid exact endpoints for generality
#     touch_pt = ls1.interpolate(t, normalized=True)
#     touch_coords = (touch_pt.x, touch_pt.y)

#     # Estimate the index in timestamp that corresponds to the touch point
#     idx = int(t * (len(timestamp) - 1))
    
#     # Create a short segment that touches ls1 at touch_pt
#     angle = np.random.uniform(0, 2 * np.pi)
#     length = np.random.uniform(0.5, 1.5)
#     p2 = (touch_coords[0] + length * np.cos(angle),
#           touch_coords[1] + length * np.sin(angle))
#     ls2 = LineString([touch_coords, p2])

#     interval = (timestamp[idx], timestamp[idx])  # instantaneous
#     return ls1, ls2, interval

import numpy as np
from shapely.geometry import LineString

def gen_Linestring_Linestring_touches(object_location, timestamp,
                                      allow_endpoints=False):
    """
    geo2 (ls2) touches geo1 (ls1) at a single vertex of geo1 (no interior point).

    Parameters
    ----------
    object_location : list[(x, y)]
        Vertex coordinates of the first LineString.
    timestamp : list[float]
        One‑to‑one timestamps for each vertex in `object_location`.
    allow_endpoints : bool, optional
        If False (default), the first and last vertices are excluded from
        selection so that the touch point lies strictly inside the polyline.
        Set to True to allow endpoints as the touch vertex.

    Returns
    -------
    ls1 : shapely.geometry.LineString
        Original trajectory polyline.
    ls2 : shapely.geometry.LineString
        Short segment that touches `ls1` only at the chosen vertex.
    interval : tuple(float, float)
        Instantaneous event time at the touch vertex.
    """
    if len(object_location) < 2 or len(object_location) != len(timestamp):
        raise ValueError("`object_location` and `timestamp` must have the same length ≥ 2")

    # Choose a vertex index
    start = 0 if allow_endpoints else 1
    end   = len(object_location) - (0 if allow_endpoints else 1)
    idx   = np.random.randint(start, end)

    touch_coords = object_location[idx]
    ls1 = LineString(object_location)

    # Create a short segment that touches `ls1` only at `touch_coords`
    angle  = np.random.uniform(0, 2 * np.pi)
    length = np.random.uniform(0.5, 1.5)
    p2 = (touch_coords[0] + length * np.cos(angle),
          touch_coords[1] + length * np.sin(angle))
    ls2 = LineString([touch_coords, p2])

    # Instantaneous interval at the chosen vertex
    t_touch = timestamp[idx]
    interval = (t_touch, t_touch)

    return ls1, ls2, interval


###############################################################################
# ---------- 2)  LineString  –  Polygon relationships ------------------------
###############################################################################

def _polygon_around(point, min_r=1.0, max_r=2.5):
    """Convenience: centred regular polygon with random rotation/radius."""
    radius = np.random.uniform(min_r, max_r)
    sides  = np.random.choice([5, 6, 7, 8])
    poly   = _make_regular_polygon(point, radius, sides)
    angle  = np.random.uniform(0, 360)
    return rotate(poly, angle, origin=point)


# def gen_Linestring_Polygon_intersects(object_location, timestamp):
#     """
#     The LineString intersects (but is not fully contained in) the polygon.
#     """
#     ls = LineString(object_location)
#     # pick interior point and build polygon around it that is large enough so that
#     # the LS exits the polygon again
#     mid_idx = len(object_location)//2
#     centre  = tuple(object_location[mid_idx])
#     poly = _polygon_around(centre, min_r=0.5, max_r=1.0)

#     # interval = (timestamp[0], timestamp[-1])
#     # collect timestamps whose sample point intersects the polygon
#     intersect_times = [ts for pt, ts in zip(object_location, timestamp)
#                        if poly.intersects(Point(pt))]

#     # If no sample intersects, interval is empty (None); otherwise span the hits.
#     interval = (intersect_times[0], intersect_times[-1])
#     return ls, poly, interval

def gen_Linestring_Polygon_intersects(object_location,
                                   timestamp,
                                   base_min_r=0.30,   # initial polygon size
                                   base_max_r=0.60,
                                   shrink_factor=0.80, # shrink by this every retry
                                   max_tries=10):
    """
    Return
        ls       : LineString of the full trajectory (after any snapping)
        poly     : Polygon that the LS crosses exactly once (in→out)
        interval : (t_entry, t_exit)  — times when LS first enters / finally exits

    If a single continuous interval cannot be obtained after `max_tries`
    the function raises RuntimeError.
    """
    pts = np.asarray(object_location, dtype=float)
    ts  = np.asarray(timestamp)

    if pts.shape[0] < 2 or pts.shape[0] != ts.shape[0]:
        raise ValueError("object_location and timestamp must be the same length ≥ 2")

    for attempt in range(max_tries):
        # ---------------------------------------------------------------------
        # 1  Pick a random segment, grab a point on it, centre a polygon there
        # ---------------------------------------------------------------------
        p, pts, seg_idx = _point_on_segment(pts)          # <- uses your helper
        min_r = base_min_r * (shrink_factor ** attempt)
        max_r = base_max_r * (shrink_factor ** attempt)
        poly  = _polygon_around((p.x, p.y), min_r=min_r, max_r=max_r)

        # ---------------------------------------------------------------------
        # 2  Which segments actually cross that polygon?
        # ---------------------------------------------------------------------
        crossing_idcs = []
        for i in range(len(pts) - 1):
            seg = LineString([pts[i], pts[i + 1]])
            if seg.intersects(poly):
                crossing_idcs.append(i)

        if not crossing_idcs:
            # polygon misses the LS entirely – shrink & retry
            continue

        crossing_idcs.sort()
        contiguous = (crossing_idcs[-1] - crossing_idcs[0] + 1 == len(crossing_idcs))
        if not contiguous:
            # LS pops in–out–in; shrink & retry
            continue

        # ---------------------------------------------------------------------
        # 3  Build final result
        # ---------------------------------------------------------------------
        entry_idx = crossing_idcs[0]
        exit_idx  = crossing_idcs[-1] + 1          # last segment’s *end* point

        interval = (ts[entry_idx], ts[exit_idx])
        ls       = LineString(pts)
        # if len(crossing_idcs) > 1:
        #     pdb.set_trace()
        return ls, poly, interval

    raise RuntimeError("Unable to create a single continuous crossing interval "
                       f"after {max_tries} attempts; you might need to allow "
                       "more retries or adjust the size parameters.")


def gen_Linestring_Polygon_within(object_location, timestamp):
    """
    The LineString lies completely inside the polygon.
    """
    ls = LineString(object_location)

    # Build a generous polygon that encloses the whole trajectory
    bounds = ls.bounds        # (minx, miny, maxx, maxy)
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    max_half_extent = max(bounds[2]-bounds[0], bounds[3]-bounds[1]) / 2
    poly = _polygon_around((cx, cy),
                           min_r=max_half_extent*1.5,
                           max_r=max_half_extent*2.0)   # certainly contains

    # interval = (timestamp[0], timestamp[-1])
    # times for which the sample point is within the polygon
    within_times = [ts for pt, ts in zip(object_location, timestamp)
                    if poly.contains(Point(pt))]

    interval = (within_times[0], within_times[-1])
    return ls, poly, interval


# def gen_Linestring_Polygon_crosses(object_location, timestamp):
#     """
#     The LineString enters the polygon and exits again – interiors intersect.
#     """
#     ls = LineString(object_location)

#     # pick a single point roughly in the first third of the LS
#     p, _, i = _point_on_segment(object_location)

#     # a smallish polygon centred on that point → LS must cross it
#     poly = _polygon_around((p.x, p.y), min_r=0.3, max_r=0.6)

#     # collect times where sample points are inside or on polygon
#     cross_times = [ts for pt, ts in zip(object_location, timestamp)
#                    if poly.intersects(Point(pt))]

#     interval = (cross_times[0], cross_times[-1])
#     return ls, poly, interval

def gen_Linestring_Polygon_crosses(object_location,
                                   timestamp,
                                   base_min_r=0.30,   # initial polygon size
                                   base_max_r=0.60,
                                   shrink_factor=0.80, # shrink by this every retry
                                   max_tries=10):
    """
    Return
        ls       : LineString of the full trajectory (after any snapping)
        poly     : Polygon that the LS crosses exactly once (in→out)
        interval : (t_entry, t_exit)  — times when LS first enters / finally exits

    If a single continuous interval cannot be obtained after `max_tries`
    the function raises RuntimeError.
    """
    pts = np.asarray(object_location, dtype=float)
    ts  = np.asarray(timestamp)

    if pts.shape[0] < 2 or pts.shape[0] != ts.shape[0]:
        raise ValueError("object_location and timestamp must be the same length ≥ 2")

    for attempt in range(max_tries):
        # ---------------------------------------------------------------------
        # 1  Pick a random segment, grab a point on it, centre a polygon there
        # ---------------------------------------------------------------------
        p, pts, seg_idx = _point_on_segment(pts)          # <- uses your helper
        min_r = base_min_r * (shrink_factor ** attempt)
        max_r = base_max_r * (shrink_factor ** attempt)
        poly  = _polygon_around((p.x, p.y), min_r=min_r, max_r=max_r)

        # ---------------------------------------------------------------------
        # 2  Which segments actually cross that polygon?
        # ---------------------------------------------------------------------
        crossing_idcs = []
        for i in range(len(pts) - 1):
            seg = LineString([pts[i], pts[i + 1]])
            if seg.crosses(poly):
                crossing_idcs.append(i)

        if not crossing_idcs:
            # polygon misses the LS entirely – shrink & retry
            continue

        crossing_idcs.sort()
        contiguous = (crossing_idcs[-1] - crossing_idcs[0] + 1 == len(crossing_idcs))
        if not contiguous:
            # LS pops in–out–in; shrink & retry
            continue

        # ---------------------------------------------------------------------
        # 3  Build final result
        # ---------------------------------------------------------------------
        entry_idx = crossing_idcs[0]
        exit_idx  = crossing_idcs[-1] + 1          # last segment’s *end* point

        interval = (ts[entry_idx], ts[exit_idx])
        ls       = LineString(pts)
        # if len(crossing_idcs) > 1:
        #     pdb.set_trace()
        return ls, poly, interval

    raise RuntimeError("Unable to create a single continuous crossing interval "
                       f"after {max_tries} attempts; you might need to allow "
                       "more retries or adjust the size parameters.")


def gen_Linestring_Polygon_touches(object_location, timestamp):
    """
    The LineString only touches the polygon boundary at a single point.
    """
    ls = LineString(object_location)

    # take an interior point on LS and build polygon whose boundary runs exactly
    # through that point but interior stays on one side
    # Pick a vertex directly from the LineString
    i = np.random.randint(0, len(object_location))
    p = Point(object_location[i])

    # create small polygon slightly shifted so LS just grazes the boundary point
    poly_core = _polygon_around((p.x + 0.8, p.y + 0.8), min_r=0.5, max_r=0.8)
    # translate polygon so that one vertex coincides with p
    verts = list(poly_core.exterior.coords)
    shift = (p.x - verts[0][0], p.y - verts[0][1])
    poly = translate(poly_core, xoff=shift[0], yoff=shift[1])

    ls = snap(ls, poly, tolerance=1e-6)
    poly = snap(poly, ls, tolerance=1e-6)

    interval = (timestamp[i], timestamp[i])    # instantaneous touching
    return ls, poly, interval

# ------------------------------------------------------------------#
# --------------------   interval helpers   ------------------------#
# ------------------------------------------------------------------#

from allen_interval_algebra import precedes, is_preceded_by, meets, is_met_by,\
    overlaps_with, is_overlapped_by, starts, is_started_by,\
    during, contains, finishes, finished_by,\
    is_equal_to

def _random_interval(min_start=0.0, max_start=10.0,
                     min_len=1.0,  max_len=10.0):
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
    num_trial = 0
    while True:
        num_trial += 1
        interval2 = builder()
        truth     = satisfy_fn(interval1, interval2)
        if num_trial >= 1_000:
            raise ValueError("Trial exceed limits! Likely cannot satisfy the temporal condition.")
        if truth == want_positive:
            # final hygiene step → no negatives anywhere
            # interval1, interval2 = _shift_pair_non_negative(interval1, interval2)
            return interval1, interval2, int(truth)

# ------------------------------------------------------------------#
# --------------  generators for all 7 relations  -----------------#
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

def gen_meets(interval1=None):
    if interval1 is None: interval1 = _random_interval()

    def pos():
        len2 = np.random.uniform(1, 20)
        s2   = interval1[1]
        return (s2, s2 + len2)

    def neg(): return _random_interval()

    return _make_pair(interval1, meets, pos, neg)

def gen_overlaps_with(interval1=None):
    if interval1 is None: interval1 = _random_interval()

    def pos():
        s2   = np.random.uniform(interval1[0] + 0.1, interval1[1] - 0.1)
        len2 = np.random.uniform(interval1[1] - s2 + 0.1, 20)
        return (s2, s2 + len2)

    def neg(): return _random_interval()

    return _make_pair(interval1, overlaps_with, pos, neg)

def gen_starts(interval1=None):
    if interval1 is None: interval1 = _random_interval()

    def pos():
        len2 = np.random.uniform(interval1[1] - interval1[0] + 0.1, 20)
        return (interval1[0], interval1[0] + len2)

    def neg(): return _random_interval()

    return _make_pair(interval1, starts, pos, neg)

def gen_during(interval1=None):
    if interval1 is None: interval1 = _random_interval()

    def pos():
        len_outer = (interval1[1] - interval1[0]) * np.random.uniform(1.2, 3.0)
        left_pad  = np.random.uniform(0.1, len_outer - (interval1[1]-interval1[0]) - 0.1)
        s2        = max(interval1[0] - left_pad, 0)
        return (s2, s2 + len_outer)

    def neg(): return _random_interval()
    return _make_pair(interval1, during, pos, neg)

def gen_finishes(interval1=None):
    if interval1 is None: interval1 = _random_interval()

    def pos():
        extra = np.random.uniform(0.1, interval1[1] - interval1[0] - 0.1)
        return (max(interval1[0] - extra, 0), interval1[1])

    def neg(): return _random_interval()
    return _make_pair(interval1, finishes, pos, neg)

def gen_is_equal_to(interval1=None):
    if interval1 is None: interval1 = _random_interval()

    def pos(): return interval1
    def neg():
        while True:
            iv = _random_interval()
            if not is_equal_to(interval1, iv):
                return iv

    return _make_pair(interval1, is_equal_to, pos, neg)

# invalid_combination = [
#     ('Linestring_Polygon_touches', 'during'),
#     ('Linestring_Polygon_touches', 'overlaps_with'),
#     ('Linestring_Polygon_touches', 'finishes'),
#     ('Linestring_Linestring_touches', 'during'),
#     ('Linestring_Linestring_touches', 'overlaps_with'),
#     ('Linestring_Linestring_touches', 'finishes'),
# ]

invalid_combination = [
    ('Linestring_Polygon_touches', None), ('Linestring_Linestring_touches', None) # Touches can result in ambiguous definition of time interval. Removed.
]

def obtain_spatio_temporal_relationship():
    spatio_temporal_relationship = []
    for sr in sub_spatial_relationship:
        for tr in sub_temporal_relationship:
            if (sr, tr) in invalid_combination or (sr, None) in invalid_combination:
                continue
            else:
                spatio_temporal_relationship.append(sr + '-' + tr)
    return spatio_temporal_relationship

if __name__ == '__main__':
    reapeat = 100
    spatial_temporal_relationship = []
    for i, sr in tqdm(enumerate(sub_spatial_relationship)):
        for j, tr in enumerate(sub_temporal_relationship):
            if (sr, tr) in invalid_combination:
                continue
            for k in range(reapeat):
                num_trial = 1_000
                # for _ in range(num_trial):
                while True:
                    # print(sr, tr)
                    geo1_type, geo2_type, _spatial = sr.split('_')
                    obj_loc = np.random.uniform(0, 10, size=(10,2))
                    timestamp = np.linspace(0, 10, 10)

                    spatial_func = globals().get('gen_' + sr)
                    geo1, geo2, interval = spatial_func(obj_loc, timestamp)

                    if not getattr(geo1, _spatial)(geo2):
                        # pdb.set_trace()
                        continue

                    temporal_func = globals().get('gen_' + tr)
                    try:
                        i1, i2, answer = temporal_func(interval)
                    except:
                        pdb.set_trace()
                        continue

                    i1 = f'({i1[0]:.4f}, {i1[1]:.4f})'
                    i2 = f'({i2[0]:.4f}, {i2[1]:.4f})'

                    event_str = event.format(
                        geo1 = format_geometry_object(geo1, geo_type=geo1_type),
                        geo2 = format_geometry_object(geo2, geo_type=geo2_type),
                        spatial_relationship = _spatial
                    ) 
                    template_str = template[0].format(
                        temporal_relationship=tr,
                        event_1=event_str,
                        interval_2=i2
                    )
                    print(i,j, template_str, answer)
                    
                    # print(sr, tr)
                    # spatial_temporal_relationship.append(sr+'-'+tr)
                    break 
    # print(spatial_temporal_relationship)