
import pdb 
# spatial_relationship = ['Point_Point_equals', 'Point_Point_intersects', 'Point_Point_contains', 'Point_Point_within', \
# 'Point_Linestring_intersects', 'Point_Linestring_within', 'Point_Linestring_touches', 'Point_Polygon_intersects', \
# 'Point_Polygon_within', 'Point_Polygon_touches', 'Linestring_Point_intersects', 'Linestring_Point_contains', \
# 'Linestring_Point_touches', 'Linestring_Linestring_equals', 'Linestring_Linestring_intersects', \
# 'Linestring_Linestring_contains', 'Linestring_Linestring_within', 'Linestring_Linestring_crosses', \
# 'Linestring_Linestring_touches', 'Linestring_Linestring_overlaps', 'Linestring_Polygon_intersects', \
# 'Linestring_Polygon_within', 'Linestring_Polygon_crosses', 'Linestring_Polygon_touches', 'Polygon_Point_intersects', \
# 'Polygon_Point_contains', 'Polygon_Point_touches', 'Polygon_Linestring_intersects', 'Polygon_Linestring_contains',\
# 'Polygon_Linestring_crosses', 'Polygon_Linestring_touches', 'Polygon_Polygon_equals', 'Polygon_Polygon_intersects', \
# 'Polygon_Polygon_contains', 'Polygon_Polygon_within', 'Polygon_Polygon_touches', 'Polygon_Polygon_overlaps']

spatial_relationship = ['Point_Point_equals', \
'Point_Linestring_intersects', 'Point_Linestring_within', 'Point_Linestring_touches', 'Point_Polygon_intersects', \
'Point_Polygon_within', 'Point_Polygon_touches', 'Linestring_Point_intersects', 'Linestring_Point_contains', \
'Linestring_Point_touches', 'Linestring_Linestring_equals', 'Linestring_Linestring_intersects', \
'Linestring_Linestring_contains', 'Linestring_Linestring_within', 'Linestring_Linestring_crosses', \
'Linestring_Linestring_touches', 'Linestring_Linestring_overlaps', 'Linestring_Polygon_intersects', \
'Linestring_Polygon_within', 'Linestring_Polygon_crosses', 'Linestring_Polygon_touches', 'Polygon_Point_intersects', \
'Polygon_Point_contains', 'Polygon_Point_touches', 'Polygon_Linestring_intersects', 'Polygon_Linestring_contains',\
'Polygon_Linestring_crosses', 'Polygon_Linestring_touches', 'Polygon_Polygon_equals', 'Polygon_Polygon_intersects', \
'Polygon_Polygon_contains', 'Polygon_Polygon_within', 'Polygon_Polygon_touches', 'Polygon_Polygon_overlaps']

"""
Balanced geometry–relationship generators based on an input (10, 2) numpy array
`object_location`.

For every relationship in

['Point_Point_equals',            'Point_Point_intersects',
 'Point_Point_contains',          'Point_Point_within',
 'Point_Linestring_intersects',   'Point_Linestring_within',
 'Point_Linestring_touches',      'Point_Polygon_intersects',
 'Point_Polygon_within',          'Point_Polygon_touches'],

a function named `gen_<relationship>(object_location)` returns

    geo1   – shapely geometry 1
    geo2   – shapely geometry 2
    label  – 1 if the relationship holds, 0 otherwise (balanced by construction)

Example
-------
>>> p1, p2, y = gen_Point_Point_equals(obj_locs)
>>> y, p1.equals(p2)
(1, True)
"""

import numpy as np
from shapely.geometry import Point, LineString, Polygon
from shapely.affinity import translate, scale
from shapely.ops import snap 

# -----------------------------------------------------------------------------#
# -------------  Helpers  -----------------------------------------------------#
# -----------------------------------------------------------------------------#

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

# def _point_on_segment(a, b, t=None):
#     """Return a point on the AB segment (not at the endpoints)."""
#     # if t is None:
#     #     t = np.random.uniform(0.15, 0.85)   # stay off the endpoints
#     # return Point(a[0] + t*(b[0]-a[0]), a[1] + t*(b[1]-a[1]))
#     ls = _random_linestring((a, b))
#     p = ls.interpolate(np.random.uniform(0.15, 0.85), normalized=True)
#     return p

def _point_on_segment(a, b, t=None):
    """Return a point on the AB segment (not at the endpoints), snapped to ensure topological intersection."""
    if t is None:
        t = np.random.uniform(0.15, 0.85)  # Avoid endpoints
    ls = LineString([a, b])
    p = ls.interpolate(t, normalized=True)
    # Snap point to line and line to point (mutual adjustment)
    p = snap(p, ls, tolerance=1e-8)
    ls = snap(ls, p, tolerance=1e-8)
    return p, ls

# -----------------------------------------------------------------------------#
# --------  Point / Point  ----------------------------------------------------#
# -----------------------------------------------------------------------------#

def gen_Point_Point_equals(object_location):
    """Balanced generator for Point–Point equals (same as intersects in 0‑D)."""
    if np.random.rand() < 0.5:                         # positive
        p = _random_point(object_location)
        return p, p, 1
    else:                                              # negative
        i, j = _two_distinct_indices(len(object_location))
        return Point(object_location[i]), Point(object_location[j]), 0


def gen_Point_Point_intersects(object_location):
    """Intersects is identical to equals for two points."""
    return gen_Point_Point_equals(object_location)     # same logic


def gen_Point_Point_contains(object_location):
    """
    For identical points shapely.contains returns True.
    (contains ⇔ equals for points).
    """
    return gen_Point_Point_equals(object_location)


def gen_Point_Point_within(object_location):
    """within is symmetric for identical points; use same generator."""
    return gen_Point_Point_equals(object_location)

# -----------------------------------------------------------------------------#
# --------  Point / LineString  ----------------------------------------------#
# -----------------------------------------------------------------------------#

def _random_linestring(loc):
    """Two‑point LineString with distinct vertices from loc array."""
    i, j = _two_distinct_indices(len(loc))
    return LineString([tuple(loc[i]), tuple(loc[j])])

def gen_Point_Linestring_intersects(object_location):
    if np.random.rand() < 0.5:                                    # positive
        ls = _random_linestring(object_location)
        # sample point on the segment (could be interior or endpoint)
        p, ls = _point_on_segment(*ls.coords)
        p = Point(p)
        # print("p:", p)
        # print("ls:", ls)
        # print("p.distance(ls):", p.distance(ls))   # should be 0
        # print("p.intersects(ls):", p.intersects(ls))  # should be True
        # pdb.set_trace()
        return p, ls, 1
    else:                                                         # negative
        p = _random_point(object_location)
        # Translate line far away to guarantee no intersection
        x_offset = np.random.uniform(-3,3)
        y_offset = np.random.uniform(-3,3)
        ls = translate(_random_linestring(object_location), xoff=x_offset, yoff=y_offset)
        return p, ls, 0

def gen_Point_Linestring_within(object_location):
    if np.random.rand() < 0.5:                                    # positive
        ls = _random_linestring(object_location)
        p, ls = _point_on_segment(*ls.coords)                         # interior
        return Point(p), ls, 1
    else:
        p = _random_point(object_location)                        # off‑segment
        ls = _random_linestring(object_location)
        return p, ls, 0

def gen_Point_Linestring_touches(object_location):
    if np.random.rand() < 0.5:                                    # positive
        ls = _random_linestring(object_location)
        p = Point(ls.coords[0])                                   # endpoint
        return p, ls, 1
    else:
        ls = _random_linestring(object_location)
        p, ls = _point_on_segment(*ls.coords)                         # interior
        return p, ls, 0

# -----------------------------------------------------------------------------#
# --------  Point / Polygon  --------------------------------------------------#
# -----------------------------------------------------------------------------#
from shapely.affinity import rotate

# def _make_small_square(centre, half=0.6):
#     """Square polygon centred at `centre` with edge length 2*half."""
#     x, y = centre
#     return Polygon([(x-half, y-half), (x+half, y-half),
#                     (x+half, y+half), (x-half, y+half)])

# def _random_polygon(loc):
#     """Axis‑aligned square derived from a random anchor in loc."""
#     centre = loc[np.random.randint(len(loc))]
#     half = np.random.uniform(0.5, 1.5)
#     return _make_small_square(centre, half)

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

def gen_Point_Polygon_intersects(object_location):
    if np.random.rand() < 0.5:                                    # positive
        poly = _random_polygon(object_location)
        # sample interior point
        p = Point(poly.centroid.x, poly.centroid.y)
        return p, poly, 1
    else:
        poly = _random_polygon(object_location)
        p = translate(poly.centroid, xoff=2*poly.bounds[2])       # outside
        return p, poly, 0

def gen_Point_Polygon_within(object_location):
    if np.random.rand() < 0.5:                                    # positive
        poly = _random_polygon(object_location)
        p = Point(poly.centroid.x, poly.centroid.y)
        return p, poly, 1
    else:
        poly = _random_polygon(object_location)
        p = translate(poly.centroid, xoff=2*poly.bounds[2])
        return p, poly, 0

def gen_Point_Polygon_touches(object_location):
    if np.random.rand() < 0.5:                                    # positive
        poly = _random_polygon(object_location)
        # choose a vertex to guarantee boundary point
        p = Point(poly.exterior.coords[0])
        return p, poly, 1
    else:
        poly = _random_polygon(object_location)
        p = Point(poly.centroid.x, poly.centroid.y)               # interior
        return p, poly, 0

# -----------------------------------------------------------------------------#
# --------  Convenience registry  --------------------------------------------#
# -----------------------------------------------------------------------------#

GENERATOR_MAP = {
    'Point_Point_equals'            : gen_Point_Point_equals,
    'Point_Point_intersects'        : gen_Point_Point_intersects,
    'Point_Point_contains'          : gen_Point_Point_contains,
    'Point_Point_within'            : gen_Point_Point_within,
    'Point_Linestring_intersects'   : gen_Point_Linestring_intersects,
    'Point_Linestring_within'       : gen_Point_Linestring_within,
    'Point_Linestring_touches'      : gen_Point_Linestring_touches,
    'Point_Polygon_intersects'      : gen_Point_Polygon_intersects,
    'Point_Polygon_within'          : gen_Point_Polygon_within,
    'Point_Polygon_touches'         : gen_Point_Polygon_touches,
}

# -----------------------------------------------------------------------------#
# --------  Example usage  ----------------------------------------------------#
# -----------------------------------------------------------------------------#

# -----------------------------------------------------------------------------#
# --------  LineString / Point  ---------------------------------------------- #
# -----------------------------------------------------------------------------#

def gen_Linestring_Point_intersects(object_location):
    if np.random.rand() < 0.5:                                  # positive
        ls = _random_linestring(object_location)
        # p  = ls.interpolate(np.random.uniform(0.05, 0.95), normalized=True)
        p, ls = _point_on_segment(*ls.coords)
        return ls, Point(p), 1
    else:                                                       # negative
        ls = _random_linestring(object_location)
        p  = _random_point(object_location)                     # likely off the line
        # push the point well away from the line
        p  = translate(p, xoff=np.random.uniform(3, 6),
                          yoff=np.random.uniform(3, 6))
        return ls, Point(p), 0


def gen_Linestring_Point_contains(object_location):
    if np.random.rand() < 0.5:                                  # positive
        ls = _random_linestring(object_location)
        # p  = ls.interpolate(np.random.uniform(0.15, 0.85), normalized=True)  # interior
        p, ls = _point_on_segment(*ls.coords)
        return ls, Point(p), 1
    else:                                                       # negative
        ls = _random_linestring(object_location)
        p  = _random_point(object_location)
        # nudge far away to avoid coincidence with endpoints or interior
        p  = translate(p, xoff=np.random.uniform(3, 6),
                          yoff=np.random.uniform(3, 6))
        return ls, Point(p), 0


def gen_Linestring_Point_touches(object_location):
    if np.random.rand() < 0.5:                                  # positive
        ls = _random_linestring(object_location)
        p  = Point(ls.coords[0])                                # endpoint ‑‑ boundary
        return ls, p, 1
    else:                                                       # negative
        ls = _random_linestring(object_location)
        # p  = ls.interpolate(np.random.uniform(0.15, 0.85), normalized=True)  # interior
        p, ls = _point_on_segment(*ls.coords)
        return ls, Point(p), 0


# -----------------------------------------------------------------------------#
# --------  LineString / LineString  ----------------------------------------- #
# -----------------------------------------------------------------------------#

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

def gen_Linestring_Linestring_equals(object_location):
    if np.random.rand() < 0.5:                                  # positive
        ls = _random_linestring(object_location)
        return ls, LineString(ls.coords), 1                     # identical copy
    else:                                                       # negative
        ls1 = _random_linestring(object_location)
        ls2 = translate(_random_linestring(object_location),
                        xoff=np.random.uniform(3, 6),
                        yoff=np.random.uniform(3, 6))
        return ls1, ls2, 0


def gen_Linestring_Linestring_intersects(object_location):
    if np.random.rand() < 0.5:                                  # positive
        centre = _random_point(object_location)
        ls1, ls2 = _orthogonal_cross((centre.x, centre.y))
        return ls1, ls2, 1
    else:  
        while True:                                               # negative
            ls1 = _random_linestring(object_location)
            ls2 = translate(_random_linestring(object_location),
                            xoff=np.random.uniform(3, 6),
                            yoff=np.random.uniform(3, 6))
            if not ls1.intersects(ls2):
                break 
        return ls1, ls2, 0


def gen_Linestring_Linestring_contains(object_location):
    if np.random.rand() < 0.5:                                  # positive
        ls_big = _random_linestring(object_location)
        ls_small, ls_big = _subsegment_of(ls_big)                       # proper subset
        return ls_big, ls_small, 1
    else:                                                       # negative
        while True:
            ls1 = _random_linestring(object_location)
            ls2 = translate(_random_linestring(object_location),
                            xoff=np.random.uniform(3, 6),
                            yoff=np.random.uniform(3, 6))
            if not ls1.contains(ls2):
                break 
        return ls1, ls2, 0


def gen_Linestring_Linestring_within(object_location):
    # simply reverse "contains"
    ls_container, ls_inside, label = gen_Linestring_Linestring_contains(object_location)
    return ls_inside, ls_container, label


def gen_Linestring_Linestring_crosses(object_location):
    if np.random.rand() < 0.5:                                  # positive
        centre = _random_point(object_location)
        ls1, ls2 = _orthogonal_cross((centre.x, centre.y))
        return ls1, ls2, 1
    else:                                                       # negative – no intersection
        while True:
            ls1 = _random_linestring(object_location)
            ls2 = translate(_random_linestring(object_location),
                            xoff=np.random.uniform(3, 6),
                            yoff=np.random.uniform(3, 6))
            if not ls1.crosses(ls2):
                break 
        return ls1, ls2, 0


# -----------------------------------------------------------------------------#
# --------  Register the new generators  ------------------------------------- #
# -----------------------------------------------------------------------------#

GENERATOR_MAP.update({
    'Linestring_Point_intersects'      : gen_Linestring_Point_intersects,
    'Linestring_Point_contains'        : gen_Linestring_Point_contains,
    'Linestring_Point_touches'         : gen_Linestring_Point_touches,
    'Linestring_Linestring_equals'     : gen_Linestring_Linestring_equals,
    'Linestring_Linestring_intersects' : gen_Linestring_Linestring_intersects,
    'Linestring_Linestring_contains'   : gen_Linestring_Linestring_contains,
    'Linestring_Linestring_within'     : gen_Linestring_Linestring_within,
    'Linestring_Linestring_crosses'    : gen_Linestring_Linestring_crosses,
})


# -----------------------------------------------------------------------------#
#  New LineString / LineString relations                                       #
# -----------------------------------------------------------------------------#

def gen_Linestring_Linestring_touches(object_location):
    """Generate two line strings that touch at a single end‑point (positive) or
    are well separated (negative)."""
    if np.random.rand() < 0.5:  # positive – touch at one end‑point
        ls1 = _random_linestring(object_location)
        p0 = Point(ls1.coords[0])
        # build ls2 from the shared end‑point in a (roughly) orthogonal direction
        dx, dy = ls1.coords[1][0] - ls1.coords[0][0], ls1.coords[1][1] - ls1.coords[0][1]
        if abs(dx) + abs(dy) < 1e-8:  # degenerate safeguard
            dx, dy = 1.0, 0.0
        # rotate 90° to avoid collinearity (and hence overlap)
        new_dir = (-dy, dx)
        norm = np.hypot(*new_dir)
        new_dir = (new_dir[0] / norm, new_dir[1] / norm)
        length = np.random.uniform(1.0, 2.5)
        p1 = (p0.x + new_dir[0] * length, p0.y + new_dir[1] * length)
        ls2 = LineString([p0, p1])
        return ls1, ls2, 1
    else:  # negative – no touching
        ls1 = _random_linestring(object_location)
        ls2 = translate(_random_linestring(object_location),
                         xoff=np.random.uniform(3, 6),
                         yoff=np.random.uniform(3, 6))
        return ls1, ls2, 0


def gen_Linestring_Linestring_overlaps(object_location):
    """Generate partially overlapping collinear line strings (positive) or
    unrelated ones (negative)."""
    if np.random.rand() < 0.5:  # positive – partial overlap
        # create a base line
        ls_base = _random_linestring(object_location)
        a, b = map(Point, ls_base.coords)
        vec = (b.x - a.x, b.y - a.y)
        # parameter points along the line (0 < t1 < t2 < 1 < t3)
        t1 = np.random.uniform(0.2, 0.4)
        t2 = np.random.uniform(0.6, 0.8)
        t3 = np.random.uniform(1.1, 1.5)
        p_t1 = (a.x + vec[0] * t1, a.y + vec[1] * t1)
        p_t2 = (a.x + vec[0] * t2, a.y + vec[1] * t2)
        p_t3 = (a.x + vec[0] * t3, a.y + vec[1] * t3)
        ls1 = LineString([a, p_t2])  # long enough to contain overlap segment
        ls2 = LineString([p_t1, p_t3])
        # The overlapping segment is between p_t1 – p_t2; neither contains the other
        # snap for alignment
        ls1 = snap(ls1, ls2, tolerance=1e-8)
        ls2 = snap(ls2, ls1, tolerance=1e-8)
        return ls1, ls2, 1
    else:  # negative – clearly separate
        ls1 = _random_linestring(object_location)
        ls2 = translate(_random_linestring(object_location),
                         xoff=np.random.uniform(3, 6),
                         yoff=np.random.uniform(3, 6))
        return ls1, ls2, 0

# -----------------------------------------------------------------------------#
#  LineString / Polygon relations                                              #
# -----------------------------------------------------------------------------#

def gen_Linestring_Polygon_intersects(object_location):
    if np.random.rand() < 0.5:  # positive – intersects
        poly = _random_polygon(object_location)
        cx, cy = poly.centroid.x, poly.centroid.y
        minx, miny, maxx, maxy = poly.bounds
        # horizontal line through the centroid that clearly passes through poly
        ls = LineString([(minx - 1.0, cy), (maxx + 1.0, cy)])
        return ls, poly, 1
    else:  # negative – separate
        poly = _random_polygon(object_location)
        while True:
            ls = translate(_random_linestring(object_location),
                            xoff=np.random.uniform(3, 6),
                            yoff=np.random.uniform(3, 6))
            if not ls.intersects(poly):
                break
        return ls, poly, 0


def gen_Linestring_Polygon_within(object_location):
    if np.random.rand() < 0.5:  # positive – line inside polygon
        poly = _random_polygon(object_location)
        cx, cy = poly.centroid.x, poly.centroid.y
        # choose small vector to make a tiny line inside
        angle = np.random.uniform(0, 2 * np.pi)
        delta = 0.3
        p1 = (cx + delta * np.cos(angle), cy + delta * np.sin(angle))
        p2 = (cx - delta * np.cos(angle), cy - delta * np.sin(angle))
        ls = LineString([p1, p2])
        return ls, poly, 1
    else:  # negative – outside
        poly = _random_polygon(object_location)
        ls = translate(_random_linestring(object_location),
                        xoff=np.random.uniform(3, 6),
                        yoff=np.random.uniform(3, 6))
        return ls, poly, 0


def gen_Linestring_Polygon_crosses(object_location):
    if np.random.rand() < 0.5:  # positive – crosses (enters & exits)
        poly = _random_polygon(object_location)
        minx, miny, maxx, maxy = poly.bounds
        # vertical line through centroid
        cx = poly.centroid.x
        ls = LineString([(cx, miny - 1.0), (cx, maxy + 1.0)])
        return ls, poly, 1
    else:  # negative – no crossing
        while True:
            poly = _random_polygon(object_location)
            ls = translate(_random_linestring(object_location),
                            xoff=np.random.uniform(3, 6),
                            yoff=np.random.uniform(3, 6))
            if not ls.crosses(poly):
                break
        return ls, poly, 0


def gen_Linestring_Polygon_touches(object_location):
    if np.random.rand() < 0.5:  # positive – touches boundary only
        poly = _random_polygon(object_location)
        # take one polygon edge and make a tiny line along it
        x0, y0 = poly.exterior.coords[0]
        x1, y1 = poly.exterior.coords[1]
        # shorten the edge so it's not identical to the whole edge
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        ls = LineString([(x0, y0), (mid_x, mid_y)])
        ls = snap(ls, poly, tolerance=1e-6)
        poly = snap(poly, ls, tolerance=1e-6)
        return ls, poly, 1
    else:  # negative – clearly apart
        poly = _random_polygon(object_location)
        ls = translate(_random_linestring(object_location),
                        xoff=np.random.uniform(3, 6),
                        yoff=np.random.uniform(3, 6))
        return ls, poly, 0

# -----------------------------------------------------------------------------#
#  Polygon / Point relations (wrappers around existing generators)             #
# -----------------------------------------------------------------------------#

def gen_Polygon_Point_intersects(object_location):
    p, poly, label = gen_Point_Polygon_intersects(object_location)
    return poly, p, label


def gen_Polygon_Point_contains(object_location):
    p, poly, label = gen_Point_Polygon_within(object_location)
    return poly, p, label


def gen_Polygon_Point_touches(object_location):
    p, poly, label = gen_Point_Polygon_touches(object_location)
    return poly, p, label

# -----------------------------------------------------------------------------#
#  Polygon / LineString relations (wrappers)                                   #
# -----------------------------------------------------------------------------#

def gen_Polygon_Linestring_intersects(object_location):
    ls, poly, label = gen_Linestring_Polygon_intersects(object_location)
    return poly, ls, label


def gen_Polygon_Linestring_contains(object_location):
    ls, poly, label = gen_Linestring_Polygon_within(object_location)
    # swap roles – contains is the inverse of within
    return poly, ls, label

# -----------------------------------------------------------------------------#
#  Registry update                                                             #
# -----------------------------------------------------------------------------#

GENERATOR_MAP.update({
    'Linestring_Linestring_touches'      : gen_Linestring_Linestring_touches,
    'Linestring_Linestring_overlaps'     : gen_Linestring_Linestring_overlaps,
    'Linestring_Polygon_intersects'      : gen_Linestring_Polygon_intersects,
    'Linestring_Polygon_within'          : gen_Linestring_Polygon_within,
    'Linestring_Polygon_crosses'         : gen_Linestring_Polygon_crosses,
    'Linestring_Polygon_touches'         : gen_Linestring_Polygon_touches,
    'Polygon_Point_intersects'           : gen_Polygon_Point_intersects,
    'Polygon_Point_contains'             : gen_Polygon_Point_contains,
    'Polygon_Point_touches'              : gen_Polygon_Point_touches,
    'Polygon_Linestring_intersects'      : gen_Polygon_Linestring_intersects,
    'Polygon_Linestring_contains'        : gen_Polygon_Linestring_contains,
})

# -----------------------------------------------------------------------------#
#  Polygon / LineString wrappers                                               #
# -----------------------------------------------------------------------------#

def gen_Polygon_Linestring_crosses(object_location):
    """Wrapper around the existing LS‑poly generator (role reversal)."""
    ls, poly, label = gen_Linestring_Polygon_crosses(object_location)
    return poly, ls, label


def gen_Polygon_Linestring_touches(object_location):
    """Wrapper around the existing LS‑poly generator (role reversal)."""
    ls, poly, label = gen_Linestring_Polygon_touches(object_location)
    return poly, ls, label


# -----------------------------------------------------------------------------#
#  Polygon / Polygon helpers                                                   #
# -----------------------------------------------------------------------------#

def _inner_polygon(poly, shrink=0.4):
    """Return a scaled‑down copy of `poly` about its centroid."""
    # shrink < 1 keeps it strictly inside
    return scale(poly, xfact=shrink, yfact=shrink, origin='center')


def _side_by_side(poly, gap=0.0):
    """
    Return a second axis‑aligned square that is translated purely in +x so that:
      * gap  = 0.0  → polygons touch (share an edge)
      * gap  > 0.0  → polygons are disjoint
      * gap  < 0.0  → polygons overlap
    """
    minx, miny, maxx, maxy = poly.bounds
    dx = (maxx - minx) + gap
    return translate(poly, xoff=dx, yoff=0.0)


# def _side_by_side(poly, gap=0.0):
#     """
#     Return a second polygon translated in +x so that:
#       * gap  = 0.0  → polygons touch (bounding boxes are adjacent)
#       * gap  > 0.0  → polygons are disjoint
#       * gap  < 0.0  → polygons overlap
#     """
#     minx, _, maxx, _ = poly.bounds
#     width = maxx - minx
#     dx = width + gap
#     return translate(poly, xoff=dx, yoff=0.0)

# -----------------------------------------------------------------------------#
#  Polygon / Polygon generators                                                #
# -----------------------------------------------------------------------------#

def gen_Polygon_Polygon_equals(object_location):
    if np.random.rand() < 0.5:                       # positive
        poly = _random_polygon(object_location)
        return poly, Polygon(poly.exterior.coords), 1
    else:                                            # negative
        poly1 = _random_polygon(object_location)
        poly2 = translate(_random_polygon(object_location),
                          xoff=np.random.uniform(3, 6),
                          yoff=np.random.uniform(3, 6))
        return poly1, poly2, 0

def gen_Polygon_Polygon_intersects(object_location):
    if np.random.rand() < 0.5:  # positive – intersect (partial overlap or containment)
        while True:
            poly1 = _random_polygon(object_location)
            poly2 = _random_polygon(object_location)

            # shrink and translate poly2 to fit inside poly1
            poly2_scaled = scale(poly2, xfact=0.4, yfact=0.4, origin='center')
            dx = np.random.uniform(-0.5, 0.5)
            dy = np.random.uniform(-0.5, 0.5)
            poly2_translated = translate(poly2_scaled, xoff=dx, yoff=dy)

            if poly1.intersects(poly2_translated) and not poly1.equals(poly2_translated):
                break
        return poly1, poly2_translated, 1

    else:  # negative – no intersection
        while True:
            poly1 = _random_polygon(object_location)
            poly2 = translate(_random_polygon(object_location),
                              xoff=np.random.uniform(3, 6),
                              yoff=np.random.uniform(3, 6))
            if not poly1.intersects(poly2):
                break
        return poly1, poly2, 0


def gen_Polygon_Polygon_contains(object_location):
    if np.random.rand() < 0.5:                       # positive
        outer = _random_polygon(object_location)
        inner = _inner_polygon(outer, shrink=0.4)
        return outer, inner, 1
    else:                                            # negative
        outer = _random_polygon(object_location)
        # put another polygon somewhere else (not contained)
        inner = translate(_random_polygon(object_location),
                          xoff=np.random.uniform(3, 6),
                          yoff=np.random.uniform(3, 6))
        return outer, inner, 0


def gen_Polygon_Polygon_within(object_location):
    # simply reverse “contains”
    outer, inner, label = gen_Polygon_Polygon_contains(object_location)
    return inner, outer, label

def gen_Polygon_Polygon_touches(object_location):
    if np.random.rand() < 0.5:  # positive – share a vertex or edge
        while True:
            poly1 = _random_polygon(object_location, sides=np.random.randint(4, 6))
            coords1 = list(poly1.exterior.coords)[:-1]  # exclude closing coord
            shared_vertex = coords1[np.random.randint(len(coords1))]
            
            poly2 = _random_polygon(object_location, sides=np.random.randint(4, 6))
            coords2 = list(poly2.exterior.coords)[:-1]
            v2 = coords2[0]  # arbitrary vertex to align
            
            # Compute translation vector to align v2 to shared_vertex
            dx = shared_vertex[0] - v2[0]
            dy = shared_vertex[1] - v2[1]
            poly2_aligned = translate(poly2, xoff=dx, yoff=dy)
            
            if poly1.touches(poly2_aligned):
                break
        return poly1, poly2_aligned, 1
    else:  # negative – clearly apart
        while True:
            poly1 = _random_polygon(object_location)
            poly2 = translate(_random_polygon(object_location),
                              xoff=np.random.uniform(3, 6),
                              yoff=np.random.uniform(3, 6))
            if not poly1.touches(poly2):
                break
        return poly1, poly2, 0


def gen_Polygon_Polygon_overlaps(object_location):
    """
    Positive: axis‑aligned squares that overlap but neither contains the other
    (area of intersection < each individual area).
    """
    if np.random.rand() < 0.5:                       # positive – partial overlap
        while True:
            poly1 = _random_polygon(object_location)
            poly2 = _side_by_side(poly1, gap=-0.3)       # small negative gap = overlap
            # make sure neither contains the other (gap not too negative)
            if poly1.overlaps(poly2):
                break
        return poly1, poly2, 1
    else:                                            # negative – disjoint or nested
        while True:
            poly1 = _random_polygon(object_location)
            if np.random.rand() < 0.5:               # disjoint
                poly2 = translate(_random_polygon(object_location),
                                  xoff=np.random.uniform(3, 6),
                                  yoff=np.random.uniform(3, 6))
            else:                                    # nested
                poly2 = _inner_polygon(poly1, shrink=0.4)
            if not poly1.overlaps(poly2):
                break
        return poly1, poly2, 0


# -----------------------------------------------------------------------------#
#  Registry update                                                             #
# -----------------------------------------------------------------------------#

GENERATOR_MAP.update({
    'Polygon_Linestring_crosses'   : gen_Polygon_Linestring_crosses,
    'Polygon_Linestring_touches'   : gen_Polygon_Linestring_touches,
    'Polygon_Polygon_equals'       : gen_Polygon_Polygon_equals,
    'Polygon_Polygon_intersects'   : gen_Polygon_Polygon_intersects,
    'Polygon_Polygon_contains'     : gen_Polygon_Polygon_contains,
    'Polygon_Polygon_within'       : gen_Polygon_Polygon_within,
    'Polygon_Polygon_touches'      : gen_Polygon_Polygon_touches,
    'Polygon_Polygon_overlaps'     : gen_Polygon_Polygon_overlaps,
})

if __name__ == "__main__":
    # obj_loc = np.random.uniform(0, 10, size=(10,2))
    # name = 'Point_Linestring_touches'
    # # name = spatial_relationship[i]
    # g1, g2, y = GENERATOR_MAP[name](obj_loc)
    
    # print(name, y, getattr(g1, name.split('_')[-1])(g2)) 

    # demo with random 10×2 cloud
    for i in range(len(spatial_relationship)):
        for j in range(20):
            obj_loc = np.random.uniform(0, 10, size=(10,2))

            # name = 'Point_Polygon_touches'
            # name = 'Point_Linestring_touches'
            # name = 'Polygon_Polygon_touches'
            name = spatial_relationship[i]
            g1, g2, y = GENERATOR_MAP[name](obj_loc)
            sanity_check = getattr(g1, name.split('_')[-1])(g2)
            print(name, y, sanity_check)  # sanity check
            if y != sanity_check:
                # print(f"**** Wrong: {list(g1.coords)} {list(g2.coords)} {name}")
                print(f"**** Wrong: {g1} {g2} {name}")
                pdb.set_trace()
        # print(g1, g2)
