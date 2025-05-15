# Allen's Interval Algebra Relations (13 Functions)

# --- Helper Validation Function ---
def _validate_interval(interval):
    """Checks if an interval tuple is valid (start <= end)."""
    start, end = interval
    if start > end:
        raise ValueError(f"Invalid interval {interval}: start cannot be greater than end.")

# --- Base Relations (and their inverses) ---

def precedes(interval1, interval2):
  """
  Relation: interval1 < interval2
  Checks if interval1 is entirely before interval2.
  interval1: (start1, end1)
  interval2: (start2, end2)
  Returns True if end1 < start2, False otherwise.
  """
  _validate_interval(interval1)
  _validate_interval(interval2)
  start1, end1 = interval1
  start2, end2 = interval2
  return end1 < start2

def is_preceded_by(interval1, interval2):
  """
  Relation: interval1 > interval2 (Inverse of precedes)
  Checks if interval1 is entirely after interval2.
  interval1: (start1, end1)
  interval2: (start2, end2)
  Returns True if start1 > end2, False otherwise.
  """
  # Equivalent to precedes(interval2, interval1)
  return precedes(interval2, interval1)

def meets(interval1, interval2):
  """
  Relation: interval1 m interval2
  Checks if interval1 meets interval2 (ends exactly when interval2 starts).
  interval1: (start1, end1)
  interval2: (start2, end2)
  Returns True if end1 == start2, False otherwise.
  """
  _validate_interval(interval1)
  _validate_interval(interval2)
  start1, end1 = interval1
  start2, end2 = interval2
  return end1 == start2

def is_met_by(interval1, interval2):
  """
  Relation: interval1 mi interval2 (Inverse of meets)
  Checks if interval1 is met by interval2 (starts exactly when interval2 ends).
  interval1: (start1, end1)
  interval2: (start2, end2)
  Returns True if start1 == end2, False otherwise.
  """
  # Equivalent to meets(interval2, interval1)
  return meets(interval2, interval1)

def overlaps_with(interval1, interval2):
  """
  Relation: interval1 o interval2
  Checks if interval1 overlaps with interval2 (starts before, ends during).
  interval1: (start1, end1)
  interval2: (start2, end2)
  Returns True if start1 < start2 < end1 < end2, False otherwise.
  """
  _validate_interval(interval1)
  _validate_interval(interval2)
  start1, end1 = interval1
  start2, end2 = interval2
  return start1 < start2 and start2 < end1 and end1 < end2

def is_overlapped_by(interval1, interval2):
  """
  Relation: interval1 oi interval2 (Inverse of overlaps_with)
  Checks if interval1 is overlapped by interval2 (starts during, ends after).
  interval1: (start1, end1)
  interval2: (start2, end2)
  Returns True if start2 < start1 < end2 < end1, False otherwise.
  """
  # Equivalent to overlaps_with(interval2, interval1)
  return overlaps_with(interval2, interval1)

def starts(interval1, interval2):
  """
  Relation: interval1 s interval2
  Checks if interval1 starts interval2 (same start, interval1 ends earlier).
  interval1: (start1, end1)
  interval2: (start2, end2)
  Returns True if start1 == start2 and end1 < end2, False otherwise.
  """
  _validate_interval(interval1)
  _validate_interval(interval2)
  start1, end1 = interval1
  start2, end2 = interval2
  return start1 == start2 and end1 < end2

def is_started_by(interval1, interval2):
  """
  Relation: interval1 si interval2 (Inverse of starts)
  Checks if interval1 is started by interval2 (same start, interval1 ends later).
  interval1: (start1, end1)
  interval2: (start2, end2)
  Returns True if start1 == start2 and end1 > end2, False otherwise.
  """
  # Equivalent to starts(interval2, interval1)
  return starts(interval2, interval1)

def during(interval1, interval2):
  """
  Relation: interval1 d interval2
  Checks if interval1 is during interval2 (contained within, but not equal).
  interval1: (start1, end1)
  interval2: (start2, end2)
  Returns True if start2 < start1 and end1 < end2, False otherwise.
  """
  _validate_interval(interval1)
  _validate_interval(interval2)
  start1, end1 = interval1
  start2, end2 = interval2
  return start2 < start1 and end1 < end2

def contains(interval1, interval2):
  """
  Relation: interval1 di interval2 (Inverse of during)
  Checks if interval1 contains interval2 (but they are not equal).
  interval1: (start1, end1)
  interval2: (start2, end2)
  Returns True if start1 < start2 and end2 < end1, False otherwise.
  """
  # Equivalent to during(interval2, interval1)
  return during(interval2, interval1)

def finishes(interval1, interval2):
  """
  Relation: interval1 f interval2
  Checks if interval1 finishes interval2 (same end, interval1 starts later).
  interval1: (start1, end1)
  interval2: (start2, end2)
  Returns True if end1 == end2 and start2 < start1, False otherwise.
  """
  _validate_interval(interval1)
  _validate_interval(interval2)
  start1, end1 = interval1
  start2, end2 = interval2
  return end1 == end2 and start2 < start1

def finished_by(interval1, interval2):
  """
  Relation: interval1 fi interval2 (Inverse of finishes)
  Checks if interval1 is finished by interval2 (same end, interval1 starts earlier).
  interval1: (start1, end1)
  interval2: (start2, end2)
  Returns True if end1 == end2 and start1 < start2, False otherwise.
  """
  # Equivalent to finishes(interval2, interval1)
  return finishes(interval2, interval1)

def is_equal_to(interval1, interval2):
  """
  Relation: interval1 = interval2
  Checks if interval1 is equal to interval2 (same start and end).
  interval1: (start1, end1)
  interval2: (start2, end2)
  Returns True if start1 == start2 and end1 == end2, False otherwise.
  """
  _validate_interval(interval1)
  _validate_interval(interval2)
  start1, end1 = interval1
  start2, end2 = interval2
  return start1 == start2 and end1 == end2