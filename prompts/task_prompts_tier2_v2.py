context_spatial = """
Help me answer question regarding spatial relationship in a 2D plane:

Given Information:
    You will be provided with geometric information involving three types of 2D geometries—Point, LineString, and Polygon—all defined using the ESRI (Environmental Systems Research Institute) geometric format. These geometries are expressed as lists of coordinates in a Cartesian plane.

    Point: A single coordinate location in space, defined as a tuple: 
    
    [(x, y)].

    LineString: A sequence of points that forms a continuous line. It is represented as an ordered list of coordinate pairs: 
    
    [(x_1, y_1), (x_2, y2), ... (x_n, y_n)].

   Polygon: A closed shape formed by a sequence of coordinate pairs where the first and last points are the same to close the loop:
    
    [(x_1, y_1), (x_2, y2), ... (x_n, y_n), (x_1, y_1)].

    Spatial Relationships (based on ArcGIS model) is as follows. During computation, you should account for floating-point precision with a numerical tolerance 1e-8. For instance, 'Equals' should return True not just for exact mathematical identity, but if two geometries are identical within this tolerance.

    ArcGIS defines spatial relationships as logical conditions between geometric objects:

    1. Equals: Returns True if two geometries represent the same shape and location.
    2. Intersects: Returns True if the geometries share any portion of space, including edges or points.
    3. Contains: Returns True if one geometry completely encloses another.
    4. Within: The reverse of Contains. Returns True if the first geometry lies completely inside the second.
    5. Crosses: Returns True if the geometries intersect in a way where they share some interior points but are of different dimensions (e.g., a line crossing another line or a line crossing a polygon).
    6. Touches: Returns True if the geometries share only a boundary or a point but no interior space.
    7. Overlaps: Returns True if the geometries share some, but not all, interior points, and are of the same dimension.

"""

template_spatial = """
Objective:

Determine whether the {geo1} has the spatial relationship **{relate}** with the {geo2}?

Answer 1 if answer is Yes. Otherwise, answer 0.
"""

return_format = """

Output Format:
    Your response **must** include the answer (0 or 1) in the following format:

    [RESULTS_START] [p] [RESULTS_END]

    You may include explanatory text elsewhere in your response. However, do not include any text or additional formatting between [RESULTS_START] and [RESULTS_END].

Example:

    [RESULTS_START] [1] [RESULTS_END]
"""

context_temporal = """
Help me answer question regarding temporal relationship:

Given Information:
    You will be provided with intervals defined by (x_i, y_i).

    x_i and x_i are non negative numbers.

    Temporal Relationships (based on Allen's interval algebra):

	1.	Precedes (A precedes B): A ends before B starts.
	2.	Preceded-by (A is-preceded-by B): A starts after B ends.
	3.	Meets (A meets B): A ends exactly when B starts.
	4.	Met-By (A met-by B): A starts exactly when B ends.
	5.	Overlaps (A overlaps B): A starts before B starts, and ends after B starts but before B ends.
	6.	Overlapped-By (A overlapped-by B): A starts after B starts, A starts before B ends, and A ends after B ends.
	7.	Starts (A starts B): A and B start at the same time, but A ends before B ends.
	8.	Started-By (A started-by B): A and B start at the same time, but A ends after B ends.
	9.	During (A during B): A starts after B starts and ends before B ends.
	10.	Contains (A contains B): A starts before B starts and ends after B ends.
	11.	Finishes (A finishes B): A and B end at the same time, but A starts after B starts.
	12.	Finished-By (A finished-by B): A and B end at the same time, but A starts before B starts.
	13.	Equals (A equals B): A and B start and end at the same time.

"""

template_temporal = """
Objective:

Determine whether the time interval {geo1} has the temporal relationship **{relate}** with the time interval {geo2}?

Answer 1 if answer is Yes. Otherwise, answer 0.
"""

context_spatial_temporal = """
Help me answer question regarding spatial relationship in a 2D plane:

Given Information:

    You will receive a series of object trajectory and the corresponding timestamps of the coordinates in the trajectory. You can treat the trajectory as linestring.

    Sensor A: [(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)]

    Timestamp: [t1, t2, ..., tn]

    You will be provided with geometric information involving three types of 2D geometries—Point, LineString, and Polygon—all defined using the ESRI (Environmental Systems Research Institute) geometric format. These geometries are expressed as lists of coordinates in a Cartesian plane.

    Point: A single coordinate location in space, defined as a tuple: 
    
    [(x, y)].

    LineString: A sequence of points that forms a continuous line. It is represented as an ordered list of coordinate pairs: 
    
    [(x_1, y_1), (x_2, y2), ... (x_n, y_n)].

   Polygon: A closed shape formed by a sequence of coordinate pairs where the first and last points are the same to close the loop:
    
    [(x_1, y_1), (x_2, y2), ... (x_n, y_n), (x_1, y_1)].

    Spatial Relationships (based on ArcGIS model) are as follows. During computation, you should account for floating-point precision with a numerical tolerance 1e-8. For instance, 'Equals' should return True not just for exact mathematical identity, but if two geometries are identical within this tolerance.

    ArcGIS defines spatial relationships as logical conditions between geometric objects:

    1. Equals: Returns True if two geometries represent the same shape and location.
    2. Intersects: Returns True if the geometries share any portion of space, including edges or points.
    3. Contains: Returns True if one geometry completely encloses another.
    4. Within: The reverse of Contains. Returns True if the first geometry lies completely inside the second.
    5. Crosses: Returns True if the geometries intersect in a way where they share some interior points but are of different dimensions (e.g., a line crossing another line or a line crossing a polygon).
    6. Touches: Returns True if the geometries share only a boundary or a point but no interior space.
    7. Overlaps: Returns True if the geometries share some, but not all, interior points, and are of the same dimension.

    Temporal Relationships (based on Allen's interval algebra):

	1.	Precedes (A precedes B): A ends before B starts.
	2.	Preceded-by (A is-preceded-by B): A starts after B ends.
	3.	Meets (A meets B): A ends exactly when B starts.
	4.	Met-By (A met-by B): A starts exactly when B ends.
	5.	Overlaps (A overlaps B): A starts before B starts, and ends after B starts but before B ends.
	6.	Overlapped-By (A overlapped-by B): A starts after B starts, A starts before B ends, and A ends after B ends.
	7.	Starts (A starts B): A and B start at the same time, but A ends before B ends.
	8.	Started-By (A started-by B): A and B start at the same time, but A ends after B ends.
	9.	During (A during B): A starts after B starts and ends before B ends.
	10.	Contains (A contains B): A starts before B starts and ends after B ends.
	11.	Finishes (A finishes B): A and B end at the same time, but A starts after B starts.
	12.	Finished-By (A finished-by B): A and B end at the same time, but A starts before B starts.
	13.	Equals (A equals B): A and B start and end at the same time.
"""


st_event = 'the following object trajectory has the spatial relationship **{spatial_relationship}** with {geo2}'

template_st = '''
Objective:

Determine whether the time interval during which the EVENT holds has the temporal relationship **{temporal_relationship}** with the reference interval {interval_2}?
EVENT: {event_1}

For any interaction between a trajectory (you can view it as a LineString) and a fixed geometry—whether that geometry is another LineString, a Point, or a Polygon—define the “event interval” as follows:

1. Pick a trajectory segment and a predicate.
2. Project the segment endpoints back to their timestamps.  For each contiguous satisfying segment, let t_1 be the time at the segment’s first vertex and t_2 be the time at its last vertex.
3. Define the event interval as the union of all **[t_1, t_2]** intervals in which the predicate is true.
4. Do not include portions of the trajectory before the relationship begins or after it ends. Do not interpolate.

Answer 1 if answer is Yes. Otherwise, answer 0.

Object trajectory: {sensor_readings}
Timestamp: {timestr}
'''
