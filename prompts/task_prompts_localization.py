
Forecasting = """
Predict the next {} readings for all {}. Output your results using the following template:
[RESULTS_START]
[[...],[...], ..., [...]]
[RESULTS_END]
Each inside list is a temporal prediction of a location. The outer list is a collection of predictions from different locations. Do not put text between the keywords [RESULTS_START] and [RESULTS_END].
"""

Tracking = """
You need to estimate the location of the object at current time using the following template:

[RESULTS_START]
[x, y, z]
[RESULTS_END]

Do not put text between the keywords [RESULTS_START] and [RESULTS_END].
"""

loc_range_prompt = """
Help me determine the location of an object in a 2D plane based on sensor measurements.

Return the location of the object in the following format:

[RESULTS_START] [x, y] [RESULTS_END]

Given Information:
    There are four sensors, each providing a range-only (distance) measurement to the object. These measurements indicate the distance from the object to the sensors.
	The sensor locations in Cartesian coordinates ([x, y]) are:
	    {sensor_loc_str}
	The sensor readings are: {X_str}.
	    - The reading represents the distance from the object to each sensor.
	    - The measurements may contain noise.

Question:

Based on the given data, where is the object most likely located? Provide the estimated [x, y] coordinates.

Do not put text between the keywords [RESULTS_START] and [RESULTS_END].
"""

loc_bearing_prompt = """
Help me determine the location of an object in a 2D plane based on sensor measurements.

Return the location of the object in the following format:

[RESULTS_START] [x, y] [RESULTS_END]

Given Information:
There are four sensors, each providing a bearing-only (angle) measurement to the object. These measurements indicate the direction of the object relative to each sensor but do not provide distance.
The sensor locations in Cartesian coordinates ([x, y]) are:
{sensor_loc_str}
The sensor readings are: {X_str}.
- Units: degrees, within the range [0, 360).
- The bearing angle for each sensor is measured from the sensor’s own position (treated as the origin for its measurement).
- The reference direction is the positive x-axis (a horizontal ray extending to the right from the sensor’s location).
- Angles increase counterclockwise (CCW) from this reference direction.
- The measurements may contain noise.

Question:

Based on the given angles, where is the object most likely located? Provide the estimated [x, y] coordinates.

Do not put text between the keywords [RESULTS_START] and [RESULTS_END].
"""

loc_range_bearing_prompt = """
Help me determine the location of an object in a 2D plane based on sensor measurements.

Return the location of the object in the following format:

[RESULTS_START] [x, y] [RESULTS_END]

Do not include any additional text between [RESULTS_START] and [RESULTS_END].

Given Information:
    There are four sensors, each providing:
	1.	A range measurement (distance) from the sensor to the object.
	2.	A bearing measurement (angle) indicating the direction of the object relative to the sensor.
	The sensor locations in Cartesian coordinates ([x, y]) are:
	    {sensor_loc_str}
	The sensor readings are: {X_str}.
		- Each reading is in the form [A, B], where:
        	- A represents the distance from the object to the sensor.
        	- B represents the bearing angle (in degrees), measured within the range [0, 360).
            	i) Angle unit: degree, within the range [0, 360).
				ii) The bearing angle for each sensor is measured from the sensor’s own position (treated as the origin for its measurement).
                iii) The reference direction is the positive x-axis (a horizontal ray extending to the right from the sensor’s location).
                iv) Angles increase counterclockwise (CCW) from this reference direction.
        - The measurements may contain noise.

Question:

Using the given range and bearing measurements, determine the most likely location of the object in Cartesian coordinates [x, y]. 
"""

loc_region_prompt = """
Help me determine the location of an object in a 2D plane based on sensor measurements.

Return the location of the object in the following format:

[RESULTS_START] [x, y] [RESULTS_END]

Do not include any additional text between [RESULTS_START] and [RESULTS_END].

Given Information:

There are four sensors, each providing a region-based measurement indicating whether the object is within a defined detection region.
	Each sensor provides a binary reading:
		- 1: The object is inside the sensor detection region.
		- 0: The object is outside the sensor detection region.
	The sensor locations in Cartesian coordinates ([x, y]) are:
		{sensor_loc_str}
	Each sensor detects objects within a disk of radius {radius} around its location

The sensor readings are: {X_str}
	- Each reading is in the form [A], where:
	- A is either 1 or 0, indicating whether the object is within the sensor detection region.
	- The measurements may contain noise.

Question:

Using the given region-based sensor measurements, determine the most likely location of the object in Cartesian coordinates [x, y]. 
"""

loc_event_temp_prompt = """
Help me determine the time in seconds when a seismic event occurred in a 2D area based on sensor measurements.

Format Requirement:

Return the estimated time only in the following format, replacing t with the computed value:

[RESULTS_START] [t] [RESULTS_END]

Do not include any additional text between [RESULTS_START] and [RESULTS_END].

Scenario:
You have a 2D region (for simplicity, a 10 km by 10 km square) in which a seismic event (like a small earthquake or underground explosion) occurs at an unknown location. A set of seismic sensors (geophones) is placed in this region at known fixed positions. Each sensor is event-based: it records the time when it detects the seismic wave.

Detection-Time Model:
	Suppose each sensor’s detection time T depends linearly on the distance d from the event, using the seismic wave speed (5km/s):

T = d/5 (second),

meaning that if the sensor is 10 kilometers away from the event, it will detect it at 2 seconds after ignition, if it is 5 kilometers away, it detects at 1 second, etc.

Given Information:

There are four sensors, each providing the time that an event is detected.
	Each sensor provides the detection of the event in second:
	The sensor locations in Cartesian coordinates ([x, y], in km) are:
		{sensor_loc_str}

The sensor readings are: {X_str}
	- Each reading is in the form [A], where:
	- A a positive number, indicating the **second** the event is detected.
	- The measurements may contain noise.
    - The event’s occurrence time t is related to sensor readings by: A = t + T.

Localization Goal:
	By collecting these detection times in **second** from multiple sensors, the objective is to estimate when the seismic event occurred in **second**. Determine the time in the required format [RESULTS_START] [t] [RESULTS_END].
"""

loc_event_spatio_prompt = """
Help me determine the location of a seismic event occurred in a 2D area based on sensor measurements.

Format Requirement:

Return the event location in the following format, replacing x, y with the computed location:

[RESULTS_START] [x, y] [RESULTS_END]

Do not include any additional text between [RESULTS_START] and [RESULTS_END].

Scenario:
You have a 2D region (for simplicity, a 10 km by 10 km square) in which a seismic event (like a small earthquake or underground explosion) occurs at an unknown location. A set of seismic sensors (geophones) is placed in this region at known fixed positions. Each sensor is event-based: it records the time when it detects the seismic wave.

Detection-Time Model:
	Suppose each sensor’s detection time T depends linearly on the distance d from the event, using the seismic wave speed (5km/s):

T = d/5 (second),

meaning that if the sensor is 10 kilometers away from the event, it will detect it at 2 seconds after ignition, if it is 5 kilometers away, it detects at 1 second, etc.

Given Information:

There are four sensors, each providing the time that an event is detected.
	Each sensor provides the detection of the event in second:
	The sensor locations in Cartesian coordinates ([x_i, y_i], in km) are:
		{sensor_loc_str}

The sensor readings are: {X_str}
	- Each reading is in the form [A], where:
	- A a positive number, indicating the second the event is detected.
	- The measurements may contain noise.
    - Suppose the event happened at time t. The event’s occurrence time is related to sensor readings by: A = t + T.

Localization Goal:
	By collecting these detection times from multiple sensors, the objective is to estimate where the seismic event occurred (in km). Determine the location in the required format [RESULTS_START] [x, y] [RESULTS_END].
"""

loc_event_spatiotemp_prompt = """
Help me determine the location and time of a seismic event happened in a 2D plane based on sensor measurements.

Format Requirement:

Return the event location (x, y) and time t in the following format, replacing x, y, t with the computed location:

[RESULTS_START] [x, y, t] [RESULTS_END]

Do not include any additional text between [RESULTS_START] and [RESULTS_END].

Scenario:
You have a 2D region (for simplicity, a 10 km by 10 km square) in which a seismic event (like a small earthquake or underground explosion) occurs at an unknown location. A set of seismic sensors (geophones) is placed in this region at known fixed positions. Each sensor is event-based: it records the time when it detects the seismic wave.

Detection-Time Model:
	Suppose each sensor’s detection time T depends linearly on the distance d from the event, using the seismic wave speed (5km/s):

T = d/5 (second),

meaning that if the sensor is 10 kilometers away from the event, it will detect it at 2 seconds after ignition, if it is 5 kilometers away, it detects at 1 second, etc.

Given Information:

There are four sensors, each providing the time that an event is detected.
	Each sensor provides the detection of the event in second:
	The sensor locations in Cartesian coordinates ([x_i, y_i]) are:
		{sensor_loc_str}

The sensor readings are: {X_str}
	- Each reading is in the form [A], where:
	- A a positive number, indicating the second the event is detected.
	- The measurements may contain noise.
    - Suppose the event happened at time t. The event’s occurrence time is related to sensor readings by: A = t + T.

Localization Goal:
	By collecting these detection times from multiple sensors, the objective is to estimate where and when the seismic event occurred. Determine the location and time in the required format [RESULTS_START] [x, y, t] [RESULTS_END].
"""