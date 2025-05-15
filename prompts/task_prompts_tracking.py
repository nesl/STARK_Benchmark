track_range_prompt_online = """
Help me track the location of an object in a 2D plane at each step based on temporal sensor measurements.

Given Information:
    You will receive a set of range-only measurements from up to four sensors at each time stamp. These measurements indicate the distance from the object to the sensors.
	The sensor locations in Cartesian coordinates ([x, y]) are:
	    {sensor_loc_str}
	The sensor readings are an array.
	    - The reading represents the distance from the object to each sensor.
	    - The measurements may contain noise.
        - Some sensors may not report a distance (i.e., a missing reading) at certain time stamps.

Objective:
	Based on the available distance measurements at each time stamp (including the historical data), estimate the most likely [x, y] coordinates of the object in the 2D plane.        

Output Format:
    Your response **must** include the object’s estimated location in the following format:

    [RESULTS_START] [x, y] [RESULTS_END]

    Do not include any text or additional formatting between the [RESULTS_START] and [RESULTS_END].

Example:
	Suppose at a certain time stamp, sensor 1 and sensor 3 provide distance measurements, while sensors 2 and 4 do not. Use only the available readings and historical data to compute the object’s best-guess location.
	
    Return the results in the exact format:

    [RESULTS_START] [1.23, 4.56] [RESULTS_END]
    

Question:

Based on the given data, where is the object most likely located at each step? Provide the estimated [x, y] coordinates. You **must** give an estimation.

"""

track_bearing_prompt_online = """
Help me track the location of an object in a 2D plane at each step based on sensor measurements.

Given Information:
    There are four sensors, each providing:
	A bearing measurement (angle) indicating the direction of the object relative to the sensor.
	The sensor locations in Cartesian coordinates ([x, y]) are:
	    {sensor_loc_str}
	The sensor readings are an array.
		- Units: degrees, within the range [0, 360).
        - The bearing angle for each sensor is measured from the sensor’s own position (treated as the origin for its measurement).
        - The reference direction is the positive x-axis (a horizontal ray extending to the right from the sensor’s location).
        - Angles increase counterclockwise (CCW) from this reference direction.
        - The measurements may contain noise. Some sensors may not report a reading at certain time stamps.

Objective:
	Based on the available bearing measurements at each time stamp (including the historical data), estimate the most likely [x, y] coordinates of the object in the 2D plane.        

Output Format:
    Your response **must** include the object’s estimated location in the following format:

    [RESULTS_START] [x, y] [RESULTS_END]

    Do not include any text or additional formatting between the [RESULTS_START] and [RESULTS_END].

Example:
	Suppose at a certain time stamp, sensor 1 and sensor 3 provide bearing measurements, while sensors 2 and 4 do not. Use only the available readings and historical data to compute the object’s best-guess location.
	
    Return the results in the exact format:

    [RESULTS_START] [1.23, 4.56] [RESULTS_END]

Question:

Using the given bearing measurements, determine the most likely location of the object in Cartesian coordinates [x, y] at each step. 
"""

track_range_bearing_prompt_online = """
Help me track the location of an object in a 2D plane at each step based on sensor measurements.

Given Information:
    There are four sensors, each providing:
	1.	A range measurement (distance) from each sensor to the object at each time stamp.
	2.	A bearing measurement (angle) indicating the direction of the object relative to the sensor.
	The sensor locations in Cartesian coordinates ([x, y]) are:
	    {sensor_loc_str}
	The sensor readings are an array.
		- bearing units: degrees, within the range [0, 360).
        - The bearing angle for each sensor is measured from the sensor’s own position (treated as the origin for its measurement).
        - The reference direction is the positive x-axis (a horizontal ray extending to the right from the sensor’s location).
        - Angles increase counterclockwise (CCW) from this reference direction.
        - The measurements may contain noise. Some sensors may not report a reading at certain time stamps.

Objective:
	Based on the available measurements (range, bearing) at each time stamp (including the historical data), estimate the most likely [x, y] coordinates of the object in the 2D plane.        

Output Format:
    Your response **must** include the object’s estimated location in the following format:

    [RESULTS_START] [x, y] [RESULTS_END]

    Do not include any text or additional formatting between the [RESULTS_START] and [RESULTS_END].

Example:
	Suppose at a certain time stamp, sensor 1 and sensor 3 provide bearing measurements, while sensors 2 and 4 do not. Use only the available readings and historical data to compute the object’s best-guess location.
	
    Return the results in the exact format:

    [RESULTS_START] [1.23, 4.56] [RESULTS_END]

Question:

Using the given measurements, determine the most likely location of the object in Cartesian coordinates [x, y] at each step. 
"""

track_region_prompt_online = """
Help me determine the location of an object in a 2D plane based on sensor measurements.

There are four sensors, each providing a region-based measurement indicating whether the object is within a defined detection region.
	Each sensor provides a binary reading:
		- 1: The object is inside the sensor detection region.
		- 0: The object is outside the sensor detection region.
	The sensor locations in Cartesian coordinates ([x, y]) are:
		{sensor_loc_str}
	Each sensor detects objects within a disk of radius {radius} around its location

Objective:
	Based on the available proximity measurements at each time stamp (including the historical data), estimate the most likely [x, y] coordinates of the object in the 2D plane.        

Output Format:
    Your response **must** include the object’s estimated location in the following format:

    [RESULTS_START] [x, y] [RESULTS_END]

    Do not include any text or additional formatting between the [RESULTS_START] and [RESULTS_END].

Example:
	Suppose at a certain time stamp, sensor 1 and sensor 3 provide positive proximity measurements, while sensors 2 and 4 do not. Use only the available readings and historical data to compute the object’s best-guess location.
	
    Return the results in the exact format:

    [RESULTS_START] [1.23, 4.56] [RESULTS_END]

Question:

Using the given proximity measurements, determine the most likely location of the object in Cartesian coordinates [x, y]. 
"""

# track_event_spatio_prompt_online = """
# Help me determine the location of fire in a 2D plane based on temporal sensor measurements.

# Given Information:
#     You will receive a set of temperature measurements from up to four sensors at each time stamp. These measurements indicate the value of temperature at a location.
# 	The sensor locations in Cartesian coordinates ([x, y]) are:
# 	    {sensor_loc_str}
# 	The sensor readings is an array.
# 	    - The reading represents the temperature readings of sensors at their corresponding location.
# 	    - The measurements may contain noise.
#         - Some sensors may not report a reading (i.e., a missing reading) at certain time stamps.

# Objective:
# 	Based on the available temperature measurements at each time stamp (including the historical data), estimate the most likely [x, y] coordinates of the fire in the 2D plane.        

# Output Format:
#     Your response **must** include the fire estimated location in the following format:

#     [RESULTS_START] [x, y] [RESULTS_END]

#     Do not include any text or additional formatting between the [RESULTS_START] and [RESULTS_END].

# Temperature-Time Model:
# 	Suppose each sensor’s reported temperature T depends linearly on the distance d from the fire:

# T = d/10 + current_temperature (Celcius Degree),
    
# Example:
# 	Suppose at a certain time stamp, sensor 1 and sensor 3 provide temperature measurements, while sensors 2 and 4 do not. Use only the available readings and historical data to compute the object’s best-guess location.
	
#     Return the results in the exact format:

#     [RESULTS_START] [1.23, 4.56] [RESULTS_END]
    

# Question:

# Based on the given data, where is the fire most likely located? Provide the estimated [x, y] coordinates. You **must** give an estimation.
# """

track_event_spatio_prompt_online = """
There is a shooter moving in a 2D plane (10 x 10 km) monitored by sensors. Help me track the location of a shooter based on time-of-arrival (TOA) measurements from up to four microphones.

Given Information:
    At each step, you will receive a set of time-of-arrival measurements from up to four sensors at each time stamp (in minute). These measurements indicate how long after the shot was fired that each sensor detected the sound. 
    The sensor locations in Cartesian coordinates ([x, y], in km) are:
        {sensor_loc_str}
    The sensor readings come in an array:
        - The reading represents the TOA at the corresponding sensor location.
        - Some sensors may not report a reading (i.e., missing data) at certain time stamps.
        - The measurements may contain noise.

Objective:
    Based on the available time-of-arrival measurements at each time stamp (including historical data), estimate the most likely [x, y] coordinates of the shooter in the 2D plane.

Output Format:
    Your response **must** include the shooter's estimated location in the following format:
    
    [RESULTS_START] [x, y] [RESULTS_END]

    Do not include any text or additional formatting between the [RESULTS_START] and [RESULTS_END].

Acoustic-Time Model:
    Assume each microphone receives the shot at time T given by:
    
        T = d / v + T_0,
    
    where
        d   = distance from the shooter to the microphone,
        v   = speed of sound (approximately 20 km/min),
        T_0 = the time offset when the shot was actually fired (which you may approximate or fit from the data).

Example:
    Suppose at a certain time stamp, sensor 1 and sensor 3 provide TOA measurements, while sensors 2 and 4 do not. Use only the available readings and historical data to compute the best-guess location of the shooter.

    Return the result in the exact format (in km) at each step:
    
    [RESULTS_START] [1.23, 4.56] [RESULTS_END]

Question:

Based on the given data, where is the shooter most likely located at each step? Provide the estimated [x, y] coordinates. You **must** give an estimation.
"""

track_event_temp_prompt_online = """
There is a shooter moving in a 2D plane (10 x 10 km) monitored by sensors. Help me track the time a gunshot occurred in a 2D plane (10 x 10 km) based on time-of-arrival (TOA) measurements from up to four microphones.

Given Information:
    At each step, you will receive a set of time-of-arrival measurements from up to four sensors at each time stamp (in minute). These measurements indicate how long after the shot was fired that each sensor detected the sound. 
    The sensor locations in Cartesian coordinates ([x, y], in km) are:
        {sensor_loc_str}
    But for the purpose of this problem, we are focusing on estimating the shot’s initial firing time (T_0).
    The sensor readings come in an array:
        - The reading represents the TOA at the corresponding sensor location.
        - Some sensors may not report a reading (i.e., missing data) at certain time stamps.
        - The measurements may contain noise.

Objective:
    Based on the available time-of-arrival measurements at each time stamp (including historical data), estimate the most likely time T_0 (in min) at which the shot was fired.

Output Format:
    Your response **must** include the shooter's estimated location in the following format:
    
    [RESULTS_START] [T_0] [RESULTS_END]

    Do not include any text or additional formatting between the [RESULTS_START] and [RESULTS_END].

Acoustic-Time Model:
    Assume each microphone receives the shot at time T given by:
    
        T = d / v + T_0,
    
    where
        d   = distance from the shooter to the microphone (need to be estimated),
        v   = speed of sound (approximately 20 km/min),
        T_0 = the time offset when the shot was actually fired (which you may approximate or fit from the data).

Example:
    Suppose at a certain time stamp, sensor 1 and sensor 3 provide TOA measurements, while sensors 2 and 4 do not. Use only the available readings and historical data to compute the best-guess firing time.

    Return the result in the exact format (in min) at each step:
    
    [RESULTS_START] [0.1234] [RESULTS_END]

Question:

Based on the given TOA data, estimate the T_0 (in min) when the shot was fired at each step. Provide a numeric answer and you **must** give an estimation.
"""

track_event_spatio_temp_prompt_online = """
There is a shooter moving in a 2D plane (10 x 10 km) monitored by sensors. Help me track the location of a shooter based on time-of-arrival (TOA) measurements from up to four microphones.

Given Information:
    At each step, you will receive a set of time-of-arrival measurements from up to four sensors at each time stamp (in minute). These measurements indicate how long after the shot was fired that each sensor detected the sound. 
    The sensor locations in Cartesian coordinates ([x, y], in km) are:
        {sensor_loc_str}
    The sensor readings come in an array:
        - The reading represents the TOA at the corresponding sensor location.
        - Some sensors may not report a reading (i.e., missing data) at certain time stamps.
        - The measurements may contain noise.

Objective:
    Based on the available time-of-arrival measurements at each time stamp (including historical data), estimate the most likely [x, y] coordinates of the shooter and the time t (in min) at which the shot was fired in the 2D plane.

Output Format:
    Your response **must** include the shooter's estimated location in the following format:
    
    [RESULTS_START] [x, y, t] [RESULTS_END]

    Do not include any text or additional formatting between the [RESULTS_START] and [RESULTS_END].

Acoustic-Time Model:
    Assume each microphone receives the shot at time T given by:
    
        T = d / v + T_0,
    
    where
        d   = distance from the shooter to the microphone,
        v   = speed of sound (approximately 20 km/min),
        T_0 = the time offset when the shot was actually fired (which you may approximate or fit from the data).

Example:
    Suppose at a certain time stamp, sensor 1 and sensor 3 provide TOA measurements, while sensors 2 and 4 do not. Use only the available readings and historical data to compute the best-guess location of the shooter.

    Return the result in the exact format at each step:
    
    [RESULTS_START] [1.23, 4.56, 7.89] [RESULTS_END]

Question:

Based on the given data, where and when is the shooter most likely located at each step? Provide the estimated [x, y, t]. You **must** give an estimation.
"""