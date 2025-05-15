range_context = """
Help me answer question regarding the trajectory of an object in a 10x10 2D plane:

Given Information:
    You will receive a set of range-only measurements d_At from up to four sensors at each time t. These measurements indicate the distance from the object to the sensors at time t. For example: 
    
    Sensor A: [d_A1, d_A2, ..., d_An]

    Timestamp: [t1, t2, ..., tn]

    The trace indicates range readings from sensor A from time t1 to tn.

	The sensor locations in Cartesian coordinates ([x, y]) are:
	    {sensor_loc_str}
	The sensor readings are an array.
	    - The reading represents the distance from the object to each sensor.
	    - The measurements may contain noise.
        - Some sensors may not report a distance (i.e., a missing reading) at certain time stamps.

"""

prompt_stpp = range_context + """
Objective:
	Based on the available distance measurements at each time stamp, estimate the closest point on the trajectory to a fixed point {X}?

Output Format:
    Your response **must** include the estimated location in the following format:

    [RESULTS_START] [x, y] [RESULTS_END]

    Do not include any text or additional formatting between the [RESULTS_START] and [RESULTS_END].

Example:

    [RESULTS_START] [1.23, 4.56] [RESULTS_END]

Sensor readings: {sensor_readings}
Timestamp: {timestr}
"""

prompt_sttp = range_context + """
Objective:
	Based on the available distance measurements at each time stamp, answer whether the trajectory crossed a line starting from {X_start} to {X_end}?

    Answer the probability of answer being Yes.

Output Format:
    Your response **must** include the probability in the following format:

    [RESULTS_START] [p] [RESULTS_END]

    Do not include any text or additional formatting between the [RESULTS_START] and [RESULTS_END].

Example:

    [RESULTS_START] [0.95] [RESULTS_END]

Sensor readings: {sensor_readings}
Timestamp: {timestr}
"""

prompt_strp = range_context + """
Objective:
	Based on the available distance measurements at each time stamp, answer whether the trajectory enter a rectangle region starting from bottom left {X_start} to top right {X_end}?

    Answer the probability of answer being Yes.

Output Format:
    Your response **must** include the probability in the following format:

    [RESULTS_START] [p] [RESULTS_END]

    Do not include any text or additional formatting between the [RESULTS_START] and [RESULTS_END].

Example:

    [RESULTS_START] [0.95] [RESULTS_END]

Sensor readings: {sensor_readings}
Timestamp: {timestr}
"""

prompt_stto = range_context + """
Objective:
	Based on the available distance measurements at each time stamp, answer whether the sensor measurements are first [\n{X_start}]\n then [\n{X_end}]\n as the object moves?

    Answer the probability of answer being Yes.

Output Format:
    Your response **must** include the probability in the following format:

    [RESULTS_START] [p] [RESULTS_END]

    Do not include any text or additional formatting between the [RESULTS_START] and [RESULTS_END].

Example:

    [RESULTS_START] [0.95] [RESULTS_END]

Sensor readings: {sensor_readings}
Timestamp: {timestr}
"""

prompt_sttov = range_context + """
Output Format:
    Your response **must** include the probability in the following format:

    [RESULTS_START] [delta_t] [RESULTS_END]

    Do not include any text or additional formatting between the [RESULTS_START] and [RESULTS_END].

Example:

    [RESULTS_START] [0.95] [RESULTS_END]

Sensor readings: {sensor_readings}
Timestamp: {timestr}
"""

prompt_sttl = range_context + """
Objective:
	Based on the available distance measurements at each time stamp, answer when did the event [\n{X_start}]\n happen as the object moved?

Output Format:
    Your response **must** include the probability in the following format:

    [RESULTS_START] [t] [RESULTS_END]

    Do not include any text or additional formatting between the [RESULTS_START] and [RESULTS_END].

Example:

    [RESULTS_START] [0.95] [RESULTS_END]

Sensor readings: {sensor_readings}
Timestamp: {timestr}
"""

prompt_sttlr = range_context + """
Objective:
	Based on the available distance measurements at each time stamp, answer when the object first enter the rectangle region starting from bottom left {X_start} to top right {X_end}?

    Answer the probability of answer being Yes.

Output Format:
    Your response **must** include the probability in the following format:

    [RESULTS_START] [p] [RESULTS_END]

    Do not include any text or additional formatting between the [RESULTS_START] and [RESULTS_END].

Example:

    [RESULTS_START] [0.95] [RESULTS_END]

Sensor readings: {sensor_readings}
Timestamp: {timestr}
"""

prompt_sttdr = range_context + """
Objective:
	Based on the available distance measurements at each time stamp, answer the time when the object stay in the rectangle region starting from bottom left {X_start} to top right {X_end}?

    Answer the probability of answer being Yes.

Output Format:
    Your response **must** include the probability in the following format:

    [RESULTS_START] [p] [RESULTS_END]

    Do not include any text or additional formatting between the [RESULTS_START] and [RESULTS_END].

Example:

    [RESULTS_START] [0.95] [RESULTS_END]

Sensor readings: {sensor_readings}
Timestamp: {timestr}
"""

prompt_sttor = range_context + """
Objective:
	Based on the available distance measurements at each time stamp, answer whether the object first enter the rectangle region starting from bottom left {X_start_1} to top right {X_end_1} than the rectangle region starting from bottom left {X_start_2} to top right {X_end_2}?

    Answer 1 if the answer being Yes. Otherwise, answer 0.

Output Format:
    Your response **must** include the answer in the following format:

    [RESULTS_START] [p] [RESULTS_END]

    Do not include any text or additional formatting between the [RESULTS_START] and [RESULTS_END].

Example:

    [RESULTS_START] [0.95] [RESULTS_END]

Sensor readings: {sensor_readings}
Timestamp: {timestr}
"""

prompt_stsm = range_context + """
Objective:
	Based on the available distance measurements at each time stamp, calculate the mean location of the object trajectory?

Output Format:
    Your response **must** include the mean location in the following format:

    [RESULTS_START] [x, y] [RESULTS_END]

    Do not include any text or additional formatting between the [RESULTS_START] and [RESULTS_END].

Example:

    [RESULTS_START] [0.123, 4.567] [RESULTS_END]

Sensor readings: {sensor_readings}
Timestamp: {timestr}
"""

prompt_stsv = range_context + """
Objective:
	Based on the available distance measurements at each time stamp, calculate the total spatial variance (biased) of the object trajectory?

Output Format:
    Your response **must** include the spatial variance in the following format:

    [RESULTS_START] [v] [RESULTS_END]

    Do not include any text or additional formatting between the [RESULTS_START] and [RESULTS_END].

Example:

    [RESULTS_START] [0.123] [RESULTS_END]

Sensor readings: {sensor_readings}
Timestamp: {timestr}
"""

prompt_sttm = range_context + """
Objective:
	Based on the available distance measurements at each time stamp, calculate the mean inter-measurement time of all sensors? 

Output Format:
    Your response **must** include the mean inter-measurement time in the following format (averaged across all sensors):

    [RESULTS_START] [v] [RESULTS_END]

    Do not include any text or additional formatting between the [RESULTS_START] and [RESULTS_END].

Example:

    [RESULTS_START] [0.123] [RESULTS_END]

Sensor readings: {sensor_readings}
Timestamp: {timestr}
"""

prompt_stsf = range_context + """
Objective:
	Based on the available measurements at each time stamp, forecast the next possible object location. 

Output Format:
    Your response **must** include the object location in the following format:

    [RESULTS_START] [x, y] [RESULTS_END]

    Do not include any text or additional formatting between the [RESULTS_START] and [RESULTS_END].

Example:

    [RESULTS_START] [0.123, 4.567] [RESULTS_END]

Sensor readings: {sensor_readings}
Timestamp: {timestr}
"""

prompt_stsi = range_context + """
Objective:
	The sensor measurements #{mask} are missing. Based on the available measurements at each time stamp, impute the missing object location at time {time}. 

Output Format:
    Your response **must** include the object location in the following format:

    [RESULTS_START] [x, y] [RESULTS_END]

    Do not include any text or additional formatting between the [RESULTS_START] and [RESULTS_END].

Example:

    [RESULTS_START] [0.123, 4.567] [RESULTS_END]

Sensor readings: {sensor_readings}
Timestamp: {timestr}
"""

prompt_ststf = range_context + """
Objective:
	Based on the available measurements at each time stamp, forecast the next possible object location and the time when a sensor will detect it.

Output Format:
    Your response **must** include the object location and time in the following format:

    [RESULTS_START] [x, y, t] [RESULTS_END]

    Do not include any text or additional formatting between the [RESULTS_START] and [RESULTS_END].

Example:

    [RESULTS_START] [0.123, 4.567, 8.910] [RESULTS_END]

Sensor readings: {sensor_readings}
Timestamp: {timestr}
"""

prompt_ststi = range_context + """
Objective:
	The sensor measurements #{mask} are missing. Based on the available measurements at each time stamp, impute the missing object location and the time when a sensor detected it. 

Output Format:
    Your response **must** include the object location and the time in the following format:

    [RESULTS_START] [x, y, t] [RESULTS_END]

    Do not include any text or additional formatting between the [RESULTS_START] and [RESULTS_END].

Example:

    [RESULTS_START] [0.123, 4.567, 8.910] [RESULTS_END]

Sensor readings: {sensor_readings}
Timestamp: {timestr}
"""