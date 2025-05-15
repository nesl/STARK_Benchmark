
BLDG_prompt = """
The dataset presents a comprehensive three-year collection of building energy and occupancy data from an office building in Berkeley, California. \
The dataset includes indoor air temperature at 16 sites across the building. The sampling rate is 10 minute.

Following is the adjacency of each room. The number indicates the room ID:

    "1": ["2", "3"],
    "2": ["1", "3", "4"],
    "3": ["1", "2", "4"],
    "4": ["2", "3", "5"],
    "5": ["4", "6", "8"],
    "6": ["5", "7", "8"],
    "7": ["6", "8"],
    "8": ["5", "6", "7"],

    "9": ["10", "11", "12"],
    "10": ["9", "11", "12"],
    "11": ["9", "10", "12", "13"],
    "12": ["9", "10", "11"],

    "13": ["11", "14", "15", "16"],
    "14": ["13", "15", "16"],
    "15": ["13", "14", "16"],
    "16": ["13", "14", "15"],

Following is the location of each site:

Second‐level office floor (South side): [1, 2, 3]
Second‐level office floor (North side): [4, 5, 6, 7, 8]
Ground‐level office floor (South side): [9, 10, 11, 12]
Ground‐level office floor (North side): [13, 14, 15, 16]

"""

Tracking_prompt = """"
You need to estimate the path taken by an object through a 1-D space over time given observations from 3 sensors located at the following locations:

[0.0, 0.0, 0.0],   # Sensor 0
[10.0, 0.0, 0.0],  # Sensor 1
[0.0, 10.0, 5.0],  # Sensor 2
[5.0, 5.0, 8.0]    # Sensor 3

You will be given the historical and current time t readings of Sensor i as in <t, x, y, z>.

This is an online task. At each time, you need to estimate the state of the object. And update the location of the object given the new sensor readings. 

If sensor readings are unavilable, estimate the object states using the historical data.

"""

