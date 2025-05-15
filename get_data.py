import numpy as np 
import pandas as pd 
import pdb 
from prompts.metadata_prompts import BLDG_prompt, Tracking_prompt
from prompts.task_prompts_localization import Forecasting, Tracking, loc_bearing_prompt, loc_range_prompt,\
    loc_range_bearing_prompt, loc_region_prompt, loc_event_temp_prompt, loc_event_spatio_prompt,\
    loc_event_spatiotemp_prompt
from prompts.task_prompts_tracking import track_range_prompt_online, track_bearing_prompt_online, track_region_prompt_online,\
    track_event_spatio_prompt_online, track_event_temp_prompt_online, track_range_bearing_prompt_online, track_event_spatio_temp_prompt_online
import random
from params import REGION_RADIUS, NORMAL_TEMPERATURE,\
      RANGE_NOISE_LEVEL, RANGE_NOISE_MEDIUM,\
    BEARING_NOISE_LEVEL, BEARING_NOISE_MEDIUM,\
    REGION_RADIUS,\
    EVENT_TEMP_NOISE_LEVEL, EVENT_SPATIO_NOISE_LEVEL, EVENT_SPATIOTEMP_NOISE_LEVEL
from sim_lib_s_v2 import spatial_relationship
from sim_lib_t_v2 import temporal_relationship

def encode_array(input_data, name="Room", encoded_str = "The collected data is as follows."):
    for i in range(len(input_data[0])):
        _data = input_data[:, i]
        if name is not None:
            _str = f"{name} {i}: ["
        for j in range(len(_data)):
            _str += "{:.2f}, ".format(_data[j])
        _str += "]\n "
        encoded_str = encoded_str + _str
    return encoded_str

def encode_1d_array(input_data, name="Room", encoded_str = "The collected data is as follows."):
    encoded_str += f"{name} ["
    for i in range(len(input_data)):
        encoded_str += "{:.2f}, ".format(input_data[i])
    encoded_str += "]\n "
    return encoded_str

def list2array(list):
    has_np = [isinstance(list[i], np.ndarray) for i in range(len(list))]
    if any(has_np):
        sensor_readings = ["{}".format(r) for r in list]
    else:
        sensor_readings = ["{:.4f}".format(r) for r in list]
    sensor_readings = ", ".join(sensor_readings)
    return sensor_readings

def read_sensor_location(file):
    df = pd.read_csv(file)
    sensor_loc_str = """
        - Sensor A: [{:.2f}, {:.2f}]
	    - Sensor B: [{:.2f}, {:.2f}]
	    - Sensor C: [{:.2f}, {:.2f}]
	    - Sensor D: [{:.2f}, {:.2f}]
    """.format(
        df['x'][0].item(), df['y'][0].item(),
        df['x'][1].item(), df['y'][1].item(),
        df['x'][2].item(), df['y'][2].item(),
        df['x'][3].item(), df['y'][3].item(),
    )
    return sensor_loc_str

def structure_tracking_data(args, ground_truth_pd, num_sensors, folder):
    sensor_readings = []
    timestamp = ground_truth_pd['t'].to_numpy()
    if args.dataset == 'track_event_spatiotemp_online':
        ground_truth = [ground_truth_pd["x"], ground_truth_pd["y"], ground_truth_pd['event_t']]
    else:
        ground_truth = [ground_truth_pd["x"], ground_truth_pd["y"]]
    ground_truth = np.array(ground_truth).T
    current_temperature = np.random.uniform(10,30)
    for t in range(len(timestamp)):
        sensor_readings_at_t = []
        for i in range(num_sensors):
            df = pd.read_csv(folder+f'{args.index}_sensor_{i}.csv')

            if args.dataset == 'track_range_bearing_online':
                try:
                    reading = np.fromstring(df['d'][t][1:-1].strip(), sep=' ')
                except:
                    reading = np.nan
            else:
                reading = df['d'][t].item()

            sensor_readings_at_t.append(reading)

        sensor_readings_at_t = list2array(sensor_readings_at_t)
        sensor_readings.append(sensor_readings_at_t)
    
    sensor_readings = {'timestamp': timestamp,'sensor_readings': sensor_readings, 'current temperature': current_temperature}
    return sensor_readings, ground_truth

def load_st_dataset(args, input_seq, output_seq, spt_seq, timestamp=0, offset=0, num_sensors=4):
    from sim_lib_st_v2 import obtain_spatio_temporal_relationship
    spatio_temporal_relationship = obtain_spatio_temporal_relationship()
    if args.dataset == 'BLDG':
        args.task = 'forecasting'
        sample_rate = '10T'
        df = pd.read_csv('./data/BLDG/zone_temp_interior.csv')
        columns = ['date'] + [f'cerc_templogger_{i}' for i in range(1, 17)]
        df = df[columns]
        df['date'] = pd.to_datetime(df['date'])

        # Interpolating NaN values
        df_interpolated = df.copy()
        df_interpolated.iloc[:, 1:] = df.iloc[:, 1:].interpolate(method='linear')
        data = df_interpolated[columns[1:]].to_numpy()
        timestamp = df_interpolated[columns[0]]
        i = args.index
        if spt_seq is None:
            input_data = data[i*input_seq: (i+1)*input_seq, :]
            output_data = data[(i+1)*input_seq: (i+1)*input_seq + output_seq, :]
        else:
            input_data = data[i*input_seq: (i+1)*input_seq, :spt_seq]
            output_data = data[(i+1)*input_seq: (i+1)*input_seq + output_seq, :spt_seq]
        input_data_str = encode_array(input_data)
        input_data_str = "The start time is {}. ".format(str(timestamp[0])) + input_data_str
        input_data_str = BLDG_prompt + input_data_str
        query_str = Forecasting.format(output_seq, "room")
    elif args.dataset == 'loc_range':
        args.task = 'loc_range'
        if args.noise_level == 'medium':
            std = RANGE_NOISE_MEDIUM
        else:
            std = RANGE_NOISE_LEVEL
        folder = f'./data/localization_range_{std}/'
        
        sensor_readings = []
        for i in range(num_sensors):
            df = pd.read_csv(folder+f'{args.index}_sensor_{i}.csv')
            reading = df['d'].item()
            sensor_readings.append(reading)
        sensor_readings = list2array(sensor_readings)
        ground_truth = pd.read_csv(folder + f"{args.index}_object_location.csv")
        ground_truth = [ground_truth["x"].item(), ground_truth["y"].item()]
        ground_truth = np.array(ground_truth)
        sensor_loc = read_sensor_location(folder + f'{args.index}_sensor_location.csv')
        input_data_str, query_str, output_data = sensor_readings, loc_range_prompt.format(X_str=sensor_readings, sensor_loc_str=sensor_loc), ground_truth
    elif args.dataset == 'loc_range_bearing':
        args.task = 'loc_range_bearing'
        folder = f'./data/localization_range_bearing_[{RANGE_NOISE_LEVEL}, {BEARING_NOISE_LEVEL}]/'
        sensor_readings = []
        for i in range(num_sensors):
            df = pd.read_csv(folder+f'{args.index}_sensor_{i}.csv')
            reading = df['d'].item()[2:-1].split(" ")
            reading = [float(r) for r in reading if r!=""]
            # pdb.set_trace()
            reading = list2array(reading)
            sensor_readings.append(f"[{reading}]")
        sensor_readings = ", ".join(sensor_readings)
        ground_truth = pd.read_csv(folder + f"{args.index}_object_location.csv")
        ground_truth = [ground_truth["x"].item(), ground_truth["y"].item()]
        ground_truth = np.array(ground_truth)
        sensor_loc = read_sensor_location(folder + f'{args.index}_sensor_location.csv')
        input_data_str, query_str, output_data = sensor_readings, loc_range_bearing_prompt.format(X_str=sensor_readings, sensor_loc_str=sensor_loc), ground_truth
    elif args.dataset == 'loc_bearing':
        args.task = 'loc_bearing'
        if args.noise_level == 'medium':
            std = BEARING_NOISE_MEDIUM
        else:
            std = BEARING_NOISE_LEVEL
        folder = f'./data/localization_bearing_{std}/'
        sensor_readings = []
        for i in range(num_sensors):
            df = pd.read_csv(folder+f'{args.index}_sensor_{i}.csv')
            reading = df['d'].item()
            sensor_readings.append(reading)
        sensor_readings = list2array(sensor_readings)
        ground_truth = pd.read_csv(folder + f"{args.index}_object_location.csv")
        ground_truth = [ground_truth["x"].item(), ground_truth["y"].item()]
        ground_truth = np.array(ground_truth)
        sensor_loc = read_sensor_location(folder + f'{args.index}_sensor_location.csv')
        input_data_str, query_str, output_data = sensor_readings, loc_bearing_prompt.format(X_str=sensor_readings, sensor_loc_str=sensor_loc), ground_truth
    elif args.dataset == 'loc_region':
        args.task = 'loc_region'
        radius = f"{REGION_RADIUS}"
        folder = f'./data/localization_region_{radius}/'
        sensor_readings = []
        for i in range(num_sensors):
            df = pd.read_csv(folder+f'{args.index}_sensor_{i}.csv')
            reading = df['d'].item()
            sensor_readings.append(reading)
        sensor_readings = list2array(sensor_readings)
        ground_truth = pd.read_csv(folder + f"{args.index}_object_location.csv")
        ground_truth = [ground_truth["x"].item(), ground_truth["y"].item()]
        ground_truth = np.array(ground_truth)
        sensor_loc = read_sensor_location(folder + f'{args.index}_sensor_location.csv')
        input_data_str, query_str, output_data = sensor_readings, loc_region_prompt.format(X_str=sensor_readings, radius=radius, sensor_loc_str=sensor_loc), ground_truth
    elif args.dataset == 'loc_event_temp':
        args.task = 'loc_event_temp'
        std = f"{EVENT_TEMP_NOISE_LEVEL}"
        folder = f'./data/localization_event_temp_{std}/'
        sensor_readings = []
        for i in range(num_sensors):
            df = pd.read_csv(folder+f'{args.index}_sensor_{i}.csv')
            reading = df['d'].item()
            sensor_readings.append(reading)
        sensor_readings = list2array(sensor_readings)
        ground_truth = pd.read_csv(folder + f"{args.index}_object_location.csv")
        ground_truth = [ground_truth["x"].item()]
        ground_truth = np.array(ground_truth)
        sensor_loc = read_sensor_location(folder + f'{args.index}_sensor_location.csv')
        input_data_str, query_str, output_data = sensor_readings, loc_event_temp_prompt.format(X_str=sensor_readings, sensor_loc_str=sensor_loc), ground_truth
    elif args.dataset == 'loc_event_spatio':
        args.task = 'loc_event_spatio'
        std = f"{EVENT_SPATIO_NOISE_LEVEL}"
        folder = f'./data/localization_event_spatio_{std}/'
        sensor_readings = []
        for i in range(num_sensors):
            df = pd.read_csv(folder+f'{args.index}_sensor_{i}.csv')
            reading = df['d'].item()
            sensor_readings.append(reading)
        sensor_readings = list2array(sensor_readings)
        ground_truth = pd.read_csv(folder + f"{args.index}_object_location.csv")
        ground_truth = [ground_truth["x"].item(), ground_truth["y"].item()]
        ground_truth = np.array(ground_truth)
        sensor_loc = read_sensor_location(folder + f'{args.index}_sensor_location.csv')
        input_data_str, query_str, output_data = sensor_readings, loc_event_spatio_prompt.format(X_str=sensor_readings, sensor_loc_str=sensor_loc), ground_truth
    elif args.dataset == 'loc_event_spatio_temp':
        args.task = 'loc_event_spatio_temp'
        std = f"{EVENT_SPATIOTEMP_NOISE_LEVEL}"
        folder = f'./data/localization_event_spatio_temp_{std}/'
        sensor_readings = []
        for i in range(num_sensors):
            df = pd.read_csv(folder+f'{args.index}_sensor_{i}.csv')
            reading = df['d'].item()
            sensor_readings.append(reading)
        sensor_readings = list2array(sensor_readings)
        ground_truth = pd.read_csv(folder + f"{args.index}_object_location.csv")
        ground_truth = [ground_truth["x"].item(), ground_truth["y"].item(), ground_truth["t"].item()]
        ground_truth = np.array(ground_truth)
        sensor_loc = read_sensor_location(folder + f'{args.index}_sensor_location.csv')
        input_data_str, query_str, output_data = sensor_readings, loc_event_spatiotemp_prompt.format(X_str=sensor_readings, sensor_loc_str=sensor_loc), ground_truth
    elif args.dataset == 'track_range_online':
        args.task = 'track_range_online'
        std = RANGE_NOISE_LEVEL
        folder = f'./data/tracking_range_{std}/'

        ground_truth_pd = pd.read_csv(folder + f"{args.index}_object_location.csv")
        sensor_loc = read_sensor_location(folder + f'{args.index}_sensor_location.csv')
        sensor_readings, ground_truth = structure_tracking_data(args, ground_truth_pd, num_sensors, folder)
        input_data_str, query_str, output_data = sensor_readings, track_range_prompt_online.format(sensor_loc_str=sensor_loc), ground_truth
    elif args.dataset == 'track_bearing_online':
        args.task = 'track_bearing_online'
        std = BEARING_NOISE_LEVEL
        folder = f'./data/tracking_bearing_{std}/'

        ground_truth_pd = pd.read_csv(folder + f"{args.index}_object_location.csv")
        sensor_loc = read_sensor_location(folder + f'{args.index}_sensor_location.csv')
        sensor_readings, ground_truth = structure_tracking_data(args, ground_truth_pd, num_sensors, folder)
        input_data_str, query_str, output_data = sensor_readings, track_bearing_prompt_online.format(sensor_loc_str=sensor_loc), ground_truth
    elif args.dataset == 'track_range_bearing_online':
        args.task = 'track_range_bearing_online'
        std_bearing = BEARING_NOISE_LEVEL
        std_range = RANGE_NOISE_LEVEL
        folder = f'./data/tracking_range_bearing_[{std_range}, {std_bearing}]/'

        ground_truth_pd = pd.read_csv(folder + f"{args.index}_object_location.csv")
        sensor_loc = read_sensor_location(folder + f'{args.index}_sensor_location.csv')
        sensor_readings, ground_truth = structure_tracking_data(args, ground_truth_pd, num_sensors, folder)
        input_data_str, query_str, output_data = sensor_readings, track_range_bearing_prompt_online.format(sensor_loc_str=sensor_loc), ground_truth
    elif args.dataset == 'track_region_online':
        args.task = 'track_region_online'
        std = f"{REGION_RADIUS}"
        folder = f'./data/tracking_region_{std}/'
        radius = REGION_RADIUS
        ground_truth_pd = pd.read_csv(folder + f"{args.index}_object_location.csv")
        sensor_loc = read_sensor_location(folder + f'{args.index}_sensor_location.csv')
        sensor_readings, ground_truth = structure_tracking_data(args, ground_truth_pd, num_sensors, folder)
        input_data_str, query_str, output_data = sensor_readings, track_region_prompt_online.format(sensor_loc_str=sensor_loc, radius=radius), ground_truth
    elif args.dataset == 'track_event_spatio_online':
        args.task = 'track_event_spatio_online'
        std = EVENT_SPATIO_NOISE_LEVEL
        folder = f'./data/tracking_event_spatio_{std}/'

        ground_truth_pd = pd.read_csv(folder + f"{args.index}_object_location.csv")
        sensor_loc = read_sensor_location(folder + f'{args.index}_sensor_location.csv')
        sensor_readings, ground_truth = structure_tracking_data(args, ground_truth_pd, num_sensors, folder)
        input_data_str, query_str, output_data = sensor_readings, track_event_spatio_prompt_online.format(sensor_loc_str=sensor_loc), ground_truth
    elif args.dataset == 'track_event_temp_online':
        args.task = 'track_event_temp_online'
        std = EVENT_TEMP_NOISE_LEVEL
        folder = f'./data/tracking_event_temp_{std}/'

        ground_truth_pd = pd.read_csv(folder + f"{args.index}_object_location.csv")
        sensor_loc = read_sensor_location(folder + f'{args.index}_sensor_location.csv')
        sensor_readings, ground_truth = structure_tracking_data(args, ground_truth_pd, num_sensors, folder)
        input_data_str, query_str, output_data = sensor_readings, track_event_temp_prompt_online.format(sensor_loc_str=sensor_loc), ground_truth[:,:1]
    elif args.dataset == 'track_event_spatiotemp_online':
        args.task = 'track_event_spatiotemp_online'
        std = EVENT_SPATIOTEMP_NOISE_LEVEL
        folder = f'./data/tracking_event_spatio_temp_{std}/'

        ground_truth_pd = pd.read_csv(folder + f"{args.index}_object_location.csv")
        sensor_loc = read_sensor_location(folder + f'{args.index}_sensor_location.csv')
        sensor_readings, ground_truth = structure_tracking_data(args, ground_truth_pd, num_sensors, folder)
        
        input_data_str, query_str, output_data = sensor_readings, track_event_spatio_temp_prompt_online.format(sensor_loc_str=sensor_loc), ground_truth
    elif args.dataset in ('stpp', 'sttp', 'strp', 'stto', 'sttov', 'sttl'):
        args.task = args.dataset
        folder = f'./data/tier2_{args.dataset}/'
        pd_file = pd.read_csv(folder + f"{args.index}.csv")
        input_data_str, query_str, output_data = None,\
            pd_file['query'][0], pd_file['answer'][0]
        
        output_data = str(output_data)
        output_data = output_data.replace("[", "").replace("]", "")

        output_data = np.fromstring(output_data, sep=" ")
    elif args.dataset in spatial_relationship:
        args.task = args.dataset
        folder = f'./data/tier2_v2_spatial/{args.dataset}/'
        pd_file = pd.read_csv(folder + f"{args.index}.csv")
        input_data_str, query_str, output_data = None,\
            pd_file['query'][0], pd_file['answer'][0]
        
        output_data = str(output_data)
        output_data = output_data.replace("[", "").replace("]", "")

        output_data = np.fromstring(output_data, sep=" ")
    elif args.dataset in temporal_relationship:
        args.task = args.dataset
        folder = f'./data/tier2_v2_temporal/{args.dataset}/'
        pd_file = pd.read_csv(folder + f"{args.index}.csv")
        input_data_str, query_str, output_data = None,\
            pd_file['query'][0], pd_file['answer'][0]
        
        output_data = str(output_data)
        output_data = output_data.replace("[", "").replace("]", "")

        output_data = np.fromstring(output_data, sep=" ")
    elif args.dataset in spatio_temporal_relationship:
        args.task = args.dataset
        folder = f'./data/tier2_v2_spatial_temporal/{args.dataset}/'
        pd_file = pd.read_csv(folder + f"{args.index}.csv")
        input_data_str, query_str, output_data = None,\
            pd_file['query'][0], pd_file['answer'][0]
        
        output_data = str(output_data)
        output_data = output_data.replace("[", "").replace("]", "")

        output_data = np.fromstring(output_data, sep=" ")
    elif args.dataset in ['intent_pred', 'poi_pred', 'route_planning', 'landmark_questions', 'direction_questions', 'travel_questions', 'subroute_duration']:
        if args.dataset == 'travel_questions':
            dataset = 'travel_questions'
        else:
            dataset = args.dataset
        args.task = args.dataset
        folder = f'./data/tier3_world_knowledge/{dataset}/'
        pd_file = pd.read_csv(folder + f"{args.index}.csv")
        input_data_str, query_str, output_data = None,\
            pd_file['query'][0], pd_file['answer'][0]
        
        output_data = str(output_data)
        output_data = output_data.replace("[", "").replace("]", "")

        output_data = np.fromstring(output_data, sep=" ")
    elif args.dataset in ['spatial_impute', 'temporal_impute']:
        dataset = args.dataset
        args.task = args.dataset
        folder = f'./data/{dataset}/'

        pd_file = pd.read_csv(folder + f"{args.index}.csv")
        input_data_str, query_str, output_data = None, pd_file['query'][0], pd_file['answer'][0]
        
        output_data = str(output_data)
        output_data = output_data.replace("[", "").replace("]", "")

        output_data = np.fromstring(output_data, sep=" ")
        query_str = query_str.replace("RESULT_", "RESULTS_")
        
    elif args.dataset in ['spatiotemporal_forecast', 'spatiotemporal_impute']:
        dataset = args.dataset
        args.task = args.dataset
        folder = f'./data/{dataset}/'

        input_data_str = None
        
        query_str, output_data = read_csv_line_by_line(folder + f"{args.index}.csv")
        output_data = output_data.replace("[", "").replace("]", "")

        output_data = np.fromstring(output_data, sep=",")
        query_str = query_str.replace("RESULT_", "RESULTS_")
    else:
        raise NotImplementedError(f"The dataset {args.dataset} has not being incorporated.")
    return input_data_str, query_str, output_data, 

def read_csv_line_by_line(file_path):
    """
    Reads a CSV file line by line without using the csv module.
    Each line is split by commas and printed as a list of values.

    Parameters:
    file_path (str): Path to the CSV file
    """
    contents = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Strip newline and split by comma
            contents.append(line)
    query = contents[-1].replace('",', "")
    contents = "".join(contents[2:-1])
    return contents, query
    
            

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="BLDG", help="ST dataset use for evaluation")
    parser.add_argument(
        "--index", type=int, default=0, help="The index to retrieve the data")
    args = parser.parse_args()
    load_st_dataset(args, input_seq=144, output_seq=2, spt_seq=None)