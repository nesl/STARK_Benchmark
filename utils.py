from openai import OpenAI
import openai 
from transformers import TextStreamer
import os
import sys
import io
import re 
import csv 
from time import sleep
import pdb 
import pandas as pd 
import numpy as np 
from together import Together
import ast

field_variable_tasks = ['spatial_impute', 'spatiotemporal_forecast', 'spatiotemporal_impute', 'temporal_impute']
# field_variable_tasks = ['temporal_impute', 'spatiotemporal_impute', 'spatiotemporal_forecast']

def chat(model, tokenizer, messages, text_streamer=None):
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")

    if text_streamer is None:
        text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    output = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 2048,
                    use_cache = True, temperature = 1.5, min_p = 0.1)
    return output

def guard_imports(code_str):
	"""
	Checks if the given Python code (as a string) imports any libraries
	other than those in the allowed list: 'scipy', 'numpy', 'shapely'.
	Args:
		code_str (str): The source code to check.

	Returns:
		Tuple[bool, set or None]: Returns a tuple where the first value is
		a boolean indicating if the code is "clean" (True if only allowed libraries
		are imported), and the second value is either None (if valid) or a
		set containing the disallowed top-level libraries.
	"""
	allowed = {'scipy', 'numpy', 'shapely', 'pandas', 'os', 'csv', 'filterpy'}
	disallowed = set()

	try:
		tree = ast.parse(code_str)
	except SyntaxError as e:
		raise ValueError(f"Invalid Python code provided: {e}")

	for node in ast.walk(tree):
		# Check statements like: "import module", "import module as alias"
		if isinstance(node, ast.Import):
			for alias in node.names:
				top_module = alias.name.split('.')[0]
				if top_module not in allowed:
					disallowed.add(top_module)
		# Check statements like: "from module import func"
		elif isinstance(node, ast.ImportFrom):
			if node.module is not None:
				top_module = node.module.split('.')[0]
				if top_module not in allowed:
					disallowed.add(top_module)

	if disallowed:
		return False, disallowed
	return True, None

def openai_api(messages, model, api_key, temperature=1, top_p=1, stop=None):

	got_result = False
	using_together = 'Llama' in model or 'Qwen' in model or 'Mistral' in model
	using_deepseek = 'deepseek' in model
	if using_together:
		# client = openai.OpenAI(
		# 	api_key=open("together_key.txt").read().strip(),
		# 	base_url="https://api.together.xyz/v1",
		# 	)
		client = Together(api_key=open("together_key.txt").read().strip())
	elif using_deepseek:
		client = openai.OpenAI(
			api_key=open("deepseek_key.txt").read().strip(),
			base_url="https://api.deepseek.com",
			)
	else:
		client = OpenAI(api_key=api_key)
	trial = 0
	if not using_deepseek or not using_together:
		max_token = 2048*8 
	else: 
		max_token = 4096

	is_stream = False if model == 'o3' else True

	while not got_result and trial <= 5:
		try:
			trial += 1
			if model in ('o1', 'o3-mini', 'o3', 'o4-mini') or using_deepseek or using_together:
				stream = client.chat.completions.create(
					model=model,
					messages=messages,
					# stream=True,
					stream=is_stream,
					stop=stop)
			else:
				stream = client.chat.completions.create(
					model=model,
					messages=messages,
					# stream=True,
					stream=is_stream,
					max_tokens=max_token, 
					temperature=temperature, top_p=top_p, stop=stop)
			message = ""
			
			if is_stream:
				for chunk in stream:
					if len(chunk.choices) != 0 and chunk.choices[0].delta.content is not None:
						print(chunk.choices[0].delta.content or "", end="", flush=True)
						message += chunk.choices[0].delta.content
				got_result = True
			else:
				message += stream.choices[0].message.content
				got_result = True
			
		except Exception:
			sleep(3)

	return message

def safe_execute(code_string: str, global_dict, local_dict, keys=None):
	ans = None
	# print(global_dict, local_dict)
	try:
		exec(code_string, global_dict, local_dict)
	except Exception as e:
		print(f"An error occurred: {e}")

	return ans

def redirect_stdout(code_to_execute, global_dict, local_dict):
	# Create a string buffer to capture the output
	buffer = io.StringIO()

	# Redirect the standard output to the buffer

	sys.stdout = buffer

	# Execute the code
	safe_execute(code_to_execute, global_dict, local_dict)

	# Reset the standard output to its original value
	sys.stdout = sys.__stdout__

	# Get the captured output from the buffer
	output = buffer.getvalue()

	# Close the buffer
	buffer.close()

	return output

def extract_code(response):

	code = ''

	# index = re.search('```', response)
	# index = index.span()

	index = [(match.start(), match.end()) for match in re.finditer('```', response, flags=re.IGNORECASE)]
	index = [i[0] for i in index]
	
	if len(index) % 2 != 0:
		# raise ValueError('Incorrect format of python code detected! Please check the reply from the model.')
		# drop the last one
		index = index[:-1]

	# it is possible that the same reply contains multiple code snippet
	for i in range(0, len(index), 2):
		if '[SUCCESS]' in response[index[i]:index[i]+10]:
			continue
		if 'python' in response[index[i]:index[i]+10] or \
			'Python' in response[index[i]:index[i]+10]:
			start = index[i]+10
		else:
			continue
			# return ""

		end = index[i+1]
		# if response[index[i]+3:index[i]+9] != 'python':
		# 	raise ValueError('The model should use python code')
		code += '\n' + response[start:end] + '\n'
	
	return code 

def iteration_program_output(output):
	if "An error occurred:" in output:
		# Print the captured output
		program_output = "The above program printed errors. Please fix it:\n" + output
	elif len(output) == 0:
		# program_output = "The above prorgam printed nothing. Please continue if it is meant to be the case."
		program_output = "The above code completed successfully or no code is written. If this is meant to be the case, state the keywords [RESULTS_START] and [RESULTS_END] and the iteration will stop."
	elif len(output) >= 2048:
		program_output = "The above program printed too lengthy output. I've cropped it to 4096 characters for you.\n" + output[:4096] 
	else:
		# Print the captured output
		program_output = "The above program printed:\n" + output
	program_output = ">>>>>>" + program_output
	print(program_output)
	return program_output

def write_to_csv_file(log_name, model, task, index, mse, mode):
	dir = f'./results/{model}/{mode}/'
	if not os.path.exists(dir):
		os.makedirs(dir)
	csv_file_path = dir + log_name + '.csv'
	# Try to append to the file if it exists, otherwise create it
	# Data to append
	data = [index, task, mse]
	if os.path.exists(csv_file_path):
		# Open the file in append mode
		with open(csv_file_path, 'a', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(data)
	else:
		# If the file does not exist, open it in write mode to create it and write the data
		with open(csv_file_path, 'w', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(['index', 'task', 'score'])
			writer.writerow(data)

def read_tracking_data(args, step, local_dict):
	idx = args.index
	if args.dataset == 'track_range_online':
		directory = "./data/tracking_range_0.01/"
	elif args.dataset == 'track_bearing_online':
		directory = "./data/tracking_bearing_0.01/"
	elif args.dataset  == 'track_range_bearing_online':
		directory = "./data/tracking_range_bearing_[0.01, 0.01]/"
	elif args.dataset == 'track_region_online':
		directory = "./data/tracking_region_4.0/"
	elif args.dataset == 'track_event_spatio_online':
		directory = "./data/tracking_event_spatio_0.01/"
	elif args.dataset == 'track_event_temp_online':
		directory = "./data/tracking_event_temp_0.01/"
	sensors_readings = []
	sensors_location = []
	sensor_loc_pd = pd.read_csv(directory + f'{idx}_sensor_location.csv')
	for i in range(4):
		sensor_i = pd.read_csv(directory + f'{idx}_sensor_{i}.csv')
		sensors_readings.append(sensor_i['d'].to_numpy())
		sensors_location.append(sensor_loc_pd.iloc[i].to_numpy())
	readings = np.array(sensors_readings).T
	coordinate = np.array(sensors_location)
	timestamp = sensor_i['time'].to_numpy()

	local_dict['readings'] = readings[:step+1,:]
	local_dict['coordinate'] = coordinate
	local_dict['timestamp'] = timestamp[:step+1]

	return (readings, coordinate, timestamp)

# def add_execution_string(args, returned_code):
	
# 	if 'track' in args.dataset:
# 		code_to_execute = """
# import numpy as np
# import scipy
# import filterpy
# 	"""
# 		code_to_execute += returned_code
# 		code_to_execute += "result=solver(readings, coordinate, timestamp)\n"
# 	else:
# 		code_to_execute = """
# import numpy as np
# import scipy
# 	"""
# 		code_to_execute += returned_code
# 		code_to_execute += "result=solver()\n"

# 	if args.dataset == 'loc_event_spatio_temp':
# 		code_to_execute += """print("[RESULTS_START] [{}, {}, {}] [RESULTS_END]".format(result[0], result[1], result[2]))\n"""
# 	elif args.dataset == 'loc_event_temp':
# 		code_to_execute += """print("[RESULTS_START] [{}] [RESULTS_END]".format(result))\n"""
# 	elif args.dataset == 'track_event_temp_online':
# 		code_to_execute += """print("[RESULTS_START] [{}] [RESULTS_END]".format(result))\n"""
# 	else:
# 		code_to_execute += """print("[RESULTS_START] [{}, {}] [RESULTS_END]".format(result[0], result[1]))\n"""
# 	code_to_execute +="print('The solver runs successfully.')\n"
# 	return code_to_execute

def add_execution_string(args, returned_code):
	
	if 'track' in args.dataset:
		code_to_execute = """
import numpy as np
import scipy
import filterpy
	"""
		code_to_execute += returned_code
		code_to_execute += "result=solver(readings, coordinate, timestamp)\n"
	else:
		code_to_execute = """
import numpy as np
import scipy
	"""
		code_to_execute += returned_code
		code_to_execute += "result=solver()\n"

	code_to_execute += """print("[RESULTS_START] [{}] [RESULTS_END]".format(result))\n"""
	code_to_execute +="print('The solver runs successfully.')\n"
	return code_to_execute
