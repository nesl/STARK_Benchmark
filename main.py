from agent import OpenAIAgent
import pdb
import os 
from arguments import get_arguments
from prompts.sys_promt_collections import AgentPromptText, AgentPromptCode, AgentPromptTrackOnlineCode
from get_data import load_st_dataset
from distance import compute_mse
from utils import write_to_csv_file, field_variable_tasks
from chat import Agent_with_APIs, Agent_text_based
import numpy as np 
from distance import string_to_np_array
import random
from params import NORMAL_TEMPERATURE
from sim_lib_s_v2 import spatial_relationship
from sim_lib_t_v2 import temporal_relationship
from sim_lib_st_v2 import obtain_spatio_temporal_relationship
import pandas as pd 

global_dict, local_dict = globals(), locals()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def execute_llm_solver(args, Agent, query_str, label_data):
	
	if args.mode == 'text':
		output = Agent_text_based(args, Agent, query_str)
	elif args.mode == 'code':
		output = Agent_with_APIs(args, Agent, query_str, global_dict, local_dict)
	else:
		raise NotImplementedError("Currently only support text- and code-based interaction!")
	
	try:
		rmspe = True if args.dataset in field_variable_tasks else False
		distance, array1 = compute_mse(args, output, label_data, rmspe=rmspe)
	except:
		distance, array1 = np.nan, np.nan

	if args.dataset == 'loc_event_spatio_temp':
		print('RMSE: {}'.format(distance))
	else:
		print('RMSE: {:.4f}'.format(distance))
	Agent.save_chat(trial=args.index, result=str(distance))
	if not args.dont_save_to_csv:
		write_to_csv_file(log_name='{}_{}_{}'.format(args.dataset, args.index, args.noise_level), 
					model=args.openai, task = args.dataset, index = args.index, mse = str(distance), mode=args.mode)
		dir = f'./results_npy/{args.openai}/{args.mode}/'
		if not os.path.exists(dir):
			os.makedirs(dir, exist_ok=True)
		np.save(dir+'{}_{}.csv'.format(args.dataset, args.index), array1)
	return output

def execute_llm_solver_online(args, Agent, input_data_str, query_str, label_data):
	
	output = []
	length = args.ts_len
	# current_temperature = np.random.uniform(10,30)
	
	for t in range(length):
		
		if args.dataset in ('track_event_spatio_online', 'track_event_temp_online', 'track_event_spatiotemp_online'):
			data_str_at_t = "Step {} | The corresponding sensor readings are {}. Return your estimation only for the current step.".format(
				t, input_data_str['sensor_readings'][t], 
			)
		else:
			data_str_at_t = "Step {} | At time {:.4f}, the sensor readings are {}. Return your estimation only for the current step.".format(
				t, input_data_str['timestamp'][t], input_data_str['sensor_readings'][t]
			)
		
		if 'track' in args.dataset and args.mode == 'code':
			from utils import read_tracking_data
			read_tracking_data(args, t, local_dict)

		is_append = True if t == 0 else False # Some model such as deepseek-r1 does not support successive user or assistent message
		Agent.update(data_str_at_t, role="user", append=is_append)

		if args.mode == 'text':
			output_str_at_t = Agent_text_based(args, Agent, query_str)
		elif args.mode == 'code':
			output_str_at_t = Agent_with_APIs(args, Agent, query_str, global_dict, local_dict)
		else:
			raise NotImplementedError("Currently only support text- and code-based interaction!")

		# write code to extract LLM output
		output_at_t = string_to_np_array(output_str_at_t)
		output.append(output_at_t)

	groundturh = label_data[:length,:].squeeze()

	try:
		rmspe = True if args.dataset in field_variable_tasks else False
		distance, _ = compute_mse(args, output, groundturh, string_to_array=False, rmspe=rmspe)
	except:
		distance = np.nan
	# print(f'GT: {groundturh}')
	if args.dataset == 'loc_event_spatio_temp':
		print('RMSE: {}'.format(distance))
	else:
		print('RMSE: {:.4f}'.format(distance))
	Agent.save_chat(trial=args.index, result=str(distance))
	if not args.dont_save_to_csv:
		write_to_csv_file(log_name='{}_{}'.format(args.dataset, args.index), 
						model=args.openai, task = args.dataset, index = args.index, mse = str(distance), mode=args.mode)
		dir = f'./results_npy/{args.openai}/{args.mode}/'
		if not os.path.exists(dir):
			os.makedirs(dir, exist_ok=True)
		np.save(dir+'{}_{}.csv'.format(args.dataset, args.index), output)
	return output

def select_instruction_prompts(args):
	if args.mode == 'text':
		return AgentPromptText
	elif args.mode == 'code':
		if 'track' in args.dataset:
			return AgentPromptTrackOnlineCode
		else:
			return AgentPromptCode
		# return AgentPromptCode
	else:
		raise NotImplementedError("Currently only support text- and code-based interaction!")

def save_2_csv(target_dir, name, query, answer):
	str_answer = str(answer.tolist())
	columns = ['query', 'answer']
	df_dict = pd.DataFrame([[query, str_answer]], columns=columns)
	if not os.path.exists(target_dir):
		os.makedirs(target_dir, exist_ok=True)
	df_dict.to_csv(target_dir+name)

def main(args):
	
	assert args.openai in ('gpt-3.5-turbo', 'gpt-4', 'gpt-4o', 'gpt-4-0125-preview', 'o1', \
		'o3-mini', 'gpt-4-turbo', 'gpt-4o-mini', 'gpt-4.1', 'o3', 'gpt-4.5-preview',\
		'Llama-2-70b', 'Llama-2-13b', 'Llama-2-7b', 'Llama-3-8b', 'Llama-3-70b', \
		'Qwen1.5-110B', 'Qwen2-72B', 'Llama-3.3-70B', 'DeepSeek-R1',\
		'Qwen2.5-Coder-32B', 'Qwen2.5-72B', 'deepseek-chat', 'deepseek-reasoner',
		'Qwen3-235B', 'Llama-4', 'o4-mini',
		'Mistral-7B')
	seed_everything(0)
	if args.openai == 'Llama-3.3-70B':
		prefix = 'meta-llama/'
		args.openai = prefix + args.openai + '-Instruct-Turbo-Free'
	elif args.openai == 'Llama-4':
		prefix = 'meta-llama/'
		args.openai = prefix + args.openai + '-Maverick-17B-128E-Instruct-FP8'
	elif 'Llama' in args.openai:
		prefix = 'meta-llama/'
		args.openai = prefix + args.openai + '-chat-hf'
	elif args.openai == 'Qwen2.5-72B':
		prefix = 'Qwen/'
		args.openai = prefix + args.openai + '-Instruct-Turbo'
	elif 'Qwen2' in args.openai:
		prefix = 'Qwen/'
		args.openai = prefix + args.openai + '-Instruct'
	elif 'Qwen3' in args.openai:
		prefix = 'Qwen/'
		args.openai = prefix + args.openai + '-A22B-fp8-tput'
	elif 'Mistral' in args.openai:
		prefix = 'mistralai/'
		args.openai = prefix + args.openai + '-Instruct-v0.2'
	
	if 'gpt' in args.openai or 'o1' in args.openai or 'o3' in args.openai or 'o4' in args.openai:
		openai_key = open("key.txt").read().strip()
	elif 'deepseek' in args.openai:
		openai_key = open("deepseek_key.txt").read().strip()
	else:
		openai_key = open("together_key.txt").read().strip()

	os.environ["OPENAI_API_KEY"] = openai_key

	AgentPrompt = select_instruction_prompts(args)
	Agent = OpenAIAgent(args, openai_key, system_prompt=
					AgentPrompt, temperature=1, top_p=1)
	# Agent.update(
	# 	"What's the date of man landing on moon?", role="user"
	# )
	# Agent.step()

	spatio_temporal_relationship = obtain_spatio_temporal_relationship()
	
	if args.dataset in ['loc_range', 'loc_bearing', 'loc_range_bearing', 'loc_region', 'loc_event_temp',\
			'loc_event_spatio', 'loc_event_spatio_temp']+\
			spatial_relationship + temporal_relationship + spatio_temporal_relationship +\
			['intent_pred', 'poi_pred'] +\
			['route_planning', 'landmark_questions', 'direction_questions', 'travel_questions', 'subroute_duration'] +\
			field_variable_tasks:
		input_data_str, query_str, label_data = load_st_dataset(args, args.input_seq, args.output_seq, spt_seq=None)
		# pdb.set_trace()
		if args.process_data:
			target_dir = f'./data_s/{args.dataset}/'
			name = f'{args.index}.csv'
			save_2_csv(target_dir, name, query_str, label_data)
		else:
			Agent.update(query_str, role="user")
			output = execute_llm_solver(args, Agent, query_str, label_data)

	elif args.dataset in ('track_range_online', 'track_bearing_online', 'track_range_bearing_online', 'track_region_online', 'track_event_spatio_online', 'track_event_temp_online', 'track_event_spatiotemp_online'):
		input_data_str, query_str, label_data = load_st_dataset(args, args.input_seq, args.output_seq, spt_seq=None)
		if args.process_data:
			target_dir = f'./data_s/{args.dataset}/'
			name = f'{args.index}.csv'
			save_2_csv(target_dir, name, query_str, label_data)
		else:
			Agent.update(query_str, role="user")
			output = execute_llm_solver_online(args, Agent, input_data_str, query_str, label_data)
	else:
		raise NotImplementedError


	

if __name__ == '__main__':
	args = get_arguments()
	main(args)