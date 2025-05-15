import pdb 
from utils import extract_code, write_to_csv_file, redirect_stdout, iteration_program_output,\
	add_execution_string, guard_imports
import re 
from prompts.mode_prompts import no_code_feedback, no_keyword_feedback, verify_results

def Agent_with_APIs(args, agent, query_str, global_dict, local_dict):

	num_iter = 0
	max_iter = n = args.n
	failed = 0
	got_result = False
	reply = ""

	while (not got_result) and num_iter <= max_iter:
		num_iter += 1

		print(f'\n========> Mode: {args.mode} | Round {num_iter} starts...')

		program_output = ""

		reply = agent.step(stop=agent.stop)

		returned_code = extract_code(reply)

		new_text = re.sub('\n', '', returned_code, flags=re.IGNORECASE)
		if len(new_text) == 0:
			print('No code is detected in the current round!')
			agent.update(role = "user", content=no_code_feedback)
			output = ""
		else:
			# code_to_execute = returned_code
			code_to_execute = add_execution_string(args, returned_code)
			
			guardrail_pass, violation = guard_imports(code_to_execute)
			
			if not guardrail_pass:
				output = f"An error occurred: You've intended to use disallowed package: {violation}. Try again"
				agent.update(role = "user", content = output)
				return output

			# input_data, sampling_rate = read_data(args.input_file)
			output = redirect_stdout(code_to_execute, global_dict, local_dict)

			program_output = iteration_program_output(output)
			agent.update(role = "user", content = program_output)

		if "An error occurred:" in output:
			if failed >= 5:
				# too many faults occur in this implementation. skip it
				return reply
			failed += 1
			continue
		if "[RESULTS_START]" in program_output and "no code is written." not in program_output:
			got_result = True
			print("The result has been obtained or the max iter has been achieve.")
			return program_output 
		elif "[RESULTS_START]" in reply and "no code is written." in program_output:
			got_result = True
			print("The result has been obtained or the max iter has been achieve.")
			return reply 
		# elif "[RESULTS_START]" in reply and program_output == "":
		# 	got_result = True
		# 	print("The model indicates no code is needed and outputs results directly.")
		# 	return reply 

	return reply 

def Agent_text_based(args, agent, query_str):

	max_iter = args.n
	failed = 0
	got_result = False
	reply = ""

	got_result = False

	iter_num = 0
	while not got_result and iter_num <= max_iter:

		iter_num += 1

		print(f'\n========> Mode: {args.mode} | Round {iter_num} starts...')

		reply = agent.step(stop=agent.stop)

		if "[RESULTS_START]" not in reply:

			reply = agent.update(
				no_keyword_feedback, role="user"
			)
			continue
		else:
			# agent.save_chat(trial=num_iter)
			return reply 
	return reply 