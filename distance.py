import numpy as np
import ast
import re 
import pdb 
import numbers

def string_to_np_array(string):
    """Converts a string representation of a list of lists into a NumPy array."""
    try:
        # Use regex to extract the content between [RESULTS_START] and [RESULTS_END]
        match = re.search(r"\[RESULTS_START\](.*?)\[RESULTS_END\]", string, re.DOTALL)
        if not match:
            raise ValueError("No valid results found in the string.")

        # Extract and clean up the matched string
        extracted_str = match.group(1).strip()
        
        # Standardize delimiters
        extracted_str = re.sub(r"\s+", ",", extracted_str)  # Replace spaces with commas

        # Ensure commas are properly formatted (remove excess commas)
        extracted_str = re.sub(r",+", ",", extracted_str)  # Remove redundant commas
        extracted_str = re.sub(r"\[,", "[", extracted_str)  # Avoid leading commas in lists
        extracted_str = re.sub(r",\]", "]", extracted_str)  # Avoid trailing commas in lists

        # Remove all characters except numbers, '.', '[', and ']'
        extracted_str = re.sub(r"[^\d\.\[\]\,]", "", extracted_str)

        array = np.array(ast.literal_eval(extracted_str))
        if array.size == 0:
            return np.array([np.nan])
        return array
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing string: {e}")
        return np.array([np.nan])

def handle_nan_values(args, array):
    if 'loc_' in args.dataset or 'track_' in args.dataset:
        if args.dataset == 'loc_event_temp' or args.dataset == 'track_event_temp_online':
            return np.array([5])
        else:
            return np.array([5, 5])
    return array

def compute_mse(args, array1_str, array2, string_to_array=True, rmspe=False):
    # array2 is ground truth
    if string_to_array:
        array1 = string_to_np_array(array1_str)
        has_nan = np.isnan(array1).any()
        if has_nan:
            array1 = handle_nan_values(args, array1)
        array2 = array2.T
    else:
        if not isinstance(array1_str, list):
            array1_str = [array1_str]
        for i, arr in enumerate(array1_str):
            has_nan = np.isnan(arr).any()
            if has_nan:
                array1_str[i] = handle_nan_values(args, arr)
            # handle shape inconsistence
            if array1_str[i].shape == ():  # scalar
                array1_str[i] = array1_str[i].reshape(1, 1)
            elif array1_str[i].size > 1:
                array1_str[i] = array1_str[i].squeeze()
        array1 = np.array(array1_str)

    if args.dataset == 'loc_event_temp':
        return np.sqrt(((array1 - array2)**2).mean()), None
    elif args.dataset == 'loc_event_spatio_temp':
        d_mse = np.sqrt(((array1[:2] - array2[:2])**2).mean())
        t_mse = np.sqrt(((array1[2] - array2[2])**2).mean())
        return (d_mse, t_mse), None

    # handle shape issue
    if array1.shape == ():  # scalar
        array1 = array1.reshape(1, 1)
    elif array1.size > 1:
        array1 = array1.squeeze()

    # print(array1, array2)
    print(f'LLM: {array1}')
    print(f'GT: {array2}')
    if len(array1) < len(array2):
        print("Invalid prediction. Return None.")
        return None, None
    elif len(array1) > len(array2):
        array1 = array1[:len(array2)]
    if rmspe:
        if np.linalg.norm(array2) <= 1e-10:
            dist = np.sqrt(((array1 - array2)**2).mean())
        else:
            # dist = np.sqrt((((array1 - array2)/(array2+1e-10))**2).mean()) # avoid being divided by zero
            dist = np.linalg.norm(array1 - array2) / np.linalg.norm(array2)
    else:
        dist = np.sqrt(((array1 - array2)**2).mean())
    return dist, array1

if __name__ == '__main__':
    string = """
    [RESULTS_START]
    [[23.38, 23.38],[23.06, 23.00],[24.12, 24.06],[23.50, 23.56],[22.75, 22.88],
    [22.44, 22.50],[22.69, 22.75],[22.88, 22.94],[22.19, 22.12],[22.06, 22.19],
    [22.19, 22.38]]
    [RESULTS_END]
    """
    np_array = string_to_np_array(string)
    print(np_array)