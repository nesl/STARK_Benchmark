import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    # default it to use gpt-4o
    parser.add_argument(
        "--openai", type=str, default='gpt-4o', help="default use gpt-4 model"
    )
    parser.add_argument(
        "--input_seq", type=int, default=72, help="Time series sequence length"
    )
    parser.add_argument(
        "--output_seq", type=int, default=2, help="Time series sequence length"
    )
    parser.add_argument(
        "--system_prompt_file", type=str, default='./system_prompt.txt', help="default system prompt"
    )
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument(
        "--mode", type=str, default='text', help="Conversational AI mode"
    )
    parser.add_argument(
        "--dataset", type=str, default="BLDG", help="ST dataset use for evaluation"
    )
    parser.add_argument(
        "--index", type=int, default=0, help="The index to retrieve the data")
    parser.add_argument(
        "--file", type=str, default=None, help="-"
    )
    parser.add_argument(
        "--ts_len", type=int, default=10, help="Time series sequence length. Perticularly used for online evaluation."
    )
    parser.add_argument(
        "--dont_save_to_csv",
        action="store_true",
        help="write results to csv file",
    )
    parser.add_argument(
        "--process_data",
        action="store_true",
        help="Not running experiments. Converting data to csv.",
    )
    parser.add_argument(
        "--query", type=str, default=None, help="user's query for testing"
    )
    parser.add_argument(
        "--n",
        type=int, default=5, help="""
        the number of trial if error happens
        """
    )
    parser.add_argument(
        "--eval",
        type=str, default='env', help="""
        Feedback from the environment or self-generated. (env | self_vis | self_coding | self_verifier)
        """
    )
    parser.add_argument(
        "--num_trial",
        type=int, default=1, help="""
        How many times can the model reflect and retry
        """
    )
    parser.add_argument(
        "--base_url", type=str, default="https://api.together.xyz/v1", help="together.ai interface"
    )
    parser.add_argument(
        "--log_name", type=str, default="test", help="The type of task we are testing."
    )
    parser.add_argument(
        "--noise_level", type=str, default="normal", help="The magnitude of noise."
    )
    args = parser.parse_args()
    return args