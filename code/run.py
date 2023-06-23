import argparse
from openai_request import prompt_chatgpt
from prompt_preparation import build_zeroshot_prompt_strategyQA_with_singleSource_evidence
from tqdm import tqdm

parser = argparse.ArgumentParser()


parser.add_argument("-input", "--input_file", help="input json file", type=str, required=False,default='data/example_data.json')
parser.add_argument("-output", "--output_file", help="output path", type=str, required=False,default='dir/output.txt')
parser.add_argument('-model', "--model_name", help="openai model name", type=str, required=False, default='gpt-3.5-turbo-0301')

args = parser.parse_args()


if __name__ == '__main__':
    # please choose the different prompt generation func in different step.
    test_data = build_zeroshot_prompt_strategyQA_with_singleSource_evidence(args.input_file, mode="explicit", evidence_order='conf_para')
    total_price = 0
    for idx, prompt in enumerate(tqdm(test_data[:10])):
        if prompt == "":
            with open(args.output_file, 'a+', encoding='utf-8') as f:
                assistant_output = str(idx)
                f.write(assistant_output + '\n')
            continue
        results, _, price = prompt_chatgpt("You are a helpful assistant.", index=idx, save_path=args.output_file,
                                           user_input=prompt, model_name=args.model_name, temperature=0)
        total_price += price
    print(total_price)