import os
import openai
import math
import sys
import time
from tqdm import tqdm
from typing import Iterable, List, TypeVar
from retrying import retry
import func_timeout
from func_timeout import func_set_timeout

T = TypeVar('T')
KEY_INDEX = 0
KEY_POOL = [
    os.environ["OPENAI_API_KEY"]
]
openai.api_key = KEY_POOL[0]


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("The function took too long to run")


@func_set_timeout(30)
def limited_execution_time(func, model, prompt, temp, max_tokens=128, default=None, **kwargs):
    try:
        if 'gpt-3.5-turbo' in model or 'gpt-4' in model:
            result = func(
                model=model,
                messages=prompt,
                temperature=temp
            )
        else:
            result = func(model=model, prompt=prompt, max_tokens=max_tokens, **kwargs)
    except func_timeout.exceptions.FunctionTimedOut:
        return None
    # raise any other exception
    except Exception as e:
        raise e
    return result


def batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    # function copied from allenai/real-toxicity-prompts
    assert batch_size > 0
    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []
        batch.append(item)

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch


def openai_unit_price(model_name, token_type="prompt"):
    if 'gpt-4' in model_name:
        if token_type == "prompt":
            unit = 0.03
        elif token_type == "completion":
            unit = 0.06
        else:
            raise ValueError("Unknown type")
    elif 'gpt-3.5-turbo' in model_name:
        unit = 0.002
    elif 'davinci' in model_name:
        unit = 0.02
    elif 'curie' in model_name:
        unit = 0.002
    elif 'babbage' in model_name:
        unit = 0.0005
    elif 'ada' in model_name:
        unit = 0.0004
    else:
        unit = -1
    return unit


def calc_cost_w_tokens(total_tokens: int, model_name: str):
    unit = openai_unit_price(model_name, token_type="completion")
    return round(unit * total_tokens / 1000, 4)


def calc_cost_w_prompt(total_tokens: int, model_name: str):
    # 750 words == 1000 tokens
    unit = openai_unit_price(model_name)
    return round(unit * total_tokens / 1000, 4)


def get_perplexity(logprobs):
    assert len(logprobs) > 0, logprobs
    return math.exp(-sum(logprobs) / len(logprobs))


def keep_logprobs_before_eos(tokens, logprobs):
    keep_tokens = []
    keep_logprobs = []
    start_flag = False
    for tok, lp in zip(tokens, logprobs):
        if start_flag:
            if tok == "<|endoftext|>":
                break
            else:
                keep_tokens.append(tok)
                keep_logprobs.append(lp)
        else:
            if tok != '\n':
                start_flag = True
                if tok != "<|endoftext>":
                    keep_tokens.append(tok)
                    keep_logprobs.append(lp)

    return keep_tokens, keep_logprobs


def catch_openai_api_error(prompt_input: list):
    global KEY_INDEX
    error = sys.exc_info()[0]
    if error == openai.error.InvalidRequestError:
        # something is wrong: e.g. prompt too long
        print(f"InvalidRequestError\nPrompt:\n\n{prompt_input}\n\n")
        assert False
    elif error == openai.error.RateLimitError:
        KEY_INDEX = (KEY_INDEX + 1) % len(KEY_POOL)
        openai.api_key = KEY_POOL[KEY_INDEX]
        print("RateLimitError", openai.api_key)
    elif error == openai.error.APIError:
        KEY_INDEX = (KEY_INDEX + 1) % len(KEY_POOL)
        openai.api_key = KEY_POOL[KEY_INDEX]
        print("APIError", openai.api_key)
    elif error == openai.error.AuthenticationError:
        KEY_INDEX = (KEY_INDEX + 1) % len(KEY_POOL)
        openai.api_key = KEY_POOL[KEY_INDEX]
        print("AuthenticationError", openai.api_key)
    elif error == TimeoutError:
        KEY_INDEX = (KEY_INDEX + 1) % len(KEY_POOL)
        openai.api_key = KEY_POOL[KEY_INDEX]
        print("TimeoutError, retrying...")
    else:
        print("API error:", error)


def prompt_gpt3(prompt_input: list, save_path, model_name='text-davinci-003', max_tokens=128,
                clean=False, batch_size=16, verbose=False, **kwargs):
    # return: output_list, money_cost

    def request_api(prompts: list):
        # prompts: list or str

        total_tokens = 0
        results = []
        for batch in tqdm(batchify(prompt_input, batch_size), total=len(prompt_input) // batch_size):
            batch_response = request_api(batch)
            total_tokens += batch_response['usage']['total_tokens']
            if not clean:
                results += batch_response['choices']
            else:
                results += [choice['text'] for choice in batch_response['choices']]
            with open(save_path, 'w+', encoding='utf-8') as f:
                for content in results:
                    content = content.replace("\n", " ")
                    f.write(content + '\n')
        return results, calc_cost_w_tokens(total_tokens, model_name)


def prompt_chatgpt(system_input, user_input, temperature, save_path, index, history=[], model_name='gpt-3.5-turbo'):
    '''
    :param system_input: "You are a helpful assistant/translator."
    :param user_input: you texts here
    :param history: ends with assistant output.
                    e.g. [{"role": "system", "content": xxx},
                          {"role": "user": "content": xxx},
                          {"role": "assistant", "content": "xxx"}]
    return: assistant_output, (updated) history, money cost
    '''
    if len(history) == 0:
        history = [{"role": "system", "content": system_input}]
    history.append({"role": "user", "content": user_input})
    while True:
        try:
            completion = limited_execution_time(openai.ChatCompletion.create,
                                                model=model_name,
                                                prompt=history,
                                                temp=temperature)
            if completion is None:
                raise TimeoutError
            break
        except:
            catch_openai_api_error(user_input)
            time.sleep(1)

    assistant_output = completion['choices'][0]['message']['content']
    history.append({"role": "assistant", "content": assistant_output})
    total_prompt_tokens = completion['usage']['prompt_tokens']
    total_completion_tokens = completion['usage']['completion_tokens']

    with open(save_path, 'a+', encoding='utf-8') as f:
        assistant_output = str(index) + "\t" + "\t".join(x for x in assistant_output.split("\n"))
        f.write(assistant_output + '\n')

    return assistant_output, history, calc_cost_w_tokens(total_prompt_tokens, model_name) + calc_cost_w_prompt(
        total_completion_tokens, model_name)
