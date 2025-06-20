import copy
import glob
import json
import os
import argparse
from pathlib import Path
from datetime import datetime
import random
import subprocess

import openai
import ai2thor.controller

import sys
sys.path.append(".")

import resources.actions as actions
import resources.robots as robots


import torch
from transformers import pipeline

# The example given by huggingg face
# def LM():
#     
#     chat = [
#         {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
#         {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
#     ]

#     pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
#     response = pipeline(chat, max_new_tokens=512)
#     print(response[0]["generated_text"][-1]["content"])

# rewritten to match the SMART_LLM
def LM(prompt, gpt_version="meta-llama/Meta-Llama-3-8B-Instruct", max_tokens=128, temperature=0.7, stop=None, logprobs=None, frequency_penalty=0):
    """
    Generates text using the Hugging Face pipeline, mimicking OpenAI's Completion and ChatCompletion.

    Args:
        prompt (str or list): The input prompt. If 'gpt' is in gpt_version, it expects a list of messages
                              like [{"role": "user", "content": "Hello!"}]. Otherwise, a string.
        gpt_version (str): The Hugging Face model identifier (e.g., "meta-llama/Meta-Llama-3-8B-Instruct").
                           If "gpt" is in the name, it will treat the input as a chat prompt.
        max_tokens (int): The maximum number of new tokens to generate.
        temperature (float): Controls the randomness of the output. Higher values mean more random.
        stop (list, optional): A list of strings that, if encountered, will stop the generation.
        logprobs (int, optional): Not directly supported in the same way by all Hugging Face pipelines
                                   for simple generation, but included for API compatibility.
        frequency_penalty (float): Not directly supported in the same way by all Hugging Face pipelines
                                   for simple generation, but included for API compatibility.

    Returns:
        tuple: A tuple containing:
               - response (list): The raw response from the Hugging Face pipeline.
               - generated_text (str): The extracted generated text.
    """

    # Initialize the pipeline
    # device_map="auto" will automatically place the model on available devices (e.g., GPU)
    # torch_dtype=torch.bfloat16 is good for newer GPUs to save memory and speed up computation
    model_pipeline = pipeline(
        task="text-generation",
        model=gpt_version,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    generation_args = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        # "do_sample": True if temperature > 0 else False, # Enable sampling if temperature is not 0
    }

    if stop:
        # Handling 'stop' words in Hugging Face pipelines is typically done via custom stopping criteria.
        # For simplicity in this direct translation, it's not directly mapped for all models,
        # as it requires more advanced tokenizer integration.
        # If crucial, you would implement a custom stopping criteria callback.
        print("Warning: 'stop' parameter might not be fully supported by all Hugging Face models directly via pipeline generation arguments. Custom stopping criteria may be needed.")
        pass # Placeholder for more sophisticated stop word handling if needed

    # The logprobs and frequency_penalty parameters are not directly available as simple
    # generation arguments in the Hugging Face pipeline in the same way as OpenAI.
    # For logprobs, you might need to access the model's outputs more directly or use
    # a different task like 'fill-mask' or custom model inference.
    # For frequency_penalty, it's typically handled by custom token samplers.
    if logprobs is not None:
        print("Warning: 'logprobs' parameter is not directly supported by the Hugging Face text-generation pipeline in the same way as OpenAI.")
    if frequency_penalty != 0:
        print("Warning: 'frequency_penalty' parameter is not directly supported by the Hugging Face text-generation pipeline in the same way as OpenAI.")


    if "gpt" not in gpt_version.lower() and not isinstance(prompt, list):
        # Treat as a simple text completion if "gpt" is not in version name and prompt is a string
        # This mirrors openai.Completion.create
        response = model_pipeline(prompt, **generation_args)
        generated_text = response[0]["generated_text"].strip()
        # For completion, the prompt is part of the generated_text, so we need to remove it
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
    else:
        # Treat as a chat completion if "gpt" is in version name or prompt is a list of messages
        # This mirrors openai.ChatCompletion.create
        if not isinstance(prompt, list):
            raise ValueError("For 'chat' style models (like GPT variants or when 'gpt' is in gpt_version), 'prompt' must be a list of message dictionaries.")

        response = model_pipeline(prompt, **generation_args)
        # For chat models, the output is typically the last turn's content
        # The structure can vary slightly by model, but usually it's in the last dict's 'content'
        if response and response[0] and "generated_text" in response[0] and len(response[0]["generated_text"]) > 0:
            # The generated_text for chat models often contains the entire conversation,
            # so we need to extract only the last assistant's reply.
            # Assuming the last element of generated_text is the assistant's response.
            # This can be model-dependent. For Llama-3, the structure is usually
            # a list of dictionaries with 'role' and 'content'.
            full_chat_history = response[0]["generated_text"]
            last_message = full_chat_history[-1]
            if isinstance(last_message, dict) and "content" in last_message:
                generated_text = last_message["content"].strip()
            else:
                # Fallback if the structure is not exactly as expected
                generated_text = str(last_message).strip()
        else:
            generated_text = "" # No content generated

    return response, generated_text

# def LM(prompt, gpt_version, max_tokens=128, temperature=0, stop=None, logprobs=1, frequency_penalty=0):
    
#     if "gpt" not in gpt_version:
#         response = openai.Completion.create(model=gpt_version, 
#                                             prompt=prompt, 
#                                             max_tokens=max_tokens, 
#                                             temperature=temperature, 
#                                             stop=stop, 
#                                             logprobs=logprobs, 
#                                             frequency_penalty = frequency_penalty)
        
#         return response, response["choices"][0]["text"].strip()
    
#     else:
#         response = openai.ChatCompletion.create(model=gpt_version, 
#                                             messages=prompt, 
#                                             max_tokens=max_tokens, 
#                                             temperature=temperature, 
#                                             frequency_penalty = frequency_penalty)
        
#         return response, response["choices"][0]["message"]["content"].strip()

def set_api_key(openai_api_key):
    openai.api_key = Path(openai_api_key + '.txt').read_text()

# Function returns object list with name and properties.
def convert_to_dict_objprop(objs, obj_mass):
    objs_dict = []
    for i, obj in enumerate(objs):
        obj_dict = {'name': obj , 'mass' : obj_mass[i]}
        # obj_dict = {'name': obj , 'mass' : 1.0}
        objs_dict.append(obj_dict)
    return objs_dict

def get_ai2_thor_objects(floor_plan_id):
    # connector to ai2thor to get object list
    controller = ai2thor.controller.Controller(scene="FloorPlan"+str(floor_plan_id))
    obj = list([obj["objectType"] for obj in controller.last_event.metadata["objects"]])
    obj_mass = list([obj["mass"] for obj in controller.last_event.metadata["objects"]])
    controller.stop()
    obj = convert_to_dict_objprop(obj, obj_mass)
    return obj

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--floor-plan", type=int, default="6")
    parser.add_argument("--openai-api-key-file", type=str, default="api_key")
    parser.add_argument("--gpt-version", type=str, default="gpt-4", 
                        choices=['gpt-3.5-turbo', 'gpt-4', 'gpt-3.5-turbo-16k'])
    
    parser.add_argument("--prompt-decompse-set", type=str, default="train_task_decompose", 
                        choices=['train_task_decompose'])
    
    parser.add_argument("--prompt-allocation-set", type=str, default="train_task_allocation", 
                        choices=['train_task_allocation'])
    
    parser.add_argument("--test-set", type=str, default="final_test", 
                        choices=['final_test'])
    
    parser.add_argument("--log-results", type=bool, default=True)
    
    args = parser.parse_args()

    # set_api_key(args.openai_api_key_file)
    
    if not os.path.isdir(f"./logs/"):
        os.makedirs(f"./logs/")
        
    # read the tasks        
    test_tasks = []
    robots_test_tasks = []  
    gt_test_tasks = []    
    trans_cnt_tasks = []
    max_trans_cnt_tasks = []  
    with open (f"./data/{args.test_set}/FloorPlan{args.floor_plan}.json", "r") as f:
        for line in f.readlines():
            test_tasks.append(list(json.loads(line).values())[0])
            robots_test_tasks.append(list(json.loads(line).values())[1])
            gt_test_tasks.append(list(json.loads(line).values())[2])
            trans_cnt_tasks.append(list(json.loads(line).values())[3])
            max_trans_cnt_tasks.append(list(json.loads(line).values())[4])
                    
    print(f"\n----Test set tasks----\n{test_tasks}\nTotal: {len(test_tasks)} tasks\n")
    # prepare list of robots for the tasks
    available_robots = []
    for robots_list in robots_test_tasks:
        task_robots = []
        for i, r_id in enumerate(robots_list):
            rob = robots.robots [r_id-1]
            # rename the robot
            rob['name'] = 'robot' + str(i+1)
            task_robots.append(rob)
        available_robots.append(task_robots)
        
    
    ######## Train Task Decomposition ########
        
    # prepare train decompostion demonstration for ai2thor samples
    prompt = f"from skills import " + actions.ai2thor_actions
    prompt += f"\nimport time"
    prompt += f"\nimport threading"
    objects_ai = f"\n\nobjects = {get_ai2_thor_objects(args.floor_plan)}"
    prompt += objects_ai
    
    # read input train prompts
    decompose_prompt_file = open(os.getcwd() + "/data/pythonic_plans/" + args.prompt_decompse_set + ".py", "r")
    decompose_prompt = decompose_prompt_file.read()
    decompose_prompt_file.close()
    
    prompt += "\n\n" + decompose_prompt
    
    print ("Generating Decompsed Plans...")
    
    decomposed_plan = []
    for task in test_tasks:
        curr_prompt =  f"{prompt}\n\n# Task Description: {task}"
        
        if "gpt" not in args.gpt_version:
            # older gpt versions
            _, text = LM(curr_prompt, args.gpt_version, max_tokens=1000, stop=["def"], frequency_penalty=0.15)
        else:            
            messages = [{"role": "user", "content": curr_prompt}]
            _, text = LM(messages,args.gpt_version, max_tokens=1300, frequency_penalty=0.0)

        decomposed_plan.append(text)
        
    print ("Generating Allocation Solution...")

    ######## Train Task Allocation - SOLUTION ########
    prompt = f"from skills import " + actions.ai2thor_actions
    prompt += f"\nimport time"
    prompt += f"\nimport threading"
    
    prompt_file = os.getcwd() + "/data/pythonic_plans/" + args.prompt_allocation_set + "_solution.py"
    allocated_prompt_file = open(prompt_file, "r")
    allocated_prompt = allocated_prompt_file.read()
    allocated_prompt_file.close()
    
    prompt += "\n\n" + allocated_prompt + "\n\n"
    
    allocated_plan = []
    for i, plan in enumerate(decomposed_plan):
        no_robot  = len(available_robots[i])
        curr_prompt = prompt + plan
        curr_prompt += f"\n# TASK ALLOCATION"
        curr_prompt += f"\n# Scenario: There are {no_robot} robots available, The task should be performed using the minimum number of robots necessary. Robots should be assigned to subtasks that match its skills and mass capacity. Using your reasoning come up with a solution to satisfy all contraints."
        curr_prompt += f"\n\nrobots = {available_robots[i]}"
        curr_prompt += f"\n{objects_ai}"
        curr_prompt += f"\n\n# IMPORTANT: The AI should ensure that the robots assigned to the tasks have all the necessary skills to perform the tasks. IMPORTANT: Determine whether the subtasks must be performed sequentially or in parallel, or a combination of both and allocate robots based on availablitiy. "
        curr_prompt += f"\n# SOLUTION  \n"

        if "gpt" not in args.gpt_version:
            # older versions of GPT
            _, text = LM(curr_prompt, args.gpt_version, max_tokens=1000, stop=["def"], frequency_penalty=0.65)
        
        elif "gpt-3.5" in args.gpt_version:
            # gpt 3.5 and its variants
            messages = [{"role": "user", "content": curr_prompt}]
            _, text = LM(messages, args.gpt_version, max_tokens=1500, frequency_penalty=0.35)
        
        else:          
            # gpt 4.0
            messages = [{"role": "system", "content": "You are a Robot Task Allocation Expert. Determine whether the subtasks must be performed sequentially or in parallel, or a combination of both based on your reasoning. In the case of Task Allocation based on Robot Skills alone - First check if robot teams are required. Then Ensure that robot skills or robot team skills match the required skills for the subtask when allocating. Make sure that condition is met. In the case of Task Allocation based on Mass alone - First check if robot teams are required. Then Ensure that robot mass capacity or robot team combined mass capacity is greater than or equal to the mass for the object when allocating. Make sure that condition is met. In both the Task Task Allocation based on Mass alone and Task Allocation based on Skill alone, if there are multiple options for allocation, pick the best available option by reasoning to the best of your ability."},{"role": "system", "content": "You are a Robot Task Allocation Expert"},{"role": "user", "content": curr_prompt}]
            _, text = LM(messages, args.gpt_version, max_tokens=400, frequency_penalty=0.69)

        allocated_plan.append(text)
    
    print ("Generating Allocated Code...")
    
    ######## Train Task Allocation - CODE Solution ########

    prompt = f"from skills import " + actions.ai2thor_actions
    prompt += f"\nimport time"
    prompt += f"\nimport threading"
    prompt += objects_ai
    
    code_plan = []

    prompt_file1 = os.getcwd() + "/data/pythonic_plans/" + args.prompt_allocation_set + "_code.py"
    code_prompt_file = open(prompt_file1, "r")
    code_prompt = code_prompt_file.read()
    code_prompt_file.close()
    
    prompt += "\n\n" + code_prompt + "\n\n"

    for i, (plan, solution) in enumerate(zip(decomposed_plan,allocated_plan)):
        curr_prompt = prompt + plan
        curr_prompt += f"\n# TASK ALLOCATION"
        curr_prompt += f"\n\nrobots = {available_robots[i]}"
        curr_prompt += solution
        curr_prompt += f"\n# CODE Solution  \n"
        
        if "gpt" not in args.gpt_version:
            # older versions of GPT
            _, text = LM(curr_prompt, args.gpt_version, max_tokens=1000, stop=["def"], frequency_penalty=0.30)
        else:            
            # using variants of gpt 4 or 3.5
            messages = [{"role": "system", "content": "You are a Robot Task Allocation Expert"},{"role": "user", "content": curr_prompt}]
            _, text = LM(messages, args.gpt_version, max_tokens=1400, frequency_penalty=0.4)

        code_plan.append(text)
    
    # save generated plan
    exec_folders = []
    if args.log_results:
        line = {}
        now = datetime.now() # current date and time
        date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
        
        for idx, task in enumerate(test_tasks):
            task_name = "{fxn}".format(fxn = '_'.join(task.split(' ')))
            task_name = task_name.replace('\n','')
            folder_name = f"{task_name}_plans_{date_time}"
            exec_folders.append(folder_name)
            
            os.mkdir("./logs/"+folder_name)
     
            with open(f"./logs/{folder_name}/log.txt", 'w') as f:
                f.write(task)
                f.write(f"\n\nGPT Version: {args.gpt_version}")
                f.write(f"\n\nFloor Plan: {args.floor_plan}")
                f.write(f"\n{objects_ai}")
                f.write(f"\nrobots = {available_robots[idx]}")
                f.write(f"\nground_truth = {gt_test_tasks[idx]}")
                f.write(f"\ntrans = {trans_cnt_tasks[idx]}")
                f.write(f"\nmax_trans = {max_trans_cnt_tasks[idx]}")

            with open(f"./logs/{folder_name}/decomposed_plan.py", 'w') as d:
                d.write(decomposed_plan[idx])
                
            with open(f"./logs/{folder_name}/allocated_plan.py", 'w') as a:
                a.write(allocated_plan[idx])
                
            with open(f"./logs/{folder_name}/code_plan.py", 'w') as x:
                x.write(code_plan[idx])
            