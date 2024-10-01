import ray
from ray.rllib.agents.ppo import PPOTrainer

import argparse
import logging
import os
import sys
import time
import pickle

from utils import remote, saving
import tf_models
import yaml
from env_wrapper import RLlibEnvWrapper

path_to_data = 'one-step-economy/ai-economist-considerate-democracy/purely_egoistic/' # TODO path of training

model_name = 'con-dem-aie-00' # TODO 'saez' or 'aie' or 'ego-dem-aie' or 'con-dem-aie-[utilitarian_weight]' (f.e. 'con-dem-aie-50')

model_desc = 'Considerate-Democratic AI Economist 00' # TODO 'Saez Formula' or 'AI Economist' or 'Ego-Democratic AI Economist' or 'Considerate-Democratic AI Economist 50' (or similar -> need to add plot_cols)

multiple_runs = 100 # TODO if more than 1 test-run per checkpoint is required

config_path = os.path.join(path_to_data, 'config.yaml') # path to config.yaml

# Initialize Ray
ray.init(webui_host="127.0.0.1")

def generate_rollout_from_current_trainer_policy(
    trainer, 
    env_obj,
    num_dense_logs=1
):
    dense_logs = {}
    for idx in range(num_dense_logs):
        # Set initial states
        agent_states = {}
        for agent_idx in range(env_obj.env.n_agents):
            agent_states[str(agent_idx)] = trainer.get_policy("a").get_initial_state()
        planner_states = trainer.get_policy("p").get_initial_state()  

        # Play out the episode
        obs = env_obj.reset(force_dense_logging=True)
        for t in range(env_obj.env.episode_length):
            actions = {}
            for agent_idx in range(env_obj.env.n_agents):
                # Use the trainer object directly to sample actions for each agent
                actions[str(agent_idx)] = trainer.compute_action(
                    obs[str(agent_idx)], 
                    agent_states[str(agent_idx)], 
                    policy_id="a",
                    full_fetch=False
                )

            # Action sampling for the planner
            actions["p"] = trainer.compute_action(
                obs['p'], 
                planner_states, 
                policy_id='p',
                full_fetch=False
            )

            obs, rew, done, info = env_obj.step(actions)        
            if done['__all__']:
                break
        dense_logs[idx] = env_obj.env.dense_log
    return dense_logs

with open(config_path, "r") as f:
        run_configuration = yaml.safe_load(f)
        
trainer_config = run_configuration.get("trainer")

# === Env ===
env_config = {
    "env_config_dict": run_configuration.get("env"),
    "num_envs_per_worker": trainer_config.get("num_envs_per_worker"),
}

# === Seed ===
if trainer_config["seed"] is None:
    try:
        start_seed = int(run_configuration["metadata"]["launch_time"])
    except KeyError:
        start_seed = int(time.time())
else:
    start_seed = int(trainer_config["seed"])

final_seed = int(start_seed % (2 ** 16)) * 1000

# === Multiagent Policies ===
dummy_env = RLlibEnvWrapper(env_config)

# Policy tuples for agent/planner policy types
agent_policy_tuple = (
    None,
    dummy_env.observation_space,
    dummy_env.action_space,
    run_configuration.get("agent_policy"),
)
planner_policy_tuple = (
    None,
    dummy_env.observation_space_pl,
    dummy_env.action_space_pl,
    run_configuration.get("planner_policy"),
)

policies = {"a": agent_policy_tuple, "p": planner_policy_tuple}

def policy_mapping_fun(i):
    if str(i).isdigit() or i == "a":
        return "a"
    return "p"

# Which policies to train
if run_configuration["general"]["train_planner"]:
    policies_to_train = ["a", "p"]
else:
    policies_to_train = ["a"]

# === Finalize and create ===
trainer_config.update(
    {
        "env_config": env_config,
        "seed": final_seed,
        "multiagent": {
            "policies": policies,
            "policies_to_train": policies_to_train,
            "policy_mapping_fn": policy_mapping_fun,
        },
        "metrics_smoothing_episodes": trainer_config.get("num_workers")
        * trainer_config.get("num_envs_per_worker"),
    }
)

trainer = PPOTrainer(
    env=RLlibEnvWrapper, config=trainer_config
)

results = [
    {
    'Inverse-Income Weighted Utility' : [],
    'Income Equality * Productivity': [],
    'Income Equality': [],
    'Productivity': [],
    'Avg. Bracket Rate: 0': [], 
    'Avg. Bracket Rate: 9': [], 
    'Avg. Bracket Rate: 39': [], 
    'Avg. Bracket Rate: 84': [], 
    'Avg. Bracket Rate: 160': [], 
    'Avg. Bracket Rate: 204': [], 
    'Avg. Bracket Rate: 510': [],
    'Environment Steps': []
    } for i in range(multiple_runs)]

def search_ckpts(rootdir):
    file_list = []
    for _, _, files in os.walk(rootdir):
        for file in files:
            if(file.startswith("agent.policy-model-weight-array")):
                file_list.append(file)
    return file_list

ckpts_list = search_ckpts(str(path_to_data) + 'ckpts/')
step_list = [int(elem.lstrip("agent.policy-model-weight-array.global-step-")) for elem in ckpts_list]
step_list.sort()
                
for step in step_list: # Iterate through all checkpoints
    print(step)
    starting_weights_path_agents = path_to_data + 'ckpts/' + "agent.tf.weights.global-step-" + str(step)
    saving.load_tf_model_weights(trainer, starting_weights_path_agents)
            
    starting_weights_path_planner = path_to_data + 'ckpts/' + "planner.tf.weights.global-step-" + str(step)
    saving.load_tf_model_weights(trainer, starting_weights_path_planner)
    
    for i in range(multiple_runs):
    
        dense_logs = generate_rollout_from_current_trainer_policy(
            trainer, 
            dummy_env,
            num_dense_logs=1
        )
        
        results[i]['Environment Steps'].append(step)
        
        for metric in results[i].keys():
            if metric == 'Inverse-Income Weighted Utility':
                results[i][metric].append(dummy_env.env.metrics['social_welfare/inv_income_weighted_utility'])
            elif metric == 'Income Equality * Productivity':
                results[i][metric].append(dummy_env.env.metrics['social_welfare/coin_eq_times_productivity'])
            elif metric == 'Income Equality':
                results[i][metric].append(dummy_env.env.metrics['social/equality'])
            elif metric == 'Productivity':
                results[i][metric].append(dummy_env.env.metrics['social/productivity'])
            elif metric == 'Avg. Bracket Rate: 0':
                results[i][metric].append(dummy_env.env.metrics['PeriodicTax/avg_bracket_rate/000'])
            elif metric == 'Avg. Bracket Rate: 9':
                results[i][metric].append(dummy_env.env.metrics['PeriodicTax/avg_bracket_rate/097'])
            elif metric == 'Avg. Bracket Rate: 39':
                results[i][metric].append(dummy_env.env.metrics['PeriodicTax/avg_bracket_rate/394'])
            elif metric == 'Avg. Bracket Rate: 84':
                results[i][metric].append(dummy_env.env.metrics['PeriodicTax/avg_bracket_rate/842'])
            elif metric == 'Avg. Bracket Rate: 160':
                results[i][metric].append(dummy_env.env.metrics['PeriodicTax/avg_bracket_rate/1607'])
            elif metric == 'Avg. Bracket Rate: 204':
                results[i][metric].append(dummy_env.env.metrics['PeriodicTax/avg_bracket_rate/2041'])
            elif metric == 'Avg. Bracket Rate: 510':
                results[i][metric].append(dummy_env.env.metrics['PeriodicTax/avg_bracket_rate/5103'])
    
    print(dummy_env.env.metrics)
            
packed_results = {
    model_name : {
        'type' : model_desc,
        'data' : results
    }
}

result_path = path_to_data + "/training_history.pkl"
with open(result_path, "wb") as F:
    pickle.dump(packed_results, F)
    
# Shutdown Ray after use
ray.shutdown()