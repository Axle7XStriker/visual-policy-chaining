import argparse
import pickle
from tqdm import tqdm
import numpy as np
import cv2
import d3rlpy
from sklearn.model_selection import train_test_split

def load_data(file_path):
    print("Loading Data...")
    with open(file_path, 'rb') as f:
        rollouts_data = pickle.load(f)
    return rollouts_data


def extract_data(rollouts_data, config=None):
    print("Extracting Observations, Actions, Rewards, and Terminals from the raw data...")
    data_count = 0
    for ep in rollouts_data:
        data_count = data_count + len(ep['actions'])
    
    ob_shape = rollouts_data[0]['ob_images'][0].shape
    ac_shape = rollouts_data[0]['actions'][0]['default'].shape
    subtasks_data = {
            'observations': np.empty([data_count, config.obs_height, config.obs_width, 3], dtype=np.uint8), 
            'actions': np.empty([data_count, *ac_shape]),
            'rewards': np.empty([data_count]),
            'terminals': np.empty([data_count])
        } 
    
    k = 0
    for ep in rollouts_data:
        for i in range(len(ep['actions'])):
            subtasks_data['observations'][k] = cv2.resize(
                ep['ob_images'][i], dsize=(config.obs_height, config.obs_width), interpolation=cv2.INTER_CUBIC)
            subtasks_data['actions'][k] = ep['actions'][i]['default']
            subtasks_data['rewards'][k] = ep['rewards'][i]
            subtasks_data['terminals'][k] = ep['dones'][i]
            k = k + 1

    subtasks_data['observations'] = subtasks_data['observations'].transpose([0, 3, 1, 2])
    
    return subtasks_data


def extract_subtasks_data(rollouts_data, num_subtasks=1, config=None):
    print("Extracting Observations, Actions, Rewards, and Terminals from the raw data...")
    subtasks_data_count = [0 for i in range(num_subtasks)]
    for ep in rollouts_data:
        for i in range(len(ep['subtasks'])):
            subtasks_data_count[ep['subtasks'][i]] = subtasks_data_count[ep['subtasks'][i]] + 1
    
    ob_shape = rollouts_data[0]['ob_images'][0].shape
    ac_shape = rollouts_data[0]['actions'][0]['default'].shape
    subtasks_data = [
        {
            'observations': np.empty([subtasks_data_count[i], config.obs_height, config.obs_width, 3], dtype=np.uint8), 
            'actions': np.empty([subtasks_data_count[i], *ac_shape]),
            'rewards': np.empty([subtasks_data_count[i]]),
            'terminals': np.empty([subtasks_data_count[i]])
        } for i in range(num_subtasks)]
    
    subtask_k = [0 for i in range(num_subtasks)]
    for ep in rollouts_data:
        last_subtask = 0
        for i in range(len(ep['subtasks'])):
            subtask = ep['subtasks'][i]
            if subtask > last_subtask:
                subtasks_data[last_subtask]['terminals'][subtask_k[last_subtask]-1] = 1
            subtasks_data[subtask]['observations'][subtask_k[subtask]] = cv2.resize(
                ep['ob_images'][i], dsize=(config.obs_height, config.obs_width), interpolation=cv2.INTER_CUBIC)
            subtasks_data[subtask]['actions'][subtask_k[subtask]] = ep['actions'][i]['default']
            subtasks_data[subtask]['rewards'][subtask_k[subtask]] = ep['rewards'][i]
            subtasks_data[subtask]['terminals'][subtask_k[subtask]] = ep['dones'][i]
            subtask_k[subtask] = subtask_k[subtask] + 1
            last_subtask = subtask

    for i in range(num_subtasks):
        subtasks_data[i]['observations'] = subtasks_data[i]['observations'].transpose([0, 3, 1, 2])
    
    return subtasks_data


def train_agent(agent_type, task_data, config):
    print("Training the agent...")
    if agent_type == 'BC':
        agent = d3rlpy.algos.BC(use_gpu=True, scaler='pixel', learning_rate=1e-3)
    elif agent_type == 'TD3+BC':
        agent = d3rlpy.algos.TD3PlusBC(use_gpu=True, scaler='pixel', actor_learning_rate=1e-7, critic_learning_rate=1e-4)
    elif agent_type == 'DDPG':
        agent = d3rlpy.algos.DDPG(use_gpu=True, scaler='pixel')
    elif agent_type == 'SAC':
        agent = d3rlpy.algos.SAC(use_gpu=True, scaler='pixel')
    elif agent_type == 'AWAC':
        agent = d3rlpy.algos.AWAC(use_gpu=True, scaler='pixel')
    else:
        raise Exception("Unknown Agent Type!!")

    agent.build_with_dataset(task_data)

    if config.checkpoint:
        agent.load_model(config.checkpoint)

    # Train
    train_episodes, test_episodes = train_test_split(task_data)
    agent.fit(
        # task_data,
        train_episodes,
        eval_episodes=test_episodes, 
        n_epochs=config.num_epochs, 
        scorers={
            # 'td_error': d3rlpy.metrics.td_error_scorer,
            'continuous_action_diff': d3rlpy.metrics.continuous_action_diff_scorer
            },
        logdir=config.logdir
    )

    return agent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_subtasks', type=int, default=2, help="Number of subtasks to train different agents on")
    parser.add_argument('--learn_subtask', type=int, help="Subtask ID to train agent on")
    parser.add_argument('--obs_height', type=int, default=100, help="Height of the observation to use for agent training")
    parser.add_argument('--obs_width', type=int, default=100, help="Width of the observation to use for agent training")
    parser.add_argument('--agent_algo', type=str, default="BC", choices=['BC', 'TD3+BC', 'DDPG', 'SAC', 'AWAC'], help="Algorithm to use for the learning agent")
    parser.add_argument('--checkpoint', type=str, help="Checkpoint to initialize learning agent's parameters with")
    parser.add_argument('--num_epochs', type=int, default=10000, help="Number of epochs to train the agent for")
    parser.add_argument('--logdir', type=str, default="d3rlpy_logs", help="Directory to store logs in")

    config = parser.parse_args()

    rollouts_data = load_data("/scratch1/amanbans/bellman/chair_ingolf_0650.gail.p0.123_step_00011468800_1000_trajs.pkl")
    # subtasks_data = extract_subtasks_data(rollouts_data=rollouts_data, num_subtasks=config.num_subtasks, config=config)
    subtask_data = extract_data(rollouts_data=rollouts_data, config=config)
    del rollouts_data

    agents = []
    if config.learn_subtask == None:
        for i in tqdm(range(config.num_subtasks), desc="Subtask for which the agent is being trained"):
            # task_mdp = d3rlpy.dataset.MDPDataset(**subtasks_data[i])
            task_mdp = d3rlpy.dataset.MDPDataset(**subtask_data)
            print(f"Training the agent for subtask {i} with {len(task_mdp)} demonstrations.")
            agents.append(train_agent(agent_type=config.agent_algo, task_data=task_mdp, config=config))
    else:
        # task_mdp = d3rlpy.dataset.MDPDataset(**subtasks_data[config.learn_subtask])
        task_mdp = d3rlpy.dataset.MDPDataset(**subtask_data)
        print(f"Training the agent for subtask {config.learn_subtask} with {len(task_mdp)} demonstrations.")
        agents.append(train_agent(agent_type=config.agent_algo, task_data=task_mdp, config=config))
