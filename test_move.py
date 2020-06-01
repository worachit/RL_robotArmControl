from os.path import dirname, join, abspath, exists
import torch
import numpy as np

from RL import Memory,PPO
from EnvromentSim import ControlArm

from torch.utils.tensorboard import SummaryWriter
import numpy as np
# from os import path

ABS_PATH = dirname(abspath(__file__))
WEIGHT_PATH = join(ABS_PATH, 'Weight/')


def train(ep,writer):
    env = ControlArm()
    ############## Hyperparameters ##############
    solved_reward = 300         # stop training if avg_reward > solved_reward
    log_interval = 10           # print avg reward in the interval
    max_episodes = 10000        # max training episodes
    max_timesteps = 600        # max timesteps in one episode
    
    update_timestep = 100      # update policy every n timesteps
    action_std = 0.5            # constant std for action distribution (Multivariate Normal)
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                 # discount factor
    
    lr = 0.001                 # parameters for Adam optimizer
    
    random_seed = 42
    
    # creating environment
    state_dim = 10+2*7*6
    print(state_dim)
    action_dim = 7
    print(action_dim)
    
    #############################################
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, gamma, K_epochs, eps_clip)
    print(lr)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    if exists(WEIGHT_PATH + 'PPO_continuous_log.pth'):
        ppo.policy.load_state_dict(torch.load(WEIGHT_PATH + 'PPO_continuous_log.pth'))

    # training loop
    for i_episode in range(1, max_episodes+1):
        episode_reward = 0
        not_find_path = True
        while(not_find_path):
            try:
                state = env.reset()
                not_find_path = False
            except:
                not_find_path = True
        

        for t in range(max_timesteps):
            time_step +=1

            # Running policy_old:
            action = ppo.select_action(state, memory)
            state, reward, done, terminate = env.step(action)
            
            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            # update if its time
            running_reward += reward
            episode_reward += reward
            if terminate:
                break
        
        ppo.update(memory)
        memory.clear_memory()
        time_step = 0
        print('Episode {} , reward : {}'.format(ep+i_episode,episode_reward))       
        
        writer.add_scalar('Train/return', episode_reward, ep+i_episode)

        avg_length += t

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            torch.save(ppo.policy.state_dict(), WEIGHT_PATH + 'PPO_continuous_log.pth')
            running_reward = 0
            avg_length = 0

        # save every 40 episodes 280
        if i_episode % 20 == 0:
            torch.save(ppo.policy.state_dict(), WEIGHT_PATH + 'PPO_continuous.pth')
            # env.shutdown()
            # break
            
    return i_episode
    
if __name__ == '__main__':
    ep = 0
    writer = SummaryWriter()

    # while(1):
    ep += train(ep,writer)