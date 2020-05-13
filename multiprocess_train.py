from os.path import dirname, join, abspath
import torch
import numpy as np

from RL import Memory,PPO
from EnvromentSim import ControlArm

from multiprocessing import Process

PROCESSES = 2

ABS_PATH = dirname(abspath(__file__))
WEIGHT_PATH = join(ABS_PATH, 'Weight/')


############## Hyperparameters ##############
solved_reward = 300         # stop training if avg_reward > solved_reward
log_interval = 20           # print avg reward in the interval
max_episodes = 10000        # max training episodes
max_timesteps = 500        # max timesteps in one episode
    
# update_timestep = 400      # update policy every n timesteps
action_std = 0.5            # constant std for action distribution (Multivariate Normal)
K_epochs = 80               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.9                 # discount factor
    
lr = 0.0003                 # parameters for Adam optimizer
betas = (0.9, 0.999)
    
random_seed = 42
    
# creating environment
state_dim = 31
print(state_dim)
action_dim = 7
print(action_dim)
#############################################

ENV = [ControlArm() for i in range(PROCESSES)]
MEM = [Memory() for i in range(PROCESSES)]
ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)

def simulate(i):
    env = ENV(i)
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    memory = MEM(i)
    print(lr,betas)

    # logging variables
    running_reward = 0
    # avg_length = 0
    memory.clear_memory()

    # ppo.policy.load_state_dict(torch.load(WEIGHT_PATH + 'PPO_continuous_log.pth'))
    
    not_find_path = True
    while(not_find_path):
        try:
            state = env.reset()
            not_find_path = False
        except:
            not_find_path = True
        

    for t in range(max_timesteps):
        # Running policy_old:
        action = ppo.select_action(state, memory)
        state, reward, done, terminate = env.step(action)
            
        # Saving reward and is_terminals:
        memory.rewards.append(reward)
        memory.is_terminals.append(done)
            
        running_reward += reward
    
        if terminate:
            break

    env.shutdown()
    print('Episode {}\t Avg reward: {}'.format(i_episode, running_reward))

# env = ControlArm()
# def train():
    # ############## Hyperparameters ##############
    # solved_reward = 300         # stop training if avg_reward > solved_reward
    # log_interval = 20           # print avg reward in the interval
    # max_episodes = 10000        # max training episodes
    # max_timesteps = 500        # max timesteps in one episode
    
    # # update_timestep = 400      # update policy every n timesteps
    # action_std = 0.5            # constant std for action distribution (Multivariate Normal)
    # K_epochs = 80               # update policy for K epochs
    # eps_clip = 0.2              # clip parameter for PPO
    # gamma = 0.9                 # discount factor
    
    # lr = 0.0003                 # parameters for Adam optimizer
    # betas = (0.9, 0.999)
    
    # random_seed = None
    
    # # creating environment
    # state_dim = 31
    # print(state_dim)
    # action_dim = 7
    # print(action_dim)
    
    # #############################################
    # if random_seed:
    #     print("Random Seed: {}".format(random_seed))
    #     torch.manual_seed(random_seed)
    #     np.random.seed(random_seed)
    
    # memory = Memory()
    # ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    # print(lr,betas)

    # # logging variables
    # running_reward = 0
    # avg_length = 0

    # # ppo.policy.load_state_dict(torch.load(WEIGHT_PATH + 'PPO_continuous_log.pth'))

    # training loop
    # for i_episode in range(1, max_episodes+1):
        
        # not_find_path = True
        # while(not_find_path):
        #     try:
        #         state = env.reset()
        #         not_find_path = False
        #     except:
        #         not_find_path = True
        

        # for t in range(max_timesteps):
        #     # time_step +=1

        #     # Running policy_old:
        #     action = ppo.select_action(state, memory)
        #     state, reward, done, terminate = env.step(action)
            
        #     # Saving reward and is_terminals:
        #     memory.rewards.append(reward)
        #     memory.is_terminals.append(done)
            
        #     running_reward += reward
        #     if terminate:
        #         break



        # avg_length += t
        # stop training if avg_reward > solved_reward
    
if __name__ == '__main__':
    # pool = mp.Pool(processes=PROCESSES)
    processes = [Process(target=simulate, args=(i,)) for i in range(PROCESSES)]
    for i_episode in range(1, max_episodes+1):
        # POOL = [pool.apply_async(simulate,(i,)) for i in range(PROCESSES)]
        # MEMORY = [POOL(i).get() for i in range(PROCESSES)]
        
        [p.start() for p in processes]
        [p.join() for p in processes]

        for memory in MEM:
            ppo.update(memory)
    
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), WEIGHT_PATH +'PPO_continuous_solved.pth')
            # env.shutdown()
            break
        
        # save every 500 episodes
        if i_episode % 500 == 0:
            torch.save(ppo.policy.state_dict(), WEIGHT_PATH + 'PPO_continuous.pth')
            
        # logging
        if i_episode % log_interval == 0:
            torch.save(ppo.policy.state_dict(), WEIGHT_PATH + 'PPO_continuous_log.pth')
            running_reward = 0
            avg_length = 0
