from re import T
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from torch.cuda import init
from using_Reinforcement_learing.approach_1.env import HangmanEnv
from torch.autograd import Variable
from using_Reinforcement_learing.approach_1.config import Config
import yaml
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from using_Reinforcement_learing.approach_1.memory import Transition, ReplayMemory
from using_Reinforcement_learing.approach_1.dqn import DQN
from using_Reinforcement_learing.approach_1.log import setup_custom_logger
import time
# import torchvision.transforms as T

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.8
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
obscured_string_len = 27

# create logger
logger = setup_custom_logger('root', "./glatest.log", "INFO")
logger.propagate = False
logger.setLevel(logging.INFO)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = None
        
class HangmanPlayer():
    def __init__(self, env, config):
        self.memory = ReplayMemory(config['rl']['memory_size'])
        self.steps_done = 0
        self.episode_durations = []
        self.last_episode = 0
        self.reward_in_episode = []
        self.env = env
        self.id = int(time.time())
        self.config = config
        self.n_actions = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compile()
        
    def compile(self):     
        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # summary(self.target_net, (128, 25, 27))
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        
    def _update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def _adjust_learning_rate(self, episode):
        delta = self.config['training']['learning_rate'] - self.config['optimizer']['lr_min']
        base = self.config['optimizer']['lr_min']
        rate = self.config['optimizer']['lr_decay']
        lr = base + delta * np.exp(-episode / rate)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
    def _get_action_for_state(self, state):
        sample = random.random()
        eps_threshold = self.config['rl']['epsilon_end'] + (self.config['rl']['epsilon_start'] - self.config['rl']['epsilon_end']) * \
            math.exp(-1. * self.steps_done / self.config['rl']['epsilon_decay'])
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                word_tensor = torch.tensor(state[0]).to(self.device)
                action_tensor = torch.tensor([state[1]]).to(self.device)
                q_values = self.policy_net(word_tensor, action_tensor)
                selected_action = q_values.argmax().item()  # 返回整数
                return selected_action
        else:
            return random.randrange(self.n_actions)
        
    def save(self):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            "reward_in_episode": self.reward_in_episode,
            "episode_durations": self.episode_durations,
            "config": self.config
            }, f"./models/pytorch_{self.id}.pt")
        
    def _remember(self, state, action, next_state, reward, done):
        word_tensor, actions_tensor = state
        next_word_tensor, next_actions_tensor = next_state
        self.memory.push(
            torch.tensor(word_tensor, device=self.device),
            torch.tensor(actions_tensor, device=self.device),
            torch.tensor(next_word_tensor, device=self.device),
            torch.tensor(next_actions_tensor, device=self.device),
            torch.tensor([action], device=self.device, dtype=torch.long),
            torch.tensor([reward], device=self.device),
            torch.tensor([done], device=self.device, dtype=torch.bool)
        )
    
    def fit(self):
        num_episodes = self.config['training']['max_episodes']
        self.episode_durations = []
        self.reward_in_episode = []
        reward_in_episode = 0
        self.epsilon_vec = []
        
        logger.info("开始训练，总轮数: %d", num_episodes)
        logger.info("预热轮数: %d", self.config['training']['warmup_episode'])
        logger.info("初始探索率: %.2f", self.config['rl']['epsilon_start'])
        logger.info("最小探索率: %.2f", self.config['rl']['epsilon_end'])
        
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            state = self.env.reset()
            state = (state[0].reshape(-1, 25, 27), state[1])
            for t in count():
                action = self._get_action_for_state(state)
                next_state, reward, done, info = self.env.step(action)
                next_state = (next_state[0].reshape(-1, 25, 27), next_state[1])
                self._remember(state, action, next_state, reward, done)
                
                if i_episode >= self.config['training']['warmup_episode']:
                    self._train_model()  # 添加训练步骤
                    self._adjust_learning_rate(i_episode - self.config['training']['warmup_episode'] + 1)
                    done = (t == self.config['rl']['max_steps_per_episode'] - 1) or done
                else:
                    done = (t == 5 * self.config['rl']['max_steps_per_episode'] - 1) or done
                
                state = next_state
                reward_in_episode += reward
                
                if done:
                    self.episode_durations.append(t + 1)
                    self.reward_in_episode.append(reward_in_episode)
                    if i_episode % 10 == 0:  # 每10个episode打印一次信息
                        eps_threshold = self.config['rl']['epsilon_end'] + (self.config['rl']['epsilon_start'] - self.config['rl']['epsilon_end']) * \
                            math.exp(-1. * self.steps_done / self.config['rl']['epsilon_decay'])
                        logger.info(f"Episode {i_episode}, Steps: {t+1}, Total Reward: {reward_in_episode:.2f}, Epsilon: {eps_threshold:.3f}")
                        if info.get('win', False):
                            logger.info("游戏胜利！")
                        elif info.get('gameover', False):
                            logger.info("游戏失败！")
                    reward_in_episode = 0
                    break
                
                if i_episode % 50 == 0:
                    self._update_target()
                
                self.last_episode = i_episode
            
            if i_episode % self.config['rl']['target_model_update_episodes'] == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                logger.info("更新目标网络")
            
            if i_episode % self.config['training']['save_freq'] == 0:
                self.save()
                logger.info("保存模型")
    
    def _train_model(self):  
        if len(self.memory) < self.config['rl']['batch_size']:
            return
        transitions = self.memory.sample(self.config['rl']['batch_size'])
        batch = Transition(*zip(*transitions))
        
        # 获取批次数据
        word_batch = torch.cat(batch.word_tensor)
        actions_batch = torch.cat(batch.actions_tensor)
        next_word_batch = torch.cat(batch.next_word_tensor)
        next_actions_batch = torch.cat(batch.next_actions_tensor)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        # 调整维度
        word_batch.resize_(self.config['rl']['batch_size'], 25, 27)
        next_word_batch.resize_(self.config['rl']['batch_size'], 25, 27)
        
        # 计算Q值
        state_action_values = self.policy_net(word_batch, actions_batch).gather(0, action_batch)
        next_state_values = torch.zeros(self.config['rl']['batch_size'], device=self.device, dtype=torch.float)
        next_state_values = self.target_net(next_word_batch, next_actions_batch).max(1)[0].detach()
        
        # 计算期望Q值
        expected_state_action_values = (next_state_values * self.config['rl']['gamma']) + reward_batch
        
        # 计算损失并更新
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1)).float()
        logger.info("trainmodel: loss = {0}".format(loss))
        
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
    def play(self, verbose:bool=False, sleep:float=0.2, max_steps:int=100):
        # Play an episode
        try:
            actions_str = ["South", "North", "East", "West", "Pickup", "Dropoff"]

            iteration = 0
            state = self.env.reset()  # reset environment to a new, random state
            state = (state[0].reshape(-1, 25, 27), state[1])
            if verbose:
                print(f"Iter: {iteration} - Action: *** - Reward ***")
            time.sleep(sleep)
            done = False

            while not done:
                action = self._get_action_for_state(state)
                iteration += 1
                state, reward, done, info = self.env.step(action)
                display.clear_output(wait=True)
                self.env.render()
                if verbose:
                    print(f"Iter: {iteration} - Action: {action}({actions_str[action]}) - Reward {reward}")
                time.sleep(sleep)
                if iteration == max_steps:
                    print("cannot converge :(")
                    break
        except KeyboardInterrupt:
            pass
            
    def evaluate(self, max_steps:int=100):
        try:
            total_steps, total_penalties = 0, 0
            episodes = 100

            for episode in trange(episodes):
                state = self.env.reset()  # reset environment to a new, random state
                state = (state[0].reshape(-1, 25, 27), state[1])
                nb_steps, penalties, reward = 0, 0, 0

                done = False

                while not done:
                    action = self._get_action_for_state(state)
                    state, reward, done, info = self.env.step(action)

                    if reward == -10:
                        penalties += 1

                    nb_steps += 1
                    if nb_steps == max_steps:
                        done = True

                total_penalties += penalties
                total_steps += nb_steps

            print(f"Results after {episodes} episodes:")
            print(f"Average timesteps per episode: {total_steps / episodes}")
            print(f"Average penalties per episode: {total_penalties / episodes}")    
        except KeyboardInterrupt:
            pass