import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from using_Reinforcement_learing.approach_1.env import HangmanEnv
from using_Reinforcement_learing.approach_1.hangman_agent import HangmanPlayer
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def plot_training_curves(rewards, win_rates, save_dir='training_curves'):
    """绘制训练曲线"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制reward曲线
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Reward Curve')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'reward_curve.png'))
    plt.close()
    
    # 绘制胜率曲线
    plt.figure(figsize=(10, 5))
    plt.plot(win_rates)
    plt.title('Training Win Rate Curve')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'winrate_curve.png'))
    plt.close()

def main():
    # 加载配置
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建环境
    env = HangmanEnv()
    
    # 创建智能体
    agent = HangmanPlayer(env, config)
    
    # 创建模型保存目录
    os.makedirs("models", exist_ok=True)
    
    # 训练记录
    rewards = []
    win_rates = []
    eval_interval = 10  # 每10轮评估一次
    eval_episodes = 20  # 每次评估20轮
    
    # 开始训练
    logger.info("开始训练...")
    try:
        agent.fit()
    except Exception as e:
        logger.error(f"训练过程中出现错误: {str(e)}", exc_info=True)
    finally:
        # 保存最终模型
        agent.save()
        logger.info("训练完成，模型已保存")
    
    # 保存最终训练曲线
    plot_training_curves(rewards, win_rates)

if __name__ == "__main__":
    main() 