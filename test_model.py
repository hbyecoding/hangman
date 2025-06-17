import os
import yaml
import random
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.serialization import add_safe_globals
from using_Reinforcement_learing.approach_1.env import HangmanEnv
from using_Reinforcement_learing.approach_1.hangman_agent import HangmanPlayer

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加安全的全局变量
add_safe_globals([np.core.multiarray.scalar])

def load_test_words(filename, num_words=100):
    """从文件中加载测试单词"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            words = [word.strip().lower() for word in f.readlines()]
        return random.sample(words, min(num_words, len(words)))
    except FileNotFoundError:
        logger.error(f"找不到测试文件: {filename}")
        return []

def get_letter_frequency(words):
    """计算字母频率"""
    freq = {}
    total = 0
    for word in words:
        for letter in set(word):  # 每个单词中每个字母只计算一次
            freq[letter] = freq.get(letter, 0) + 1
            total += 1
    return {k: v/total for k, v in freq.items()}

def baseline_strategy(word, letter_freq):
    """基于字母频率的baseline策略"""
    guessed_letters = set()
    mistakes = 0
    revealed = set('_' * len(word))
    
    while mistakes < 6 and '_' in revealed:
        # 选择未猜过且频率最高的字母
        available_letters = {k: v for k, v in letter_freq.items() if k not in guessed_letters}
        if not available_letters:
            break
        guess = max(available_letters.items(), key=lambda x: x[1])[0]
        guessed_letters.add(guess)
        
        if guess in word:
            # 更新已揭示的字母
            for i, letter in enumerate(word):
                if letter == guess:
                    revealed[i] = guess
        else:
            mistakes += 1
    
    return mistakes < 6 and '_' not in revealed

def test_model(model_path, test_words):
    """测试模型性能"""
    # 加载配置
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建环境和智能体
    env = HangmanEnv()
    agent = HangmanPlayer(env, config)
    
    # 加载训练好的模型
    try:
        if model_path.endswith('.pt'):
            checkpoint = torch.load(model_path, weights_only=False)
            agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            agent.target_net.load_state_dict(checkpoint['target_state_dict'])
            logger.info(f"成功加载模型: {model_path}")
            logger.info(f"模型训练轮次: {checkpoint.get('epoch', '未知')}")
            logger.info(f"模型训练损失: {checkpoint.get('loss', '未知')}")
        else:
            agent.load_model(model_path)
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        return
    
    # 计算字母频率
    letter_freq = get_letter_frequency(test_words)
    
    # 测试结果
    total_games = len(test_words)
    wins = 0
    total_guesses = 0
    mistakes_list = []
    word_results = []
    
    # Baseline测试结果
    baseline_wins = 0
    baseline_mistakes = []
    
    for word in test_words:
        # 测试模型
        env.word = word
        state = env.reset()
        done = False
        guesses = 0
        mistakes = 0
        guessed_letters = set()
        
        result = "失败"  # 默认失败
        while not done:
            action = agent._get_action_for_state(state)
            if isinstance(action, torch.Tensor):
                action_idx = action.argmax().item()
            else:
                action_idx = int(action)
            state, reward, done, info = env.step(action_idx)
            guesses += 1
            guessed_letters.add(chr(action_idx + ord('a')))
            
            if not info.get('win', False) and reward < 0:
                mistakes += 1
            
            if info['win']:
                wins += 1
                result = "胜利"
                break
            elif mistakes >= 6:  # 使用错误次数判断游戏结束
                result = "失败"
                break
        
        total_guesses += guesses
        mistakes_list.append(mistakes)
        word_results.append({
            "单词": word,
            "结果": result,
            "猜测次数": guesses,
            "错误次数": mistakes,
            "猜测字母": "".join(sorted(guessed_letters))
        })
        
        # 测试baseline
        if baseline_strategy(word, letter_freq):
            baseline_wins += 1
            baseline_mistakes.append(mistakes)
    
    # 计算统计结果
    win_rate = wins / total_games
    baseline_win_rate = baseline_wins / total_games
    avg_guesses = total_guesses / total_games
    avg_mistakes = sum(mistakes_list) / len(mistakes_list)
    baseline_avg_mistakes = sum(baseline_mistakes) / len(baseline_mistakes) if baseline_mistakes else 0
    
    # 输出详细测试结果
    logger.info("\n=== 测试结果汇总 ===")
    logger.info(f"总游戏数: {total_games}")
    logger.info(f"模型胜率: {win_rate:.2%}")
    logger.info(f"Baseline胜率: {baseline_win_rate:.2%}")
    logger.info(f"平均猜测次数: {avg_guesses:.2f}")
    logger.info(f"平均错误次数: {avg_mistakes:.2f}")
    logger.info(f"Baseline平均错误次数: {baseline_avg_mistakes:.2f}")
    
    # 输出每个单词的详细结果
    logger.info("\n=== 详细测试结果 ===")
    for result in word_results:
        logger.info(f"单词: {result['单词']:<15} 结果: {result['结果']:<6} "
                   f"猜测次数: {result['猜测次数']:<3} 错误次数: {result['错误次数']:<3} "
                   f"猜测字母: {result['猜测字母']}")
    
    # 绘制结果对比图
    plt.figure(figsize=(12, 6))
    
    # 胜率对比
    plt.subplot(1, 2, 1)
    plt.bar(['模型', 'Baseline'], [win_rate, baseline_win_rate])
    plt.title('胜率对比')
    plt.ylim(0, 1)
    
    # 平均错误次数对比
    plt.subplot(1, 2, 2)
    plt.bar(['模型', 'Baseline'], [avg_mistakes, baseline_avg_mistakes])
    plt.title('平均错误次数对比')
    plt.ylim(0, 6)
    
    plt.tight_layout()
    plt.savefig('test_results.png')
    plt.close()
    
    return win_rate >= 0.5  # 返回是否达到50%胜率目标

def main():
    # 加载测试单词
    test_words = load_test_words("words.txt", 100)
    if not test_words:
        return
    
    # 测试模型
    model_path = "/data/hongboye/scripts/hangman/models/pytorch_1750042964.pt"
    if os.path.exists(model_path):
        success = test_model(model_path, test_words)
        if success:
            logger.info("恭喜！模型达到了50%的胜率目标！")
        else:
            logger.info("模型未达到50%的胜率目标，需要继续优化。")
    else:
        logger.error(f"找不到模型文件: {model_path}")
        # 尝试查找其他模型文件
        model_dir = "models"
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith(('.pt', '.pth'))]
            if model_files:
                logger.info(f"发现以下模型文件: {model_files}")
                logger.info("请指定要使用的模型文件路径")

if __name__ == "__main__":
    main() 