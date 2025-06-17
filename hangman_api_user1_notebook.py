# %% [markdown]
# # Trexquant Interview Project (The Hangman Game)
# 
# * Copyright Trexquant Investment LP. All Rights Reserved. 
# * Redistribution of this question without written consent from Trexquant is prohibited

# %% [markdown]
# ## Instruction:
# For this coding test, your mission is to write an algorithm that plays the game of Hangman through our API server. 
# 
# When a user plays Hangman, the server first selects a secret word at random from a list. The server then returns a row of underscores (space separated)—one for each letter in the secret word—and asks the user to guess a letter. If the user guesses a letter that is in the word, the word is redisplayed with all instances of that letter shown in the correct positions, along with any letters correctly guessed on previous turns. If the letter does not appear in the word, the user is charged with an incorrect guess. The user keeps guessing letters until either (1) the user has correctly guessed all the letters in the word
# or (2) the user has made six incorrect guesses.
# 
# You are required to write a "guess" function that takes current word (with underscores) as input and returns a guess letter. You will use the API codes below to play 1,000 Hangman games. You have the opportunity to practice before you want to start recording your game results.
# 
# Your algorithm is permitted to use a training set of approximately 250,000 dictionary words. Your algorithm will be tested on an entirely disjoint set of 250,000 dictionary words. Please note that this means the words that you will ultimately be tested on do NOT appear in the dictionary that you are given. You are not permitted to use any dictionary other than the training dictionary we provided. This requirement will be strictly enforced by code review.
# 
# You are provided with a basic, working algorithm. This algorithm will match the provided masked string (e.g. a _ _ l e) to all possible words in the dictionary, tabulate the frequency of letters appearing in these possible words, and then guess the letter with the highest frequency of appearence that has not already been guessed. If there are no remaining words that match then it will default back to the character frequency distribution of the entire dictionary.
# 
# This benchmark strategy is successful approximately 18% of the time. Your task is to design an algorithm that significantly outperforms this benchmark.

# %%
import json
import requests
import random
import string
import secrets
import time
import re
import collections
import sys
from datetime import datetime

try:
    from urllib.parse import parse_qs, urlencode, urlparse
except ImportError:
    from urlparse import parse_qs, urlparse
    from urllib import urlencode

from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# %%
class HangmanAPI(object):
    def __init__(self, access_token=None, session=None, timeout=None):
        self.hangman_url = self.determine_hangman_url()
        print(f"Initialized HangmanAPI with URL: {self.hangman_url}")
        
        self.access_token = access_token
        self.session = session or requests.Session()
        self.timeout = timeout
        self.guessed_letters = []
        
        full_dictionary_location = "words_250000_train.txt"
        print(f"Loading dictionary from: {full_dictionary_location}")
        self.full_dictionary = self.build_dictionary(full_dictionary_location)        
        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()
        print(f"Loaded {len(self.full_dictionary)} words from dictionary")
        
        self.current_dictionary = []
        # Define common letter combinations (you can expand this list)
        self.common_combinations = {
            # 2字母组合
            'th', 'he', 'in', 'er', 'an', 're', 'ed', 'on', 'es', 'st',
            'nd', 'at', 'or', 'nt', 'is', 'ar', 'te', 'en', 'al', 'to',
            'ch', 'sh', 'ph', 'gh', 'wh', 'ck', 'ng', 'qu', 'sc', 'sp',
            
            # 3字母组合
            'ing', 'ion', 'ati', 'ent', 'and', 'tha', 'str', 'all', 'out',
            'tch', 'igh', 'ough', 'tion', 'sion', 'ence', 'ance', 'ment', 'able', 'ible',
            'log', 'phy', 'cal', 'ous', 'ial', 'ian', 'ist', 'ism', 'ize', 'ise',
            'ate', 'ify', 'ify', 'ous', 'ious', 'eous', 'ious', 'eous',
            
            # 4字母组合
            'tion', 'sion', 'ment', 'able', 'ible', 'ance', 'ence', 'ally', 'fully',
            'ical', 'ical', 'ical', 'ical', 'ical', 'ical', 'ical', 'ical',
            'logy', 'nomy', 'graph', 'scope', 'meter', 'ology', 'pathy', 'phobia',
            'berry', 'berry', 'berry', 'berry', 'berry',
            
            # 5字母组合
            'ation', 'sion', 'ment', 'able', 'ible', 'berry', 'berry', 'berry',
            'ology', 'graphy', 'metry', 'scopy', 'phobia', 'pathy', 'logy',
            'strawberry', 'raspberry', 'blueberry', 'cranberry'
        }
        print("Initialized common combinations and patterns")
        
        # Pre-calculate combination frequencies from the full dictionary for efficiency
        self.combination_frequencies = self._precompute_combination_frequencies()
        print("Precomputed combination frequencies")

        # 添加 n-gram 相关初始化
        import nltk
        from nltk import ngrams
        from nltk import FreqDist
        
        print("Initializing n-gram analysis")
        # 建立 n-gram 频率统计
        f = open(full_dictionary_location, "r")
        df = []
        for x in f:
            df.append(x[:-1])

        self._2gram = []
        self._3gram = []
        self._4gram = [] 
        self._5gram = []

        for word in df:
            self._2gram.extend(list(ngrams(word, 2, pad_left=True, pad_right=True)))
            self._3gram.extend(list(ngrams(word, 3, pad_left=True, pad_right=True)))
            self._4gram.extend(list(ngrams(word, 4, pad_left=True, pad_right=True)))
            self._5gram.extend(list(ngrams(word, 5, pad_left=True, pad_right=True)))

        # 计算频率分布
        freq_2 = FreqDist(self._2gram)
        freq_3 = FreqDist(self._3gram)
        freq_4 = FreqDist(self._4gram)
        freq_5 = FreqDist(self._5gram)

        self.freq_2 = [(elem, freq_2.get(elem)) for elem in freq_2]
        self.freq_3 = [(elem, freq_3.get(elem)) for elem in freq_3]
        self.freq_4 = [(elem, freq_4.get(elem)) for elem in freq_4]
        self.freq_5 = [(elem, freq_5.get(elem)) for elem in freq_5]
        
        print("Completed n-gram analysis")
        
        # 其他辅助数据结构
        self.vowels = ['a','e','i','o','u']
        self.word_len_dict = {}
        for i in range(3, 30):
            self.word_len_dict[i] = []
            for words in df:
                if(len(words)>i):
                    for j in range(len(words)-i+1):
                        self.word_len_dict[i].append(words[j:j+i])
        
        print("Initialized word length dictionary")

        # 添加位置相关的组合统计
        self.position_combinations = self._build_position_combinations()
        print("Initialized position-based combinations")
        
        # 添加常见词尾组合
        self.common_endings = {
            # 基本词尾
            'ing', 'ed', 'er', 'est', 'ly', 'ful', 'less', 'ment', 'ness',
            'able', 'ible', 'ous', 'ious', 'eous', 'ious', 'eous', 'ious',
            
            # 专业词尾
            'logy', 'nomy', 'graph', 'scope', 'meter', 'ology', 'pathy', 'phobia',
            'ical', 'ical', 'ical', 'ical', 'ical', 'ical', 'ical', 'ical',
            'ation', 'sion', 'ment', 'able', 'ible', 'ance', 'ence',
            
            # 特殊词尾
            'berry', 'berry', 'berry', 'berry', 'berry'
        }
        
        # 添加常见词首组合
        self.common_beginnings = {
            # 基本前缀
            'un', 're', 'in', 'im', 'il', 'ir', 'dis', 'mis', 'pre', 'pro',
            'sub', 'super', 'trans', 'inter', 'intra', 'extra', 'ultra',
            
            # 专业前缀
            'bio', 'geo', 'hydro', 'micro', 'macro', 'photo', 'tele', 'thermo',
            'psycho', 'neuro', 'electro', 'astro', 'auto', 'semi', 'multi',
            
            # 特殊前缀
            'straw', 'rasp', 'blue', 'cran'
        }
        
        # 添加常见词根
        self.common_roots = {
            'log', 'graph', 'scope', 'meter', 'ology', 'pathy', 'phobia',
            'bio', 'geo', 'hydro', 'micro', 'macro', 'photo', 'tele', 'thermo',
            'psycho', 'neuro', 'electro', 'astro', 'auto', 'semi', 'multi'
        }
        
        print("Initialized enhanced common combinations and patterns")

    @staticmethod
    def determine_hangman_url():
        links = ['https://trexsim.com']

        data = {link: 0 for link in links}

        for link in links:
            requests.get(link)
            for i in range(10):
                s = time.time()
                requests.get(link)
                data[link] = time.time() - s

        link = sorted(data.items(), key=lambda x: x[1])[0][0]
        link += '/trexsim/hangman'
        return link

    def candsort(self, cands, invalids, vowels, vowel_ratio):
        for cand in cands:
            if cand[0] == None or cand[0] in invalids:
                continue
            if cand[0] in vowels and vowel_ratio > 0.5:
                continue
            return cand
        return ('!', 0, 1)

    def weighted_count(self, new_dict):
        dictx = collections.Counter()
        for words in new_dict:
            temp = collections.Counter(words)
            for i in temp:
                temp[i] = 1
                dictx += temp
        return dictx

    def _precompute_combination_frequencies(self):
        """
        Precomputes frequencies of common combinations from the full dictionary.
        This helps in quickly evaluating which combinations are most likely.
        """
        combination_counts = collections.Counter()
        for word in self.full_dictionary:
            for combo in self.common_combinations:
                if combo in word:
                    combination_counts[combo] += 1
        return combination_counts

    def _build_position_combinations(self):
        """构建基于位置的字母组合统计"""
        position_stats = {}
        for word in self.full_dictionary:
            word_len = len(word)
            if word_len not in position_stats:
                position_stats[word_len] = {
                    'beginnings': collections.Counter(),
                    'endings': collections.Counter(),
                    'middle': collections.Counter()
                }
            
            # 记录词首组合（前2-3个字母）
            if word_len >= 2:
                position_stats[word_len]['beginnings'][word[:2]] += 1
            if word_len >= 3:
                position_stats[word_len]['beginnings'][word[:3]] += 1
                
            # 记录词尾组合（后2-3个字母）
            if word_len >= 2:
                position_stats[word_len]['endings'][word[-2:]] += 1
            if word_len >= 3:
                position_stats[word_len]['endings'][word[-3:]] += 1
                
            # 记录中间组合（3-5个字母）
            for i in range(1, word_len-2):
                if i+3 <= word_len:
                    position_stats[word_len]['middle'][word[i:i+3]] += 1
                if i+4 <= word_len:
                    position_stats[word_len]['middle'][word[i:i+4]] += 1
                if i+5 <= word_len:
                    position_stats[word_len]['middle'][word[i:i+5]] += 1
        
        return position_stats

    def _analyze_word_pattern(self, word):
        """分析当前单词的模式，返回可能的组合"""
        clean_word = word[::2].replace("_", ".")
        word_len = len(clean_word)
        revealed_letters = set(word[::2]) - {'_'}
        
        # 获取当前位置的统计信息
        position_stats = self.position_combinations.get(word_len, {})
        
        # 分析可能的组合
        possible_combinations = []
        
        # 1. 检查词首模式
        if word_len >= 2 and clean_word[0] != '.':
            prefix = clean_word[:2].replace('.', '')
            if prefix:
                # 检查常见前缀
                for prefix in self.common_beginnings:
                    if clean_word.startswith(prefix.replace('.', '')):
                        possible_combinations.append((prefix, 1000, 'prefix'))
                
                # 检查位置统计
                for combo, count in position_stats.get('beginnings', {}).most_common():
                    if combo.startswith(prefix):
                        possible_combinations.append((combo, count, 'beginning'))
        
        # 2. 检查词尾模式
        if word_len >= 2 and clean_word[-1] != '.':
            suffix = clean_word[-2:].replace('.', '')
            if suffix:
                # 检查常见后缀
                for suffix in self.common_endings:
                    if clean_word.endswith(suffix.replace('.', '')):
                        possible_combinations.append((suffix, 1000, 'suffix'))
                
                # 检查位置统计
                for combo, count in position_stats.get('endings', {}).most_common():
                    if combo.endswith(suffix):
                        possible_combinations.append((combo, count, 'ending'))
        
        # 3. 检查中间模式
        for i in range(1, word_len-2):
            if clean_word[i] != '.':
                pattern = clean_word[i:i+3].replace('.', '')
                if pattern:
                    # 检查常见词根
                    for root in self.common_roots:
                        if pattern in root:
                            possible_combinations.append((root, 1000, 'root'))
                    
                    # 检查位置统计
                    for combo, count in position_stats.get('middle', {}).most_common():
                        if pattern in combo:
                            possible_combinations.append((combo, count, 'middle'))
        
        # 4. 特别检查 berry 相关组合
        if 'berry' in clean_word or 'berry' in ''.join(revealed_letters):
            possible_combinations.extend([
                ('berry', 1000, 'special'),
                ('strawberry', 1000, 'special'),
                ('raspberry', 1000, 'special'),
                ('blueberry', 1000, 'special'),
                ('cranberry', 1000, 'special')
            ])
        
        # 5. 检查专业术语模式
        for combo in self.common_combinations:
            if len(combo) >= 4 and combo in clean_word:
                possible_combinations.append((combo, 1000, 'special'))
        
        return possible_combinations 