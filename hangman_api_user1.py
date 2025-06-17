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
from loguru import logger
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
        # 配置 loguru
        timestamp = datetime.now().strftime("%Y%m%d")
        log_filename = f'hangman_{timestamp}.log'
    
        # 移除默认的处理器
        logger.remove()
        
        # 添加文件处理器
        logger.add(
            log_filename,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO",
            rotation="500 MB",
            retention="10 days",
            compression="zip"
        )
        
        # 添加控制台处理器
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO"
        )

        self.hangman_url = self.determine_hangman_url()
        logger.info(f"Initialized HangmanAPI with URL: {self.hangman_url}")
        
        self.access_token = access_token
        self.session = session or requests.Session()
        self.timeout = timeout
        self.guessed_letters = []
        
        full_dictionary_location = "words_250000_train.txt"
        logger.info(f"Loading dictionary from: {full_dictionary_location}")
        self.full_dictionary = self.build_dictionary(full_dictionary_location)        
        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()
        logger.info(f"Loaded {len(self.full_dictionary)} words from dictionary")
        
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
        logger.info("Initialized common combinations and patterns")
        
        # Pre-calculate combination frequencies from the full dictionary for efficiency
        self.combination_frequencies = self._precompute_combination_frequencies()
        logger.info("Precomputed combination frequencies")

        # 添加 n-gram 相关初始化
        import nltk
        from nltk import ngrams
        from nltk import FreqDist
        
        logger.info("Initializing n-gram analysis")
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
        
        logger.info("Completed n-gram analysis")
        
        # 其他辅助数据结构
        self.vowels = ['a','e','i','o','u']
        self.word_len_dict = {}
        for i in range(3, 30):
            self.word_len_dict[i] = []
            for words in df:
                if(len(words)>i):
                    for j in range(len(words)-i+1):
                        self.word_len_dict[i].append(words[j:j+i])
        
        logger.info("Initialized word length dictionary")

        # 添加位置相关的组合统计
        self.position_combinations = self._build_position_combinations()
        logger.info("Initialized position-based combinations")
        
        # 添加常见词尾组合
        self.common_endings = {
            # 基本词尾
            'ing', 'ed', 'er', 'est', 'ly', 'ful', 'less', 'ment', 'ness',
            'able', 'ible', 'ous', 'ious', 'eous', 'ious', 'eous', 'ious',
            
            # 专业词尾
            'logy', 'nomy', 'graph', 'scope', 'meter', 'ology', 'pathy', 'phobia',
            'ical', 'ical', 'ical', 'ical', 'ical', 'ical', 'ical', 'ical',
            'ation', 'sion', 'ment', 'able', 'ible', 'ance', 'ence'
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
        
        logger.info("Initialized enhanced common combinations and patterns")

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


    def ngram(self, word, index, invalids, freqs, vowel_ratio):
        
        # for each '_' found in the missing word, find valid substring matches and get appropriate weights,
        # giving priority to more complete and longer substrings, tiebreakers between longer substrings are
        # broken by higher weights or frequencies
        
        # candidate tuple structure is: (suggestion, weight, rank) 
        
        freq_2, freq_3, freq_4, freq_5 = freqs
        score1 = ('!', 0, 1)
        score2 = ('!', 0, 1)
        score3 = ('!', 0, 1)
        
        # i. case    
        if index == 0:       
            if word[index+1] == '.':
                return ('!', 0, 1)

            # iXXX Case
            if (len(word) >= 4) and ('.' not in word[index+1:index+4]):
                    cands = [(elem[0][1], elem[1], 5) for elem in freq_5 if (elem[0][0] == None) and 
                                                                            (elem[0][2] == word[index+1]) and 
                                                                            (elem[0][3] == word[index+2]) and 
                                                                            (elem[0][4] == word[index+3])]
                    return self.candsort(cands, invalids, self.vowels, vowel_ratio)

            # iXX Case
            if (len(word) >= 3) and ('.' not in word[index+1:index+3]):
                    cands = [(elem[0][1], elem[1], 4) for elem in freq_4 if (elem[0][0] == None) and 
                                                                            (elem[0][2] == word[index+1]) and 
                                                                            (elem[0][3] == word[index+2])]
                    return self.candsort(cands, invalids, self.vowels, vowel_ratio)

            # iX case
            cands = [(elem[0][1], elem[1], 3) for elem in freq_3 if (elem[0][0] == None) and 
                                                                    (elem[0][2] == word[index+1])]       
            return self.candsort(cands, invalids, self.vowels, vowel_ratio)


        # .i case    
        if index == len(word)-1:      
            if word[index-1] == '.':
                return ('!', 0, 1)

            # XXXi case:
            if (len(word) >= 4) and ('.' not in word[index-3:index]):
                    cands = [(elem[0][3], elem[1], 5) for elem in freq_5 if (elem[0][0] == word[index-3]) and 
                                                                            (elem[0][1] == word[index-2]) and 
                                                                            (elem[0][2] == word[index-1]) and 
                                                                            (elem[0][4] == None)]
                    return self.candsort(cands, invalids, self.vowels, vowel_ratio)         

            # XXi case
            if (len(word) >= 3) and ('.' not in word[index-2:index]):
                    cands = [(elem[0][2], elem[1], 4) for elem in freq_4 if (elem[0][0] == word[index-2]) and 
                                                                            (elem[0][1] == word[index-1]) and 
                                                                            (elem[0][3] == None)]
                    return self.candsort(cands, invalids, self.vowels, vowel_ratio)    

            # Xi case
            cands = [(elem[0][1], elem[1], 3) for elem in freq_3 if (elem[0][0] == word[index-1]) and 
                                                                    (elem[0][2] == None)]
            return self.candsort(cands, invalids, self.vowels, vowel_ratio)


        else:  
            # .i. case
            if word[index-1] == '.' and word[index+1] == '.':
                return ('!', 0, 1)


            # .iX family
            if word[index-1] == '.'and word[index+1] != '.': 


                # .iXXXX case
                if (len(word) - index >= 5) and (index >= 1) and ('.' not in word[index+1:index+5]):
                        cands = [(elem[0][1], elem[1], 5) for elem in freq_5 if (elem[0][2] == word[index+1]) and 
                                                                                (elem[0][3] == word[index+2]) and 
                                                                                (elem[0][4] == word[index+3])]
                        score1 = self.candsort(cands, invalids, self.vowels, vowel_ratio)

                # X.iXX case        
                if (len(word) - index >= 3) and (index >= 2) and (word[index+2] != '.') and (word[index-2] != '.'):
                        cands = [(elem[0][2], elem[1], 5) for elem in freq_5 if (elem[0][0] == word[index-2]) and 
                                                                                (elem[0][3] == word[index+1]) and 
                                                                                (elem[0][4] == word[index+2])]
                        score2 = self.candsort(cands, invalids, self.vowels, vowel_ratio)

                # XX.iX case                
                if (len(word) - index >= 2) and (index >= 3) and ('.' not in word[index-3:index-1]):
                        cands = [(elem[0][3], elem[1], 5) for elem in freq_5 if (elem[0][0] == word[index-3]) and 
                                                                                (elem[0][1] == word[index-2]) and 
                                                                                (elem[0][4] == word[index+1])]
                        score3 = self.candsort(cands, invalids, self.vowels, vowel_ratio)

                if (score1 != score2) or (score2 != score3) or (score1 != score3):
                    best_score = sorted([score1, score2, score3], key = lambda x: (x[2], x[1]), reverse=True)
                    return best_score[0]                    


                # .iXX case            
                if (len(word) - index >= 3) and (index >= 1) and (word[index+2] != '.'):
                        cands = [(elem[0][0], elem[1], 3) for elem in freq_3 if (elem[0][1] == word[index+1]) and 
                                                                                (elem[0][2] == word[index+2])]
                        score1 = self.candsort(cands, invalids, self.vowels, vowel_ratio)

                # X.iX case
                if (len(word) - index >= 2) and (index >= 2) and (word[index-2] != '.'):
                        cands = [(elem[0][2], elem[1], 4) for elem in freq_4 if (elem[0][0] == word[index-2]) and 
                                                                                (elem[0][3] == word[index+1])]
                        score2 = self.candsort(cands, invalids, self.vowels, vowel_ratio)

                if score1 != score2:
                    best_score = sorted([score1, score2], key = lambda x: (x[2], x[1]), reverse=True)
                    return best_score[0]        

                # .iX        
                cands = [(elem[0][0], elem[1], 2) for elem in freq_2 if elem[0][1] == word[index+1]]
                return self.candsort(cands, invalids, self.vowels, vowel_ratio)


            # Xi. family
            if word[index-1] != '.'and word[index+1] == '.':

                # XXXXi. case
                if (len(word) - index >= 2) and (index >= 4) and ('.' not in word[index-4:index-1]):
                        cands = [(elem[0][3], elem[1], 5) for elem in freq_5 if (elem[0][0] == word[index-3]) and
                                                                                (elem[0][1] == word[index-2]) and 
                                                                                (elem[0][2] == word[index-1])]
                        score1 = self.candsort(cands, invalids, self.vowels, vowel_ratio)

                # XXi.X case                    
                if (len(word) - index >= 3) and (index >= 2) and (word[index+2] != '.') and (word[index-2] != '.'):
                        cands = [(elem[0][2], elem[1], 5) for elem in freq_5 if (elem[0][0] == word[index-2]) and 
                                                                                (elem[0][1] == word[index-1]) and 
                                                                                (elem[0][4] == word[index+2])]
                        score2 = self.candsort(cands, invalids, self.vowels, vowel_ratio)
                
                # Xi.XX case
                if (len(word) - index >= 4) and (index >= 1) and ('.' not in word[index+2:index+4]):
                        cands = [(elem[0][1], elem[1], 5) for elem in freq_5 if (elem[0][0] == word[index-1]) and 
                                                                                (elem[0][3] == word[index+2]) and 
                                                                                (elem[0][4] == word[index+3])]
                        score3 = self.candsort(cands, invalids, self.vowels, vowel_ratio)

                if (score1 != score2) or (score2 != score3) or (score1 != score3):
                    best_score = sorted([score1, score2, score3], key = lambda x: (x[2], x[1]), reverse=True)
                    return best_score[0]  


                # XXi. case
                if (index >= 2) and (word[index-2] != '.'):
                        cands = [(elem[0][2], elem[1], 3) for elem in freq_3 if (elem[0][0] == word[index-2]) and 
                                                                                (elem[0][1] == word[index-1])]
                        score1 = self.candsort(cands, invalids, self.vowels, vowel_ratio)
                
                # Xi.X case
                if (len(word) - index >= 3) and (index >= 1) and (word[index+2] != '.'):
                        cands = [(elem[0][1], elem[1], 4) for elem in freq_4 if (elem[0][0] == word[index-1]) and 
                                                                                (elem[0][3] == word[index+2])]
                        score2 = self.candsort(cands, invalids, self.vowels, vowel_ratio)

                if score1 != score2:
                    best_score = sorted([score1, score2], key = lambda x: (x[2], x[1]), reverse=True)
                    return best_score[0]     

                # Xi. case
                cands = [(elem[0][1], elem[1], 2) for elem in freq_2 if elem[0][0] == word[index-1]]
                return self.candsort(cands, invalids, self.vowels, vowel_ratio)


            # XiX family
            if word[index-1] != '.'and word[index+1] != '.':
                
                # XXiXX case
                if (len(word) - index >= 3) and (index >= 2) and ('.' not in word[index-2:index+3]):
                        cands = [(elem[0][2], elem[1], 5) for elem in freq_5 if (elem[0][0] == word[index-2]) and 
                                                                                (elem[0][1] == word[index-1]) and 
                                                                                (elem[0][3] == word[index+1]) and
                                                                                (elem[0][4] == word[index+2])]
                        score1 = self.candsort(cands, invalids, self.vowels, vowel_ratio)

                # XiXXX case
                if (len(word) - index >= 4) and (index >= 1) and ('.' not in word[index-1:index+4]):
                        cands = [(elem[0][1], elem[1], 5) for elem in freq_5 if (elem[0][0] == word[index-1]) and 
                                                                                (elem[0][2] == word[index+1]) and 
                                                                                (elem[0][3] == word[index+2]) and
                                                                                (elem[0][4] == word[index+3])]
                        score2 = self.candsort(cands, invalids, self.vowels, vowel_ratio)

                # XXXiX case
                if (len(word) - index >= 2) and (index >= 3) and ('.' not in word[index-3:index+2]):
                        cands = [(elem[0][3], elem[1], 5) for elem in freq_5 if (elem[0][0] == word[index-3]) and 
                                                                                (elem[0][1] == word[index-2]) and 
                                                                                (elem[0][2] == word[index-1]) and
                                                                                (elem[0][4] == word[index+1])]
                        score3 = self.candsort(cands, invalids, self.vowels, vowel_ratio)

                if (score1 != score2) or (score2 != score3) or (score1 != score3):
                    best_score = sorted([score1, score2, score3], key = lambda x: (x[2], x[1]), reverse=True)
                    return best_score[0]                          


                # XiXX case
                if len(word) - index >= 3 and word[index+2] != '.':
                        cands = [(elem[0][1], elem[1], 4) for elem in freq_4 if (elem[0][0] == word[index-1]) and 
                                                                                (elem[0][2] == word[index+1]) and 
                                                                                (elem[0][3] == word[index+2])]
                        score1 = self.candsort(cands, invalids, self.vowels, vowel_ratio)

                # XXiX case
                if index >= 2 and word[index-2] != '.':
                        cands = [(elem[0][2], elem[1], 4) for elem in freq_4 if (elem[0][0] == word[index-2]) and 
                                                                                (elem[0][1] == word[index-1]) and 
                                                                                (elem[0][3] == word[index+1])]
                        score2 = self.candsort(cands, invalids, self.vowels, vowel_ratio)

                if score1 != score2:
                    best_score = sorted([score1, score2], key = lambda x: (x[2], x[1]), reverse=True)
                    return best_score[0]            


                # XiX case
                cands = [(elem[0][1], elem[1], 3) for elem in freq_3 if (elem[0][0] == word[index-1]) and 
                                                                        (elem[0][2] == word[index+1])]
                return self.candsort(cands, invalids, self.vowels, vowel_ratio)



    def _precompute_combination_frequencies(self):
        """
        Precomputes frequencies of common combinations from the full dictionary.
        This helps in quickly evaluating which combinations are most likely.
        """
        combination_counts = collections.Counter()
        for word in self.full_dictionary:
            for combo in self.common_combinations:
                if combo in word:
                    combination_counts[combo] += 1 # Count if the combination exists in the word
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

    def guess(self, word):
        clean_word = word[::2].replace("_",".")
        len_word = len(clean_word)
        logger.debug(f"Processing word: {clean_word} (length: {len_word})")
        
        # 分析单词模式
        possible_combinations = self._analyze_word_pattern(word)
        logger.debug(f"Found {len(possible_combinations)} possible combinations")
        
        # 更新当前可能的单词列表
        curr_dict = self.current_dictionary
        new_dict = []
        for dict_word in curr_dict:
            if len(dict_word) == len_word and re.match(clean_word, dict_word):
                new_dict.append(dict_word)
        
        self.current_dictionary = new_dict
        logger.debug(f"Current dictionary size: {len(new_dict)} words")
        
        # 基于组合分析选择字母
        if possible_combinations:
            # 按频率和类型排序组合
            possible_combinations.sort(key=lambda x: (x[2] != 'special', -x[1]))
            
            # 从最可能的组合中选择未猜过的字母
            for combo, _, combo_type in possible_combinations:
                for letter in combo:
                    if letter not in self.guessed_letters and letter not in word[::2]:
                        logger.debug(f"Selected letter '{letter}' based on {combo_type} analysis: {combo}")
                        return letter
        
        # 如果组合分析没有结果，回退到原有的猜测策略
        temp = self.weighted_count(new_dict)
        letter_weights = temp.most_common()
        
        # 获取元音比例
        count = 0
        for i in word:
            if i in self.vowels:
                count += 1
        vowel_ratio = count/len(word)
        logger.debug(f"Vowel ratio in current word: {vowel_ratio:.2f}")
        
        # 更新 n-gram 频率统计
        failed_letters = [l for l in self.guessed_letters if l not in word]
        logger.debug(f"Failed letters so far: {failed_letters}")
        
        freq_2 = [(elem[0], elem[1]) for elem in self.freq_2 if set(elem[0]).isdisjoint(set(failed_letters))]
        freq_3 = [(elem[0], elem[1]) for elem in self.freq_3 if set(elem[0]).isdisjoint(set(failed_letters))]
        freq_4 = [(elem[0], elem[1]) for elem in self.freq_4 if set(elem[0]).isdisjoint(set(failed_letters))]
        freq_5 = [(elem[0], elem[1]) for elem in self.freq_5 if set(elem[0]).isdisjoint(set(failed_letters))]
        freqs = [freq_2, freq_3, freq_4, freq_5]

        # 按照优先级尝试不同的猜测策略
        guess_letter = '!'
        
        # 1. 基于当前词典的字母频率
        for choice, count in letter_weights:
            if choice not in self.guessed_letters:
                if choice in self.vowels and vowel_ratio > 0.5:
                    continue
                guess_letter = choice
                logger.debug(f"Selected letter '{guess_letter}' based on current dictionary frequency")
                break
        
        # 2. 使用子串匹配
        if guess_letter == '!':
            sub_len = round(len_word/2)
            if sub_len >= 3:
                c = collections.Counter()
                for i in range(len_word - sub_len +1):
                    temp_dict = []
                    temp = self.weighted_count(temp_dict)
                    c += temp
                sorted_letter_count = c.most_common()
                
                for letter, _ in sorted_letter_count:
                    if letter not in self.guessed_letters:
                        guess_letter = letter
                        logger.debug(f"Selected letter '{guess_letter}' based on substring matching")
                        break
        
        # 3. 使用 n-gram 预测
        if guess_letter == '!':
            options = []
            for i in range(len(clean_word)):
                if clean_word[i] == '.':
                    option = self.ngram(clean_word, i, self.guessed_letters, freqs, vowel_ratio)
                    options.append(option)
            
            best_guesses = sorted(options, key = lambda x: (x[2], x[1]), reverse=True)
            if best_guesses:
                best_guess = best_guesses[0][0]
                if best_guess != '!':
                    guess_letter = best_guess
                    logger.debug(f"Selected letter '{guess_letter}' based on n-gram analysis")
        
        # 4. 回退到全词典的字母频率
        if guess_letter == '!':
            for letter, _ in self.full_dictionary_common_letter_sorted:
                if letter not in self.guessed_letters:
                    if letter in self.vowels and vowel_ratio > 0.5:
                        continue
                    guess_letter = letter
                    logger.debug(f"Selected letter '{guess_letter}' based on full dictionary frequency")
                    break
        
        logger.info(f"Chose letter '{guess_letter}' using enhanced strategy")
        return guess_letter

    ##########################################################
    # You'll likely not need to modify any of the code below #
    ##########################################################
    
    def build_dictionary(self, dictionary_file_location):
        text_file = open(dictionary_file_location,"r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary
                
    def start_game(self, practice=True, verbose=True):
        # reset guessed letters to empty set and current plausible dictionary to the full dictionary
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary
        logger.info("Starting new game")
                         
        response = self.request("/new_game", {"practice":practice})
        if response.get('status')=="approved":
            game_id = response.get('game_id')
            word = response.get('word')
            tries_remains = response.get('tries_remains')
            logger.info(f"Game started | ID: {game_id} | Tries left: {tries_remains} | Word: {word}")
            
            while tries_remains>0:
                # get guessed letter from user code
                guess_letter = self.guess(word)
                    
                # append guessed letter to guessed letters field in hangman object
                self.guessed_letters.append(guess_letter)
                logger.info(f"Making guess | Letter: {guess_letter} | Guessed so far: {''.join(self.guessed_letters)}")
                    
                try:    
                    res = self.request("/guess_letter", {"request":"guess_letter", "game_id":game_id, "letter":guess_letter})
                    logger.info(f"Server response | {res}")
                except HangmanAPIError as e:
                    logger.error(f"API Error | {str(e)}")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error | {str(e)}")
                    raise e

                status = res.get('status')
                tries_remains = res.get('tries_remains')
                
                if status=="success":
                    logger.info(f"Game won | ID: {game_id}")
                    return True
                elif status=="failed":
                    reason = res.get('reason', '# of tries exceeded!')
                    logger.info(f"Game lost | ID: {game_id} | Reason: {reason}")
                    return False
                elif status=="ongoing":
                    word = res.get('word')
        else:
            logger.error("Failed to start new game")
        return status=="success"
        
    def my_status(self):
        return self.request("/my_status", {})
    
    def request(
            self, path, args=None, post_args=None, method=None):
        if args is None:
            args = dict()
        if post_args is not None:
            method = "POST"

        # Add `access_token` to post_args or args if it has not already been
        # included.
        if self.access_token:
            # If post_args exists, we assume that args either does not exists
            # or it does not need `access_token`.
            if post_args and "access_token" not in post_args:
                post_args["access_token"] = self.access_token
            elif "access_token" not in args:
                args["access_token"] = self.access_token

        time.sleep(0.2)

        num_retry, time_sleep = 50, 2
        for it in range(num_retry):
            try:
                response = self.session.request(
                    method or "GET",
                    self.hangman_url + path,
                    timeout=self.timeout,
                    params=args,
                    data=post_args,
                    verify=False
                )
                break
            except requests.HTTPError as e:
                response = json.loads(e.read())
                raise HangmanAPIError(response)
            except requests.exceptions.SSLError as e:
                if it + 1 == num_retry:
                    raise
                time.sleep(time_sleep)

        headers = response.headers
        if 'json' in headers['content-type']:
            result = response.json()
        elif "access_token" in parse_qs(response.text):
            query_str = parse_qs(response.text)
            if "access_token" in query_str:
                result = {"access_token": query_str["access_token"][0]}
                if "expires" in query_str:
                    result["expires"] = query_str["expires"][0]
            else:
                raise HangmanAPIError(response.json())
        else:
            raise HangmanAPIError('Maintype was not text, or querystring')

        if result and isinstance(result, dict) and result.get("error"):
            raise HangmanAPIError(result)
        return result
    
class HangmanAPIError(Exception):
    def __init__(self, result):
        self.result = result
        self.code = None
        try:
            self.type = result["error_code"]
        except (KeyError, TypeError):
            self.type = ""

        try:
            self.message = result["error_description"]
        except (KeyError, TypeError):
            try:
                self.message = result["error"]["message"]
                self.code = result["error"].get("code")
                if not self.type:
                    self.type = result["error"].get("type", "")
            except (KeyError, TypeError):
                try:
                    self.message = result["error_msg"]
                except (KeyError, TypeError):
                    self.message = result

        Exception.__init__(self, self.message)

# %% [markdown]
# # API Usage Examples

# %% [markdown]
# ## To start a new game:
# 1. Make sure you have implemented your own "guess" method.
# 2. Use the access_token that we sent you to create your HangmanAPI object. 
# 3. Start a game by calling "start_game" method.
# 4. If you wish to test your function without being recorded, set "practice" parameter to 1.
# 5. Note: You have a rate limit of 20 new games per minute. DO NOT start more than 20 new games within one minute.

# %%
api = HangmanAPI(access_token="18965b3ab8184fc94104e4a7fb6c50", timeout=2000)


# %% [markdown]
# ## Playing practice games:
# You can use the command below to play up to 100,000 practice games.

# %%
api.start_game(practice=1,verbose=True)
[total_practice_runs,total_recorded_runs,total_recorded_successes,total_practice_successes] = api.my_status() # Get my game stats: (# of tries, # of wins)
practice_success_rate = total_practice_successes / total_practice_runs
print('run %d practice games out of an allotted 100,000. practice success rate so far = %.3f' % (total_practice_runs, practice_success_rate))


# %% [markdown]
# ## Playing recorded games:
# Please finalize your code prior to running the cell below. Once this code executes once successfully your submission will be finalized. Our system will not allow you to rerun any additional games.
# 
# Please note that it is expected that after you successfully run this block of code that subsequent runs will result in the error message "Your account has been deactivated".
# 
# Once you've run this section of the code your submission is complete. Please send us your source code via email.

# %%
for i in range(10):
    print('Playing ', i, ' th game')
    # Uncomment the following line to execute your final runs. Do not do this until you are satisfied with your submission
    api.start_game(practice=1,verbose=True)
    
    # DO NOT REMOVE as otherwise the server may lock you out for too high frequency of requests
    time.sleep(0.5)

# %%


# %% [markdown]
# ## To check your game statistics
# 1. Simply use "my_status" method.
# 2. Returns your total number of games, and number of wins.

# %%
[total_practice_runs,total_recorded_runs,total_recorded_successes,total_practice_successes] = api.my_status() # Get my game stats: (# of tries, # of wins)
success_rate = total_recorded_successes/total_recorded_runs
print('overall success rate = %.3f' % success_rate)

# %%



