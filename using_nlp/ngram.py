import re
import random
import collections
from collections import defaultdict, Counter
from nltk import ngrams
from functools import lru_cache
import pickle
import os
import time

class HangmanSolver:
    def __init__(self, dictionary_path="words_250000_train.txt", use_checkpoint=True):
        self.dictionary_path = dictionary_path
        self.checkpoint_dir = "checkpoints"
        self.use_checkpoint = use_checkpoint
        
        # 创建checkpoint目录
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        self.dictionary = self._load_dictionary(dictionary_path)
        
        if use_checkpoint:
            self._load_or_create_checkpoint()
        else:
            self._initialize_from_scratch()
        
        self.guessed_letters = set()
        self.current_dictionary = self.full_dictionary.copy()

    def _get_checkpoint_path(self):
        # 使用字典文件名作为checkpoint文件名的一部分
        dict_name = os.path.basename(self.dictionary_path).replace('.txt', '')
        return os.path.join(self.checkpoint_dir, f"hangman_stats_{dict_name}.pkl")

    def _load_or_create_checkpoint(self):
        checkpoint_path = self._get_checkpoint_path()
        
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
                self.global_letter_freq = checkpoint['global_letter_freq']
                self.ngram_model = checkpoint['ngram_model']
                self.length_to_words = checkpoint['length_to_words']
                self.full_dictionary = checkpoint['full_dictionary']
                self.full_dictionary_common_letter_sorted = checkpoint['full_dictionary_common_letter_sorted']
                self.common_combinations = checkpoint['common_combinations']
                self.combination_frequencies = checkpoint['combination_frequencies']
                self.length_index = checkpoint['length_index']
                self.pattern_cache = checkpoint['pattern_cache']
        else:
            print("Creating new checkpoint...")
            self._initialize_from_scratch()
            self._save_checkpoint()

    def _initialize_from_scratch(self):
        print("Calculating letter frequencies...")
        self.global_letter_freq = self._calc_global_letter_freq()
        
        print("Building n-gram model...")
        self.ngram_model = self._build_ngram_model()
        
        print("Indexing words by length...")
        self.length_to_words = self._index_words_by_length()
        
        print("Creating dictionary copies...")
        self.full_dictionary = self.dictionary.copy()
        
        # 创建长度索引
        print("Creating length index...")
        self.length_index = defaultdict(list)
        for word in self.full_dictionary:
            self.length_index[len(word)].append(word)
        
        print("Calculating letter statistics...")
        self.full_dictionary_common_letter_sorted = self._calc_letter_frequencies()
        
        print("Finding common combinations...")
        self.common_combinations, self.combination_frequencies = self._find_common_combinations()
        
        # 预计算模式缓存
        print("Building pattern cache...")
        self._build_pattern_cache()

    def _save_checkpoint(self):
        checkpoint_path = self._get_checkpoint_path()
        print(f"Saving checkpoint to {checkpoint_path}")
        
        checkpoint = {
            'global_letter_freq': self.global_letter_freq,
            'ngram_model': self.ngram_model,
            'length_to_words': self.length_to_words,
            'full_dictionary': self.full_dictionary,
            'full_dictionary_common_letter_sorted': self.full_dictionary_common_letter_sorted,
            'common_combinations': self.common_combinations,
            'combination_frequencies': self.combination_frequencies,
            'length_index': self.length_index,
            'pattern_cache': self.pattern_cache
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)

    def _load_dictionary(self, path):
        with open(path) as f:
            return [line.strip().lower() for line in f if line.strip()]
    
    def _calc_global_letter_freq(self):
        return Counter(c for word in self.dictionary for c in word)
    
    def _build_ngram_model(self, n=3):
        model = defaultdict(Counter)
        for word in self.dictionary:
            if len(word) >= n:
                for gram in ngrams(word, n):
                    model[len(word)][''.join(gram)] += 1
        return model
    
    def _index_words_by_length(self):
        index = defaultdict(list)
        for word in self.dictionary:
            index[len(word)].append(word)
        return index
    
    def _calc_letter_frequencies(self):
        # 计算字母在整个词典中的频率
        freq = collections.Counter()
        for word in self.full_dictionary:
            for ch in set(word):  # 使用set避免重复计数
                freq[ch] += 1
        return freq.most_common()
    
    def _find_common_combinations(self, min_length=2, max_length=3):
        # 找出常见的字母组合
        combinations = collections.Counter()
        for word in self.full_dictionary:
            for length in range(min_length, max_length + 1):
                for i in range(len(word) - length + 1):
                    combo = word[i:i+length]
                    combinations[combo] += 1
        
        # 只保留最常见的组合
        top_combinations = [combo for combo, _ in combinations.most_common(1000)]
        return top_combinations, combinations

    def guess(self, pattern):
        # 1. 预处理
        clean_pattern = pattern.replace('_', '.')
        len_word = len(clean_pattern)
        current_dictionary = self.current_dictionary
        new_dictionary = []

        # 2. 筛选可能的单词
        for dict_word in current_dictionary:
            if len(dict_word) != len_word:
                continue
            if re.match(clean_pattern, dict_word):
                new_dictionary.append(dict_word)
        self.current_dictionary = new_dictionary

        # 如果没有匹配的单词，回退到完整词典
        if not self.current_dictionary:
            self.current_dictionary = self.full_dictionary
            new_dictionary = []
            for dict_word in self.full_dictionary:
                if len(dict_word) != len_word:
                    continue
                if re.match(clean_pattern, dict_word):
                    new_dictionary.append(dict_word)
            self.current_dictionary = new_dictionary

        # 基于常见字母组合的优先级猜测
        best_combo_letter = None
        best_combo_score = -1

        current_combo_scores = collections.Counter()
        for dict_word in self.current_dictionary:
            for combo in self.common_combinations:
                combo_len = len(combo)
                for i in range(len_word - combo_len + 1):
                    word_slice = clean_pattern[i:i+combo_len]
                    match_possible = True
                    for j in range(combo_len):
                        if word_slice[j] != '.' and word_slice[j] != combo[j]:
                            match_possible = False
                            break
                        if combo[j] in self.guessed_letters and word_slice[j] == '.':
                            match_possible = False
                            break
                    if match_possible:
                        for k in range(combo_len):
                            if word_slice[k] == '.' and combo[k] not in self.guessed_letters:
                                current_combo_scores[combo[k]] += self.combination_frequencies[combo]
        
        if current_combo_scores:
            for letter, score in current_combo_scores.most_common():
                if letter not in self.guessed_letters:
                    return letter

        # 3. 按位置统计字母频率
        position_counters = [collections.Counter() for _ in range(len_word)]
        for w in new_dictionary:
            for i, ch in enumerate(w):
                if clean_pattern[i] == '.' and ch not in self.guessed_letters:
                    position_counters[i][ch] += 1

        # 4. 统计总频率
        total_counter = collections.Counter()
        for w in new_dictionary:
            for ch in set(w):
                if ch not in self.guessed_letters:
                    total_counter[ch] += 1

        # 5. 优先猜元音
        vowels = ['e', 'a', 'o', 'i', 'u']
        for v in vowels:
            if v not in self.guessed_letters and total_counter[v] > 0:
                return v

        # 6. 按位置优先猜测高频字母
        best_letter = None
        best_score = -1
        for i, counter in enumerate(position_counters):
            if not counter:
                continue
            letter, score = counter.most_common(1)[0]
            if score > best_score and letter not in self.guessed_letters:
                best_letter = letter
                best_score = score
        if best_letter:
            return best_letter

        # 7. 猜测总频率最高的字母
        for letter, _ in total_counter.most_common():
            if letter not in self.guessed_letters:
                return letter

        # 8. 回退到全字典统计
        for letter, _ in self.full_dictionary_common_letter_sorted:
            if letter not in self.guessed_letters:
                return letter

        # 9. 兜底返回未猜过的字母
        for ch in 'abcdefghijklmnopqrstuvwxyz':
            if ch not in self.guessed_letters:
                return ch
        return 'e'

    def start_game(self, word=None, max_attempts=6, verbose=True):
        word = word or random.choice(self.dictionary)
        pattern = '_' * len(word)
        self.guessed_letters = set()
        self.current_dictionary = self.full_dictionary.copy()  # 重置当前词典

        attempt = 0
        while attempt < max_attempts:
            if verbose:
                print(f"Attempts left: {max_attempts - attempt} | Current: {pattern}")
            
            guess = self.guess(pattern)
            self.guessed_letters.add(guess)
            
            if verbose:
                print(f"Guessing: {guess}")
            
            if guess in word:
                pattern = ''.join(
                    c if c in self.guessed_letters else '_' 
                    for c in word
                )
                if pattern == word:
                    if verbose:
                        print(f"Success! Word: {word}")
                    return True
            else:
                attempt += 1
        
        if verbose:
            print(f"Failed! Word was: {word}")
        return False

def evaluate(solver, test_words, max_attempts=6, verbose=False):
    wins = 0
    total_attempts = 0
    
    for i, word in enumerate(test_words):
        solver.guessed_letters = set()
        solver.current_dictionary = solver.full_dictionary.copy()  # 重置当前词典
        pattern = '_' * len(word)
        success = False
        
        attempt = 0
        while attempt < max_attempts:
            guess = solver.guess(pattern)
            solver.guessed_letters.add(guess)
            
            if guess in word:
                pattern = ''.join(
                    c if c in solver.guessed_letters else '_' 
                    for c in word
                )
                if pattern == word:
                    wins += 1
                    success = True
                    break
            else:
                attempt += 1
        
        if verbose:
            status = "Success" if success else "Failed"
            print(f"Word {i+1}/{len(test_words)}: {status} - {word}")

    win_rate = wins / len(test_words)
    return win_rate

if __name__ == "__main__":
    # 初始化求解器
    start_time = time.time()
    print("Initializing solver...")
    solver = HangmanSolver(dictionary_path="words_250000_train.txt", use_checkpoint=True)
    print(f"Initialization took {time.time() - start_time:.2f} seconds")
    
    # 单局游戏演示
    print("\n=== Demo Game ===")
    solver.start_game(word="result")
    
    # 批量评估性能
    print("\n=== Evaluation ===")
    start_time = time.time()
    print("Loading test words...")
    test_words = random.sample(open("/Users/hbye/quantT/kunwang/hangman/using_nlp/words_test.txt").readlines(), 100)
    test_words = [word.strip().lower() for word in test_words]
    
    print("Starting evaluation...")
    win_rate = evaluate(solver, test_words, verbose=True)
    print(f"\nOverall Win Rate: {win_rate:.1%}")
    print(f"Evaluation took {time.time() - start_time:.2f} seconds")
