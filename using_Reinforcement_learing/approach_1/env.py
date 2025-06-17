import gym
from gym import spaces
import string
from gym.spaces.multi_discrete import MultiDiscrete
import numpy as np
from gym.utils import seeding
import random
import collections
from sklearn.feature_extraction.text import CountVectorizer
import yaml
import logging
import os

config = None

MAX_WORDLEN = 25

config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")
with open(config_path, 'r') as stream:
	try:
		config = yaml.safe_load(stream)
	except yaml.YAMLError as exc:
		print(exc)

logger = logging.getLogger('root')
# logger.warning('is when this event was logged.')


class HangmanEnv(gym.Env):

	def __init__(self):
		# super().__init__()
		self.vocab_size = 26
		self.mistakes_done = 0
		self.action_space = spaces.Discrete(26)  # 26个字母
		self.observation_space = spaces.Box(low=0, high=1, shape=(26,), dtype=np.float32)
		self.words = self.load_words('words_250000_train.txt')
		self.vectorizer = CountVectorizer(tokenizer=lambda x: list(x))
		# self.wordlist = [w.strip() for w in f]
		self.vectorizer.fit([string.ascii_lowercase])
		self.config = config
		self.char_to_id = {chr(97+x): x for x in range(self.vocab_size)}
		self.char_to_id['_'] = self.vocab_size
		self.id_to_char = {v:k for k, v in self.char_to_id.items()}
		self.reset()

	def filter_and_encode(self, word, vocab_size, min_len, char_to_id):
		"""
		checks if word length is greater than threshold and returns one-hot encoded array along with character sets
		:param word: word string
		:param vocab_size: size of vocabulary (26 in this case)
		:param min_len: word with length less than this is not added to the dataset
		:param char_to_id
		"""

		#don't consider words of lengths below a threshold
		word = word.strip().lower()
		if len(word) < min_len:
			return None, None, None

		encoding = np.zeros((len(word), vocab_size + 1))
		#dict which stores the location at which characters are present
		#e.g. for 'hello', chars = {'h':[0], 'e':[1], 'l':[2,3], 'o':[4]}
		chars = {k: [] for k in range(vocab_size+1)}

		for i, c in enumerate(word):
			idx = char_to_id[c]
			#update chars dict
			chars[idx].append(i)
			#one-hot encode
			encoding[i][idx] = 1

		zero_vec = np.zeros((MAX_WORDLEN - encoding.shape[0], vocab_size + 1))
		encoding = np.concatenate((encoding, zero_vec), axis=0)

		return encoding

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def load_words(self, filename):
		"""加载单词文件"""
		try:
			with open(filename, 'r', encoding='utf-8') as f:
				words = [word.strip().lower() for word in f.readlines()]
			return words
		except FileNotFoundError:
			logger.error(f"找不到单词文件: {filename}")
			return []

	def choose_word(self):
		"""从单词列表中选择一个单词"""
		if not self.words:
			logger.error("单词列表为空")
			return "hangman"  # 默认单词
		return random.choice(self.words)

	def count_words(self, word):
		lens = [len(w) for w in self.wordlist]
		counter=dict(collections.Counter(lens))
		return counter[len(word)]

	def reset(self):
		self.mistakes_done = 0
		# inputs, labels, miss_chars, input_lens, status = self.dataloader.return_batch()
		self.word = self.choose_word()
		self.wordlen = len(self.word)
		self.gameover = False
		self.win = False
		self.guess_string = "_"*self.wordlen
		self.actions_used = set()
		self.actions_correct = set()
		logger.info("Reset: Resetting for new word")

		logger.info("Reset: Selected word= {0}".format(self.word))


		self.state = (
			self.filter_and_encode(self.guess_string, 26, 0, self.char_to_id),
			np.array([0]*26)
		)

		logger.debug("Reset: Init State = {self.state}")

		return self.state

	def vec2letter(self, action):
		letters = string.ascii_lowercase
		# idx = np.argmax(action==1)
		return letters[action]

	def getGuessedWord(self, secretWord, lettersGuessed):
		"""
		secretWord: string, the word the user is guessing
		lettersGuessed: list, what letters have been guessed so far
		returns: string, comprised of letters and underscores that represents
		what letters in secretWord have been guessed so far.
		"""
		secretList = []
		secretString = ''
		for letter in secretWord:
			secretList.append(letter)
		for letter in secretList:
			if letter not in lettersGuessed:
				letter = '_'
			secretString += letter
		return secretString


	def check_guess(self, letter):
		if letter in self.word:
			self.prev_string = self.guess_string
			self.actions_correct.add(letter)
			self.guess_string = self.getGuessedWord(self.word, self.actions_correct)
			return True
		else:
			return False

	def step(self, action):
		# 检查action类型
		if isinstance(action, int):
			action_idx = action
		else:
			action_idx = action.argmax()
			
		# 将action转换为字母
		letter = chr(action_idx + ord('a'))
		
		# 初始化done变量
		done = False
		
		# 检查是否已经猜过这个字母
		if letter in self.actions_used:
			return self.state, -5.0, True, {'win': False, 'message': 'Letter already guessed'}
		
		# 添加猜测的字母
		self.actions_used.add(letter)
		
		# 检查字母是否在单词中
		if self.check_guess(letter):
			logger.info("ENV STEP: Correct guess, evaluating reward, guess was = {0}".format(letter))
			if(set(self.word) == self.actions_correct):
				# 胜利奖励：基础分 + 剩余机会奖励
				remaining_chances = 6 - self.mistakes_done
				reward = 50.0 + (remaining_chances * 5.0)  # 基础50分，每剩余一次机会加5分
				done = True
				self.win = True
				self.gameover = True
				logger.info("ENV STEP: Won Game, evaluating reward, guess was = {0}".format(letter))
			else:
				# 根据猜对的字母数量和单词长度给予奖励
				new_revealed = len([c for c in self.word if c in self.actions_correct])
				old_revealed = len([c for c in self.word if c in (self.actions_correct - {letter})])
				# 基础奖励3分，根据单词长度增加奖励
				base_reward = 3.0
				length_bonus = min(len(self.word) / 10, 2.0)  # 单词越长，奖励越高，最多加2分
				reward = (base_reward + length_bonus) * (new_revealed - old_revealed)
				self.actions_correct.add(letter)
		# 如果猜错了
		else:
			logger.info("ENV STEP: Incorrect guess, evaluating reward, guess was = {0}".format(letter))
			self.mistakes_done += 1
			if(self.mistakes_done >= 6):
				# 失败惩罚：基础分 + 剩余未猜出字母的惩罚
				remaining_letters = len(set(self.word) - self.actions_correct)
				reward = -20.0 - (remaining_letters * 2.0)  # 基础-20分，每个未猜出的字母-2分
				done = True
				self.gameover = True
			else:
				# 错误惩罚：基础分 + 剩余机会相关的惩罚
				remaining_chances = 6 - self.mistakes_done
				reward = -2.0 - (1.0 / remaining_chances)  # 基础-2分，剩余机会越少惩罚越大

		logger.info("ENV STEP: actions used = {0}".format(" ".join(self.actions_used)))
		self.state = (
			self.filter_and_encode(self.guess_string, 26, 0, self.char_to_id),
			self.vectorizer.transform(list(self.actions_used)).toarray()[0]
		)
		logger.debug("Intermediate State = {self.state}")
		return (self.state, reward, done, {'win': self.win, 'gameover': self.gameover})
