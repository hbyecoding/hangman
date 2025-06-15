import random
import collections
import re

class SimpleHangman:
    def __init__(self):
        # 初始化一个简单的字典
        self.dictionary = [
            "apple", "banana", "cherry", "date", "elderberry",
            "fig", "grape", "honeydew", "kiwi", "lemon",
            "mango", "orange", "pear", "quince", "raspberry"
        ]
        self.guessed_letters = []
        self.current_dictionary = self.dictionary
        self.secret_word = ""
        self.masked_word = ""
        self.tries_remaining = 6
        
    def start_game(self):
        # 随机选择一个单词
        self.secret_word = random.choice(self.dictionary)
        self.masked_word = "_ " * len(self.secret_word)
        import pdb; pdb.set_trace()
        self.guessed_letters = []
        self.tries_remaining = 6
        self.current_dictionary = self.dictionary
        print(f"\n新游戏开始！单词长度: {len(self.secret_word)}")
        print(f"当前单词: {self.masked_word}")
        return self.masked_word
    
    def guess(self, word):
        # 清理输入单词
        clean_word = word.replace(" ", "")
        import pdb; pdb.set_trace()
        # 在字典中查找可能的单词
        possible_words = []
        for dict_word in self.current_dictionary:
            if len(dict_word) != len(clean_word):
                continue
            if re.match(clean_word.replace("_", "."), dict_word):
                possible_words.append(dict_word)
        
        # 更新当前字典
        self.current_dictionary = possible_words
        
        # 统计字母频率
        letter_counts = collections.Counter("".join(possible_words))
        
        # 选择最可能的字母
        guess_letter = None
        for letter, count in letter_counts.most_common():
            if letter not in self.guessed_letters:
                guess_letter = letter
                break
        
        # 如果没有找到可能的字母，使用整体字典频率
        if not guess_letter:
            all_letters = collections.Counter("".join(self.dictionary))
            for letter, count in all_letters.most_common():
                if letter not in self.guessed_letters:
                    guess_letter = letter
                    break
        
        return guess_letter
    
    def make_guess(self, letter):
        if letter in self.guessed_letters:
            print(f"字母 {letter} 已经被猜过了！")
            return False
        
        self.guessed_letters.append(letter)
        
        if letter in self.secret_word:
            # 更新掩码单词
            new_masked = list(self.masked_word)
            for i, char in enumerate(self.secret_word):
                if char == letter:
                    new_masked[i*2] = letter
            self.masked_word = "".join(new_masked)
            print(f"\n猜对了！字母 {letter} 在单词中。")
            print(f"当前单词: {self.masked_word}")
            return True
        else:
            self.tries_remaining -= 1
            print(f"\n猜错了！字母 {letter} 不在单词中。")
            print(f"还剩 {self.tries_remaining} 次机会。")
            return False
    
    def is_game_over(self): 
        if "_" not in self.masked_word:
            print(f"\n恭喜你赢了！单词是: {self.secret_word}")
            return True
        if self.tries_remaining <= 0:
            print(f"\n游戏结束！正确答案是: {self.secret_word}")
            return True
        return False

def play_game():
    game = SimpleHangman()
    game.start_game()
    
    while not game.is_game_over():
        # 使用算法猜测
        guess_letter = game.guess(game.masked_word)
        print(f"\n算法猜测字母: {guess_letter}")
        
        # 执行猜测
        game.make_guess(guess_letter)
        
        # 显示当前状态
        print(f"已猜测的字母: {', '.join(game.guessed_letters)}")
        print(f"当前单词: {game.masked_word}")
        print(f"剩余尝试次数: {game.tries_remaining}")

if __name__ == "__main__":
    play_game() 