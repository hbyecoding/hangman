import json
from collections import Counter
from typing import List, Set, Dict
import re

class HangmanSolver:
    def __init__(self, dictionary_path: str):
        """Initialize the solver with a dictionary of words."""
        with open(dictionary_path, 'r') as f:
            self.dictionary = set(word.strip().lower() for word in f)
        
        # Initialize letter frequency for the entire dictionary
        self.letter_freq = Counter()
        for word in self.dictionary:
            self.letter_freq.update(word)
        
        # Create word length groups for faster matching
        self.length_groups: Dict[int, Set[str]] = {}
        for word in self.dictionary:
            length = len(word)
            if length not in self.length_groups:
                self.length_groups[length] = set()
            self.length_groups[length].add(word)
        
        # Track guessed letters
        self.guessed_letters: Set[str] = set()
        
    def _get_possible_words(self, masked_word: str) -> Set[str]:
        """Get all possible words that match the current masked word pattern."""
        # Convert masked word to regex pattern
        pattern = masked_word.replace('_', '.')
        regex = re.compile(f'^{pattern}$')
        
        # Get words of matching length
        length = len(masked_word.replace(' ', ''))
        candidates = self.length_groups.get(length, set())
        
        # Filter words that match the pattern
        return {word for word in candidates if regex.match(word)}
    
    def _calculate_letter_scores(self, possible_words: Set[str]) -> Dict[str, float]:
        """Calculate weighted scores for each letter based on possible words."""
        if not possible_words:
            return {letter: freq for letter, freq in self.letter_freq.items() 
                   if letter not in self.guessed_letters}
        
        # Count letter frequencies in possible words
        letter_counts = Counter()
        for word in possible_words:
            letter_counts.update(word)
        
        # Calculate scores with position-based weighting
        scores = {}
        for letter in set('abcdefghijklmnopqrstuvwxyz') - self.guessed_letters:
            if letter not in letter_counts:
                continue
                
            # Base score is the frequency in possible words
            score = letter_counts[letter]
            
            # Bonus for letters that appear in multiple positions
            position_bonus = 0
            for word in possible_words:
                if letter in word:
                    # Count how many times the letter appears in the word
                    position_bonus += word.count(letter)
            
            # Penalty for letters that appear in many words but in the same position
            if len(possible_words) > 1:
                position_penalty = 0
                for i in range(len(masked_word)):
                    if masked_word[i] == '_':
                        position_count = sum(1 for word in possible_words if word[i] == letter)
                        if position_count == len(possible_words):
                            position_penalty += 1
                
                score -= position_penalty
            
            scores[letter] = score + (position_bonus * 0.5)
        
        return scores
    
    def guess(self, masked_word: str) -> str:
        """Make a guess based on the current state of the game."""
        # Update masked word format
        masked_word = masked_word.replace(' ', '')
        
        # Get possible words that match the current pattern
        possible_words = self._get_possible_words(masked_word)
        
        # Calculate letter scores
        letter_scores = self._calculate_letter_scores(possible_words)
        
        if not letter_scores:
            # If no scores calculated, fall back to dictionary frequency
            return max((letter for letter, freq in self.letter_freq.items() 
                       if letter not in self.guessed_letters),
                      key=lambda x: self.letter_freq[x])
        
        # Make the guess
        guess = max(letter_scores.items(), key=lambda x: x[1])[0]
        self.guessed_letters.add(guess)
        
        return guess
    
    def reset(self):
        """Reset the solver's state for a new game."""
        self.guessed_letters.clear() 