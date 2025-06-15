import unittest
from src.hangman_solver import HangmanSolver
import os

class TestHangmanSolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a small test dictionary
        cls.test_dict_path = "test_dictionary.txt"
        with open(cls.test_dict_path, "w") as f:
            f.write("\n".join([
                "apple",
                "banana",
                "cherry",
                "date",
                "elderberry",
                "fig",
                "grape",
                "honeydew"
            ]))
        
        cls.solver = HangmanSolver(cls.test_dict_path)
    
    @classmethod
    def tearDownClass(cls):
        # Clean up test dictionary
        os.remove(cls.test_dict_path)
    
    def setUp(self):
        self.solver.reset()
    
    def test_initial_guess(self):
        """Test that the initial guess is based on overall letter frequency."""
        guess = self.solver.guess("_ _ _ _ _")
        self.assertIn(guess, "aeiou")  # Should guess a vowel first
    
    def test_pattern_matching(self):
        """Test that the solver correctly matches word patterns."""
        possible_words = self.solver._get_possible_words("a _ _ _ e")
        self.assertIn("apple", possible_words)
        self.assertNotIn("banana", possible_words)
    
    def test_letter_scoring(self):
        """Test that letter scoring takes into account position and frequency."""
        scores = self.solver._calculate_letter_scores({"apple", "ample"})
        self.assertGreater(scores.get("p", 0), 0)  # 'p' should have a positive score
    
    def test_guessed_letters(self):
        """Test that the solver doesn't guess the same letter twice."""
        first_guess = self.solver.guess("_ _ _ _ _")
        second_guess = self.solver.guess("_ _ _ _ _")
        self.assertNotEqual(first_guess, second_guess)
    
    def test_word_completion(self):
        """Test that the solver can complete a partially revealed word."""
        guess = self.solver.guess("a _ _ l e")
        self.assertIn(guess, "p")  # Should guess 'p' to complete "apple"

if __name__ == "__main__":
    unittest.main() 