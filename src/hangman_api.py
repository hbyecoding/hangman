import requests
import json
from typing import Dict, Any
from src.hangman_solver import HangmanSolver

class HangmanAPI:
    def __init__(self, api_key: str, dictionary_path: str):
        """Initialize the API client with an API key and dictionary path."""
        self.api_key = api_key
        self.base_url = "https://hangman-api.com/api/v1"
        self.solver = HangmanSolver(dictionary_path)
        
    def start_game(self) -> Dict[str, Any]:
        """Start a new game of Hangman."""
        response = requests.post(
            f"{self.base_url}/hangman",
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return response.json()
    
    def make_guess(self, game_id: str, letter: str) -> Dict[str, Any]:
        """Make a guess in the current game."""
        response = requests.put(
            f"{self.base_url}/hangman/{game_id}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"letter": letter}
        )
        return response.json()
    
    def play_game(self) -> bool:
        """Play a complete game of Hangman using the solver."""
        # Start new game
        game = self.start_game()
        game_id = game["id"]
        masked_word = game["masked_word"]
        
        # Reset solver state
        self.solver.reset()
        
        # Play until game is over
        while True:
            # Make guess using solver
            guess = self.solver.guess(masked_word)
            
            # Submit guess
            result = self.make_guess(game_id, guess)
            
            # Check if game is over
            if result.get("game_over"):
                return result.get("won", False)
            
            # Update masked word for next guess
            masked_word = result["masked_word"]
    
    def play_multiple_games(self, num_games: int = 1000) -> Dict[str, Any]:
        """Play multiple games and return statistics."""
        wins = 0
        total_guesses = 0
        
        for _ in range(num_games):
            won = self.play_game()
            if won:
                wins += 1
            total_guesses += 1
        
        return {
            "total_games": num_games,
            "wins": wins,
            "win_rate": wins / num_games,
            "losses": num_games - wins
        } 