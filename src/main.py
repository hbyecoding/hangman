import argparse
from src.hangman_api import HangmanAPI
import json
import logging

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(description='Play Hangman using an AI solver')
    parser.add_argument('--api-key', required=True, help='API key for the Hangman API')
    parser.add_argument('--dictionary', required=True, help='Path to the dictionary file')
    parser.add_argument('--games', type=int, default=1000, help='Number of games to play')
    parser.add_argument('--output', help='Path to save results JSON file')
    
    args = parser.parse_args()
    setup_logging()
    
    # Initialize API client
    api = HangmanAPI(args.api_key, args.dictionary)
    
    # Play games
    logging.info(f"Starting {args.games} games of Hangman...")
    results = api.play_multiple_games(args.games)
    
    # Log results
    logging.info(f"Games completed: {results['total_games']}")
    logging.info(f"Wins: {results['wins']}")
    logging.info(f"Win rate: {results['win_rate']:.2%}")
    
    # Save results if output path provided
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to {args.output}")

if __name__ == "__main__":
    main() 