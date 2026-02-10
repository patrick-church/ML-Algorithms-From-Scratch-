import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List

def run_simulation(rounds=1000, random_seed=None):
    
    def generate_data(num_iterations: int) -> List[List[str]]:
        """Generate multiple shuffled decks."""
        red = '1' * 26  # Represent red cards with '1'
        black = '0' * 26  # Represent black cards with '0'
        deck = black + red  # Combine into a full deck

        results = []

        for i in tqdm(range(num_iterations)):
            seed = random_seed or i + 1  # Use the seed for shuffling, or use a default seed if not provided
            shuffled_deck = generate_sequence(deck, seed)  # Shuffle the deck using the seed
            results.append(shuffled_deck)  # Store the shuffled deck
        
        return results  # Return all shuffled decks

    def generate_sequence(seq: str, seed: int) -> List[str]:
        """Shuffle a sequence using a specific seed."""
        np.random.seed(seed)  
        seq_list = list(seq)
        np.random.shuffle(seq_list)  
        return seq_list
    
    def convert_to_brb_format(seq: str) -> str:
        """Convert binary sequence '000' -> 'BBB', '101' -> 'BRB'."""
        return seq.replace('0', 'B').replace('1', 'R')

    def score_deck(deck: List[str], seq1: str, seq2: str, score_by_points=False) -> tuple[int]:
        """Simulate a single game between Player 1 and Player 2 based on their sequences.
           score_by_points: If True, score by points, otherwise score by total cards collected."""
        p1_score = 0
        p2_score = 0
        pile = 2  # Initial pile size

        i = 0
        while i < len(deck) - 2:
            current_sequence = ''.join(deck[i:i+3])  # Take a slice of 3 cards from the deck as a string
            if current_sequence == seq1:
                if score_by_points:
                    p1_score += 1  # Award 1 point if scoring by points
                else:
                    p1_score += pile  # Add cards to Player 1's total
                pile = 2  # Reset pile after Player 1 takes the cards
                i += 3  # Skip the next 3 cards (since Player 1 won this sequence)
            elif current_sequence == seq2:
                if score_by_points:
                    p2_score += 1  # Award 1 point if scoring by points
                else:
                    p2_score += pile  # Add cards to Player 2's total
                pile = 2  # Reset pile after Player 2 takes the cards
                i += 3  # Skip the next 3 cards (since Player 2 won this sequence)
            else:
                i += 1  # No match, move to the next card

        return p1_score, p2_score

    def run_iteration(num_iter: int):
        """Run the simulation for a specified number of iterations."""
        # Binary sequences (in 3-bit format)
        sequences = ['000', '001', '010', '011', '100', '101', '110', '111']
        
        # Convert sequences to 'BRB' format
        brb_sequences = [convert_to_brb_format(seq) for seq in sequences]
        
        # Initialize two matrices to store win counts: one for total cards, one for points
        win_matrix_cards = pd.DataFrame(0.0, index=brb_sequences, columns=brb_sequences)
        win_matrix_points = pd.DataFrame(0.0, index=brb_sequences, columns=brb_sequences)

        # Generate all shuffled decks
        all_decks = generate_data(num_iter)  # This will generate `num_iter` decks

        # Simulate games for each deck and count wins (both by cards and points)
        for deck in all_decks:
            # Play each combination of Player 1 and Player 2 sequences
            for i in range(len(sequences)):
                for j in range(len(sequences)):
                    if j != i:  # Only compare distinct sequences
                        p1_seq = sequences[i]  # Player 1's sequence (binary)
                        p2_seq = sequences[j]  # Player 2's sequence (binary)

                        # Score the deck by total cards
                        p1_cards, p2_cards = score_deck(deck, p1_seq, p2_seq)

                        # Score the deck by points
                        p1_points, p2_points = score_deck(deck, p1_seq, p2_seq, score_by_points=True)

                        # Determine the winner based on cards collected
                        if p1_cards > p2_cards:
                            win_matrix_cards.loc[brb_sequences[i], brb_sequences[j]] += 1

                        # Determine the winner based on points
                        if p1_points > p2_points:
                            win_matrix_points.loc[brb_sequences[i], brb_sequences[j]] += 1

        # Convert win counts to probabilities by dividing by the number of rounds (decks)
        win_probabilities_cards = win_matrix_cards / num_iter
        win_probabilities_points = win_matrix_points / num_iter

        return win_probabilities_cards, win_probabilities_points  # Return both matrices

    # Run the iteration and return the win probability matrices (by cards and by points)
    return run_iteration(rounds)

# Example of running the simulation
if __name__ == "__main__":
    probability_matrix_cards, probability_matrix_points = run_simulation(100000)  # This will generate 80,000 decks and play all combinations
    print("Win Probability Matrix by Total Cards:")
    print(probability_matrix_cards)
    print("\nWin Probability Matrix by Points:")
    print(probability_matrix_points)
