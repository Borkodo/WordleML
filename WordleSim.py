import random
import csv


def read_csv(file_name):
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        return [row[0] for row in reader]


def character_count(s):
    count_dict = {}
    for char in s:
        if char in count_dict:
            count_dict[char] += 1
        else:
            count_dict[char] = 1
    return count_dict


class WordleSimulator:

    def __init__(self, valid_solutions, valid_guesses, max_guesses=6):
        self.valid_guesses = valid_guesses + valid_solutions
        self.valid_solutions = valid_solutions

        self.target_word = random.choice(self.valid_solutions)
        self.guesses = 0
        self.max_guesses = max_guesses
        self.total_score = 0

    def provide_feedback(self, guess):
        feedback = [0, 0, 0, 0, 0]

        counts = character_count(self.target_word)
        # Iterate for the greens
        for i in range(len(guess)):
            if guess[i] == self.target_word[i]:
                feedback[i] = 2
                counts[guess[i]] -= 1

        # Iterate for the yellows
        for i in range(len(guess)):
            if feedback[i] == 2:
                continue
            if guess[i] in self.target_word and counts[guess[i]] != 0:
                feedback[i] = 1
                counts[guess[i]] -= 1

        return feedback

    def play_round(self, guess):
        if guess == self.target_word:
            self.reset_game()
            self.total_score += 1
            return [2, 2, 2, 2, 2]

        self.guesses += 1
        return self.provide_feedback(guess)

    def reset_game(self):
        self.target_word = random.choice(self.valid_solutions)
        self.guesses = 0

# valid_solutions = read_csv("valid_solutions.csv")
# valid_guesses = read_csv("valid_guesses.csv")
#
# simulator = WordleSimulator(valid_solutions, valid_guesses)
#
#
# while simulator.guesses < simulator.max_guesses:
#     print(f"Attempt {simulator.guesses+1}/{simulator.max_guesses}")
#     guess = input("Enter your guess: ")
#     if guess not in simulator.valid_guesses:
#         print("Invalid guess. Try again!")
#         continue
#     print(simulator.play_round(guess))
#
#
# print("Ran out of guesses. Game over!")
# print(f"The word was {simulator.target_word}")
# print(f"Your total score was {simulator.total_score}")
