import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from collections import deque
from WordleSim import read_csv, WordleSimulator
import random
import gc


def interpret_feedback(feedback, previous_word, word_info, valid_solutions):
    for i in range(len(feedback)):
        if feedback[i] == 2:
            word_info.mark_as_green(previous_word[i], i)
            word_info.set_letter_amount(previous_word[i], previous_word[:i + 1].count(previous_word[i]))
        if feedback[i] == 1:
            word_info.exclude_letter(previous_word[i], i)
            word_info.set_letter_amount(previous_word[i], previous_word[:i + 1].count(previous_word[i]))
        if feedback[i] == 0:
            word_info.exclude_letter(previous_word[i], i)
            if previous_word.count(previous_word[i]) == 1:
                word_info.set_letter_amount(previous_word[i], 0, True)
            else:
                add_to_amount = 0
                for j in range(i + 1, len(feedback)):
                    if feedback[j] == 2 and previous_word[j] == previous_word[i]:
                        add_to_amount += 1
                if i == 0:
                    word_info.set_letter_amount(previous_word[i], add_to_amount, True)
                else:
                    word_info.set_letter_amount(previous_word[i], previous_word[:i].count(previous_word[i])
                                                + add_to_amount, True)
    return word_info.filter_solutions(valid_solutions)


class DQNWordleBot:
    class WordInfo:
        def __init__(self):
            self.green_letters = [None] * 5
            self.excluded_letters = [set() for _ in range(5)]
            self.letter_amounts = {}

        def mark_as_green(self, letter, index):
            self.green_letters[index] = letter

        def exclude_letter(self, letter, index):
            self.excluded_letters[index].add(letter)

        def set_letter_amount(self, letter, amount, is_max=False):
            if letter in self.letter_amounts and self.letter_amounts[letter] >= 10:
                return
            if letter in self.letter_amounts and self.letter_amounts[letter] > amount:
                return
            additional_amount = 10 if is_max else 0
            self.letter_amounts[letter] = amount + additional_amount

        def filter_solutions(self, valid_solutions):
            return [
                word for word in valid_solutions if self._is_word_valid(word)
            ]

        def _is_word_valid(self, word):
            # print(f"Checking {word}")

            # Check green letters
            for i, letter in enumerate(self.green_letters):
                if letter is not None and letter != word[i]:
                    # print(f"Excluding {word} due to mismatch with green letter {letter} at index {i}")
                    return False

            # Check excluded letters
            for i, excluded in enumerate(self.excluded_letters):
                if word[i] in excluded:
                    # print(f"Excluding {word} due to presence of excluded letter {word[i]} at index {i}")
                    return False

            # Check letter amounts
            for letter, amount in self.letter_amounts.items():
                if amount >= 10:
                    if word.count(letter) != amount - 10:
                        # print(
                        #    f"Excluding {word} due to mismatch in count for letter {letter}. Expected {amount - 10} but got {word.count(letter)}")
                        return False
                else:
                    if word.count(letter) < amount:
                        # print(f"Excluding {word} because it has less than {amount} occurrences of letter {letter}")
                        return False

            return True

    def __init__(self, simulator):
        self.simulator = simulator
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.05  # exploration rate
        self.epsilon_min = 0.0001
        self.epsilon_decay = 1
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.valid_guesses = simulator.valid_guesses
        self.valid_solutions = simulator.valid_solutions
        self.tested_states = []
        self.guessed_words = []

    def _build_model(self):
        # Neural Network for Deep-Q learning Model
        model = Sequential()

        # Input layer
        model.add(Dense(256, input_dim=166, activation='relu'))  # Increased neurons
        model.add(Dropout(0.5))  # Adjusted dropout rate

        # First Hidden layer
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))

        # Second Hidden layer
        model.add(Dense(128, activation='relu'))  # Increased neurons
        model.add(Dropout(0.5))

        # Third Hidden layer
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))

        # Fourth Hidden layer
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))

        # Output layer
        model.add(Dense(len(self.simulator.valid_guesses), activation='linear'))

        model.compile(loss='mse', optimizer='adam')

        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def state_representation(self, feedback, word_info):
        # Create an array for letters
        letter_arr = np.array(list(map(chr, range(97, 123))))

        # Letter counts using NumPy
        letter_counts = np.array([word_info.letter_amounts.get(l, 0) for l in letter_arr])

        # Letter positions
        letter_positions = np.array([1 if l is not None else 0 for l in word_info.green_letters])

        # Excluded letter representation using NumPy
        excluded_representation = np.array(
            [[1 if l in word_info.excluded_letters[pos] else 0 for l in letter_arr] for pos in range(5)]
        ).flatten()

        # Combine and reshape using NumPy
        combined_array = np.concatenate([letter_counts, feedback, letter_positions, excluded_representation])

        return combined_array.reshape(1, combined_array.size)

    def act(self, state, epsilon_val):
        if np.random.rand() <= epsilon_val:
            available_choices = [word for word in self.simulator.valid_guesses if word not in self.guessed_words]
            return random.choice(available_choices)

        act_values = self.model.predict(state)[0]

        for i, word in enumerate(self.simulator.valid_guesses):
            if word in self.guessed_words:
                act_values[i] = -float('inf')  # Set the Q-value of guessed words to negative infinity

        return self.simulator.valid_guesses[np.argmax(act_values)]

    def replay(self, num_simulations):
        batch_size = min(len(self.memory), num_simulations)
        minibatch = random.sample(self.memory, batch_size)

        # Batch the states and next_states
        states = np.array([item[0] for item in minibatch])
        next_states = np.array([item[3] for item in minibatch])

        if states.ndim == 3:
            states = states.squeeze(axis=1)
        if next_states.ndim == 3:
            next_states = next_states.squeeze(axis=1)

        # Predict Q-values in batch using predict_on_batch
        targets = self.model.predict_on_batch(states)
        next_state_values = self.target_model.predict_on_batch(next_states)
        next_state_values_online = self.model.predict_on_batch(next_states)

        # Use in-place operation for target value updates
        for i, (state, action, reward, _, done) in enumerate(minibatch):
            action_index = self.simulator.valid_guesses.index(action)
            if done:
                targets[i][action_index] = reward
            else:
                best_action = np.argmax(next_state_values_online[i])
                targets[i][action_index] = reward + self.gamma * next_state_values[i][best_action]

        # Train model in batch
        self.model.train_on_batch(states, targets)

        # Delete unused variables
        del minibatch, states, next_states, targets, next_state_values, next_state_values_online
        gc.collect()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def simulate_new_state(self, state, num_simulations=50):
        new_sim = WordleSimulator(self.valid_solutions, self.valid_guesses, float('inf'))
        for _ in range(num_simulations):
            print(f"Running state {_}")
            action = self.act(state, max(self.epsilon, 0.9))

            feedback = new_sim.play_round(action)

            word_info = self.WordInfo()
            interpret_feedback(feedback, action, word_info, self.valid_solutions)

            next_state = self.state_representation(feedback, word_info)

            initial_solutions_count = len(self.valid_solutions)

            remaining_solutions = interpret_feedback(feedback, action, word_info, self.valid_solutions)

            actual_reduction_percentage = ((initial_solutions_count - len(remaining_solutions) + 1)
                                           / initial_solutions_count)
            expected_reduction_percentage = 0.9

            percentage_deviation = (actual_reduction_percentage - expected_reduction_percentage) * 100

            if percentage_deviation >= 0:
                reward = (percentage_deviation ** 2) * 10
            else:
                reward = (percentage_deviation ** 2) * -2

            if feedback == [2, 2, 2, 2, 2]:
                done = True
                reward = 1000
            else:
                done = False
            print(f"{action} with {reward} reward")
            self.remember(state, action, reward, next_state, done)

            new_sim.reset_game()

        self.replay(num_simulations)
        self.memory.clear()
        self.update_target_model()

    def play_games(self):
        game_num = 0
        while True:
            print(f"Game {game_num}, Epsilon {self.epsilon}")
            self.play_game()
            game_num += 1
            if game_num % 100 == 0:
                bot.save_model(f"bot_weights_game_{game_num}.h5")
                print(f"Saved bot's weights after game {game_num}")

    def play_game(self, batch_size=32):
        word_info = self.WordInfo()  # instantiate a new WordInfo object
        self.simulator.reset_game()
        self.valid_solutions = simulator.valid_solutions
        state = self.state_representation([0, 0, 0, 0, 0], word_info)
        self.guessed_words = []
        for t in range(self.simulator.max_guesses):
            if len(self.valid_solutions) < 3:
                action = self.valid_solutions[0]
            else:
                action = self.act(state, self.epsilon)
                if len(self.valid_solutions) >= 70:
                    if str(state) not in self.tested_states:
                        self.simulate_new_state(state, len(self.valid_solutions))
                        self.tested_states.append(str(state))
                else:
                    a = self.test_all_feedback()
                    if a:
                        action = a
            self.guessed_words.append(action)
            # Print the chosen action
            print(f"Round {t + 1}: Chosen action (word): {action}")

            feedback = self.simulator.play_round(action)

            # Print the feedback received
            print(f"Round {t + 1}: Received feedback: {feedback}")

            initial_solutions_count = len(self.valid_solutions)  # Count before filtering

            remaining_solutions = interpret_feedback(feedback, action, word_info, self.valid_solutions)

            actual_reduction_percentage = ((initial_solutions_count - len(remaining_solutions) + 1)
                                           / initial_solutions_count)
            expected_reduction_percentage = 0.9

            percentage_deviation = (actual_reduction_percentage - expected_reduction_percentage) * 100

            if percentage_deviation >= 0:
                reward = (percentage_deviation ** 2) * 10
            else:
                reward = (percentage_deviation ** 2) * -2

            if feedback == [2, 2, 2, 2, 2]:
                done = True
                reward = 1000
            else:
                done = False
            print(f"Round {t + 1} reward: {reward}")
            next_state = self.state_representation(feedback, word_info)

            self.valid_solutions = remaining_solutions

            # Print the remaining solutions
            print(f"Round {t + 1}: Remaining solutions {len(remaining_solutions)}")

            self.remember(state, action, reward, next_state, done)

            if reward < 0 and t != 0:
                self.simulate_new_state(state, max(initial_solutions_count, 50))

            state = next_state

            if len(self.memory) > batch_size:
                self.replay(batch_size)
                self.update_target_model()

                # Print the updated epsilon value
                print(f"Updated epsilon value: {self.epsilon}")

            if done:
                print("Bot won!")
                break

        # Print the end of game information
        print(f"Game ended. Finished with {len(self.valid_solutions)} possible solutions left")
        print("-------------------------------------------------------------")  # A separator line for clarity

    # ... (the remaining methods remain unchanged)
    def make_best_guess(self, state):
        act_values = self.model.predict(state)
        return self.simulator.valid_guesses[np.argmax(act_values[0])]

    def test_all_feedback(self):

        all_feedbacks = [[i, j, k, l, m] for i in range(3) for j in range(3) for k in range(3) for l in range(3) for m
                         in range(3)]

        for guess in self.valid_solutions:
            good_guess = True
            for feedback in all_feedbacks:
                word_info = self.WordInfo()
                if len(interpret_feedback(feedback, guess, word_info, self.valid_solutions)) > 1:
                    good_guess = False
                    break
            if good_guess:
                print(f"Found a good guess: {guess}")
                return guess
        print("Could not find a good guess")

    def save_model(self, file_name):
        self.model.save_weights(file_name)

    def load_model(self, file_name):
        self.model.load_weights(file_name)

    def top_3_recommendations(self, state):
        """Return the top 3 recommended actions based on the model's prediction."""
        act_values = self.model.predict(state)[0]

        for i, word in enumerate(self.simulator.valid_guesses):
            if word in self.guessed_words:
                act_values[i] = -float('inf')

        top_3_indices = np.argsort(act_values)[-3:][::-1]

        top_3_actions = [self.simulator.valid_guesses[idx] for idx in top_3_indices]

        return top_3_actions

    def play_with_user(self):
        word_info = self.WordInfo()  # instantiate a new WordInfo object
        self.simulator.reset_game()
        self.valid_solutions = simulator.valid_solutions
        state = self.state_representation([0, 0, 0, 0, 0], word_info)

        print("Starting a new game!")
        self.guessed_words = []
        for t in range(self.simulator.max_guesses):

            # Get the top 3 recommended actions from the model
            recommendations = self.top_3_recommendations(state)
            print(f"Model's top 3 recommendations: {', '.join(recommendations)}")

            action = input("Enter your word (or one from the recommendations): ")

            self.guessed_words.append(action)
            feedback = self.simulator.play_round(action)

            print(f"Guess {t + 1}: {action} - Feedback: {feedback}")

            if feedback == [2, 2, 2, 2, 2]:
                print("You guessed it!")
                break

            # Update word_info and filter solutions using interpret_feedback
            remaining_solutions = interpret_feedback(feedback, action, word_info, self.valid_solutions)
            self.valid_solutions = remaining_solutions

            # Update the state for the next iteration
            state = self.state_representation(feedback, word_info)

        else:
            print(f"Game over! Finished with {len(self.valid_solutions)} possible solutions left.")
            print("-------------------------------------------------------------")  # A separator line for clarity


valid_solutions = read_csv("valid_solutions.csv")[1:]
valid_guesses = read_csv("valid_guesses.csv")[1:]

simulator = WordleSimulator(valid_solutions, valid_guesses)
bot = DQNWordleBot(simulator)

bot.load_model('bot_weights_game_19400.h5')

# Train the bot
# bot.play_games()

# bot.load_model('bot_weights_game_9700.h5')

bot.play_with_user()

# word_info = bot.WordInfo()
#
# while simulator.guesses < simulator.max_guesses:
#     print(f"Attempt {simulator.guesses + 1}/{simulator.max_guesses}")
#     guess = input("Enter your guess: ")
#     if guess not in simulator.valid_guesses:
#         print("Invalid guess. Try again!")
#         continue
#     feedback = simulator.play_round(guess)
#     if feedback == [2, 2, 2, 2, 2]:
#         print(feedback)
#         word_info = bot.WordInfo()
#         bot.valid_solutions = simulator.valid_solutions
#     else:
#         print(f"New solution length: {len(bot.feedback_gen(feedback, guess, word_info))}")
#         print(feedback)
#
# print("Ran out of guesses. Game over!")
# print(f"The word was {simulator.target_word}")
# print(f"Your total score was {simulator.total_score}")
