import pygame
import random
import pandas as pd
import os
from data.classes.Board import Board
from data.classes.StartEnd import StartEnd
import time

pygame.init()

# Read the original CSV file
input_csv = "/Users/abhinavgoel/chess4brainhealth/Health Data.csv"
output_csv = "Filtered Health Data.csv"

# List of columns to keep
columns_to_keep = [
    "Blood Oxygen Saturation (%)",
    "Heart Rate [Max] (count/min)"
]

# Read the CSV file into a DataFrame
data = pd.read_csv(input_csv)

# Select only the desired columns
filtered_data = data[columns_to_keep]

filtered_data = filtered_data.dropna()

# Save the filtered data to a new CSV file
filtered_data.to_csv(output_csv, index=False)

sleep_input_csv = "/Users/abhinavgoel/chess4brainhealth/Sleep Analysis Data.csv"

# Read the CSV file into a DataFrame
sleep_data = pd.read_csv(sleep_input_csv)

# Keep only the "Value" column
value_column = sleep_data["Value"]

# Count instances of different values
value_counts = value_column.value_counts()

# Find the value with the most instances
most_common_value = value_counts.idxmax()
most_common_value_count = value_counts.max()

WINDOW_SIZE = (1200, 600)
screen = pygame.display.set_mode(WINDOW_SIZE)
screen2 = pygame.display.set_mode(WINDOW_SIZE)
screen3 = pygame.display.set_mode(WINDOW_SIZE)
screen4 = pygame.display.set_mode(WINDOW_SIZE)
csv_path = "/Users/abhinavgoel/chess4brainhealth/train.csv"

# Read the CSV file
df_filtered = pd.read_csv(csv_path)
random_row = random.choice(df_filtered.index)
initial_fen = df_filtered.loc[random_row, "fen"]
print(initial_fen)

# Determine whose turn it is (White or Black)
fen_parts = initial_fen.split()
if (fen_parts[1] == 'w'):
	print('white to move')
else:
	print('black to move')


board = Board(WINDOW_SIZE[0], WINDOW_SIZE[1], initial_fen)
board2 = StartEnd(WINDOW_SIZE[0], WINDOW_SIZE[1])
board3 = StartEnd(WINDOW_SIZE[0], WINDOW_SIZE[1])
board4 = StartEnd(WINDOW_SIZE[0], WINDOW_SIZE[1])

def draw(display, remaining_time, score, evaluation, blood_O2, max_heart_rate, most_common_value):
	display.fill('white')
	board.draw(display, remaining_time, score, evaluation, blood_O2, max_heart_rate, most_common_value)
	pygame.display.update()
def draw2(display):
	display.fill('black')
	board2.startDraw(display)
	pygame.display.update()
def draw3(display):
	display.fill('black')
	board3.timeUpDraw(display)
	pygame.display.update()
def draw4(display, winner):
	display.fill('black')
	board4.endDraw(display, winner)
	pygame.display.update()
def getBiometrics(path, index):
	biometric_data = pd.read_csv(path)
	num_rows = len(biometric_data)
	blood_O2 = biometric_data.at[index, "Blood Oxygen Saturation (%)"]
	max_heart_rate = biometric_data.at[index, "Heart Rate [Max] (count/min)"]
	return blood_O2, max_heart_rate



def calculate_piece_differential(fen):
	piece_values = {'Q': 9, 'R': 5, 'B': 3, 'N': 3, 'P': 1,
					'q': -9, 'r': -5, 'b': -3, 'n': -3, 'p': -1}

	fen_parts = fen.split(' ')
	position_part = fen_parts[0]

	white_value = 0
	black_value = 0

	for char in position_part:
		if char in piece_values:
			if char.isupper():
				white_value += piece_values[char]
			else:
				black_value += piece_values[char]

	piece_differential = white_value + black_value

	# Format the positive score with a plus sign
	if piece_differential > 0:
		formatted_score = f"+{piece_differential}"
	else:
		formatted_score = str(piece_differential)

	return formatted_score

score = calculate_piece_differential(initial_fen)

if __name__ == '__main__':
	pygame.init()
	# Initialize your game settings, screen, and other variables

	start_time = time.time()
	running = True
	running2 = True
	running3 = True
	running4 = True
	elapsed_time = 0  # Initialize elapsed time
	elapsed_time2 = 0
	elapsed_time3 = 0
	timer_duration = 15
	evaluation = "M1"
	path = "/Users/abhinavgoel/chess4brainhealth/Filtered Health Data.csv"
	index = 0
	biometric1, biometric2 = getBiometrics(path, index)
	last_biometric_update_time = start_time

	while running:
		# Calculate elapsed time
		elapsed_time = time.time() - start_time

		if running2 and elapsed_time < 5:
			draw2(screen2)  # Display screen2
		else:
			running2 = False  # Stop displaying screen2
			remaining_time = max(timer_duration - elapsed_time, 0)  # Calculate remaining time
			remaining_time2 = 1
			if time.time() - last_biometric_update_time >= 1:
				index = index + 1
				biometric1, biometric2 = getBiometrics(path, index)
				last_biometric_update_time = time.time()  # Update the last update time
			draw(screen, remaining_time, score, evaluation, biometric1, biometric2, most_common_value)
			if remaining_time == 0:
				running = False
				start_time2 = time.time()
				while running3:
					elapsed_time2 = time.time() - start_time2
					if running3 and elapsed_time2 < 2:
						draw3(screen3)
					else:
						running3 = False
			mx, my = pygame.mouse.get_pos()
			for event in pygame.event.get():
				# Quit the game if the user presses the close button
				if event.type == pygame.QUIT:
					running = False
				elif event.type == pygame.MOUSEBUTTONDOWN:
					# If the mouse is clicked
					if event.button == 1:
						board.handle_click(mx, my)
			if board.is_in_checkmate('black'):  # If black is in checkmate
				running = False
				start_time3 = time.time()
				while running4:
					elapsed_time3 = time.time() - start_time3
					if running3 and elapsed_time2 < 2:
						draw4(screen4, "White")
					else:
						running4 = False
			elif board.is_in_checkmate('white'):  # If white is in checkmate
				running = False
				start_time3 = time.time()
				while running4:
					elapsed_time3 = time.time() - start_time3
					if running3 and elapsed_time2 < 2:
						draw4(screen4, "Black")
					else:
						running4 = False
		pygame.display.flip()
