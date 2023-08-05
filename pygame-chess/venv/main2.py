import pygame
import random
import pandas as pd
import os

from data.classes.Board import Board

pygame.init()

WINDOW_SIZE = (600, 600)
screen = pygame.display.set_mode(WINDOW_SIZE)
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

def draw(display):
	display.fill('white')
	board.draw(display)
	pygame.display.update()


if __name__ == '__main__':
	running = True
	while running:
		mx, my = pygame.mouse.get_pos()
		for event in pygame.event.get():
			# Quit the game if the user presses the close button
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.MOUSEBUTTONDOWN:
       			# If the mouse is clicked
				if event.button == 1:
					board.handle_click(mx, my)
		if board.is_in_checkmate('black'): # If black is in checkmate
			print('White wins!')
			running = False
		elif board.is_in_checkmate('white'): # If white is in checkmate
			print('Black wins!')
			running = False
		# Draw the board
		draw(screen)