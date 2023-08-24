# /* Board.py

import pygame
from data.classes.Square import Square
from data.classes.pieces.Rook import Rook
from data.classes.pieces.Bishop import Bishop
from data.classes.pieces.Knight import Knight
from data.classes.pieces.Queen import Queen
from data.classes.pieces.King import King
from data.classes.pieces.Pawn import Pawn
import pandas as pd
import time

# Game state checker
class Board:
    def __init__(self, width, height, fen):
        self.width = width
        self.height = height
        self.tile_width = width // 8
        self.tile_height = height // 8
        self.selected_piece = None
        self.turn = self.extract_turn_from_fen(fen)
        self.config = [
            ['', '', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', ''],
            ['','','','','','','',''],
            ['','','','','','','',''],
            ['','','','','','','',''],
            ['','','','','','','',''],
            ['', '', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', ''],
        ]
        if fen:
            self.parse_fen(fen)
        self.squares = self.generate_squares()
        self.setup_board()
    def extract_turn_from_fen(self, fen):
        fen_parts = fen.split()
        if (fen_parts[1] == 'w'):
            return 'white'
        else:
            return 'black'
    def parse_fen(self, fen):
        fen_parts = fen.split()
        fen_board = fen_parts[0]
        rank_strings = fen_board.split('/')
        for row_idx, rank in enumerate(rank_strings):
            col_idx = 0
            for char in rank:
                if char.isdigit():
                    col_idx += int(char)
                else:
                    piece_color = 'b' if char.islower() else 'w'
                    piece_value = char.upper()
                    self.config[row_idx][col_idx] = f'{piece_color}{piece_value}'
                    col_idx += 1

    def generate_squares(self):
        output = []
        for y in range(8):
            for x in range(8):
                output.append(
                    Square(x,  y, self.tile_width, self.tile_height)
                )
        return output

    def get_square_from_pos(self, pos):
        for square in self.squares:
            if (square.x, square.y) == (pos[0], pos[1]):
                return square

    def get_piece_from_pos(self, pos):
        return self.get_square_from_pos(pos).occupying_piece
    def setup_board(self):
        for y, row in enumerate(self.config):
            for x, piece in enumerate(row):
                if piece != '':
                    square = self.get_square_from_pos((x, y))
                    # looking inside contents, what piece does it have
                    if piece[1] == 'R':
                        square.occupying_piece = Rook(
                            (x, y), 'white' if piece[0] == 'w' else 'black', self
                        )
                    # as you notice above, we put `self` as argument, or means our class Board
                    elif piece[1] == 'N':
                        square.occupying_piece = Knight(
                            (x, y), 'white' if piece[0] == 'w' else 'black', self
                        )
                    elif piece[1] == 'B':
                        square.occupying_piece = Bishop(
                            (x, y), 'white' if piece[0] == 'w' else 'black', self
                        )
                    elif piece[1] == 'Q':
                        square.occupying_piece = Queen(
                            (x, y), 'white' if piece[0] == 'w' else 'black', self
                        )
                    elif piece[1] == 'K':
                        square.occupying_piece = King(
                            (x, y), 'white' if piece[0] == 'w' else 'black', self
                        )
                    elif piece[1] == 'P':
                        square.occupying_piece = Pawn(
                            (x, y), 'white' if piece[0] == 'w' else 'black', self
                        )
    def handle_click(self, mx, my):
        adjusted_tile_width = (self.width - 500) // 8
        x = mx // adjusted_tile_width
        y = my // self.tile_height
        clicked_square = self.get_square_from_pos((x, y))
        if self.selected_piece is None:
            if clicked_square.occupying_piece is not None:
                if clicked_square.occupying_piece.color == self.turn:
                    self.selected_piece = clicked_square.occupying_piece
        elif self.selected_piece.move(self, clicked_square):
            self.turn = 'white' if self.turn == 'black' else 'black'
        elif clicked_square.occupying_piece is not None:
            if clicked_square.occupying_piece.color == self.turn:
                self.selected_piece = clicked_square.occupying_piece
    # check state checker
    def is_in_check(self, color, board_change=None): # board_change = [(x1, y1), (x2, y2)]
        output = False
        king_pos = None
        changing_piece = None
        old_square = None
        new_square = None
        new_square_old_piece = None
        if board_change is not None:
            for square in self.squares:
                if square.pos == board_change[0]:
                    changing_piece = square.occupying_piece
                    old_square = square
                    old_square.occupying_piece = None
            for square in self.squares:
                if square.pos == board_change[1]:
                    new_square = square
                    new_square_old_piece = new_square.occupying_piece
                    new_square.occupying_piece = changing_piece
        pieces = [
            i.occupying_piece for i in self.squares if i.occupying_piece is not None
        ]
        if changing_piece is not None:
            if changing_piece.notation == 'K':
                king_pos = new_square.pos
        if king_pos == None:
            for piece in pieces:
                if piece.notation == 'K' and piece.color == color:
                        king_pos = piece.pos
        for piece in pieces:
            if piece.color != color:
                for square in piece.attacking_squares(self):
                    if square.pos == king_pos:
                        output = True
        if board_change is not None:
            old_square.occupying_piece = changing_piece
            new_square.occupying_piece = new_square_old_piece
        return output
    # checkmate state checker
    def is_in_checkmate(self, color):
        output = False
        for piece in [i.occupying_piece for i in self.squares]:
            if piece != None:
                if piece.notation == 'K' and piece.color == color:
                    king = piece
        if king.get_valid_moves(self) == []:
            if self.is_in_check(color):
                output = True
        return output

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
        return piece_differential
    def drawScore(self, display, score):
        font = pygame.font.Font(None, 30)
        timer_text = font.render(f"Point Differential: {score}", True, (0, 0, 0))  # White color
        timer_rect = timer_text.get_rect(topleft=(10, 10))
        display.blit(timer_text, timer_rect)
    def drawTimer(self, display, remaining_time):
        font = pygame.font.Font(None, 30)
        timer_text = font.render(f"Time: {remaining_time:.1f}", True, (0, 0, 0))  # White color
        timer_rect = timer_text.get_rect(topright=(self.width - 600 - 10, 10))
        display.blit(timer_text, timer_rect)
    def drawEvaluation(self, display, evaluation):
        font = pygame.font.Font(None, 30)
        timer_text = font.render(f"Evaluation: {evaluation}", True, (0, 0, 0))  # White color
        timer_rect = timer_text.get_rect(topright=(450, 10))
        display.blit(timer_text, timer_rect)
    def draw(self, display, remaining_time, score, evaluation, resting_heart, norm_heart, sd_norm_heart):
        board_width = int(self.width * 18 / 20)
        board_height = int(self.height * 18 / 20)

        # Calculate the starting position to center the board
        start_x = (self.width - board_width) // 2
        start_y = (self.height - board_height) // 2

        # Calculate tile dimensions for the resized board
        tile_width = ((board_width - 500) // 8)
        tile_height = board_height // 8

        # Draw the board background
        pygame.draw.rect(display, (0, 0, 0), (start_x, start_y, board_width, board_height))
        if self.selected_piece is not None:
            self.get_square_from_pos(self.selected_piece.pos).highlight = True
            for square in self.selected_piece.get_valid_moves(self):
                square.highlight = True
        for square in self.squares:
            square.draw(display, start_x, start_y, tile_width, tile_height)
        self.drawTimer(display, remaining_time)
        self.drawScore(display, score)
        self.drawEvaluation(display, evaluation)
        self.drawBiometrics(display, resting_heart, norm_heart, sd_norm_heart)
    def drawBiometrics(self, display, blood_O2, max_heart_rate, most_common_value):
        font = pygame.font.Font(None, 30)
        if(most_common_value == "InBed"):
            most_common_value = "NREM"

        biometric_text_rest = f"Blood Oxygen Levels: {blood_O2}"
        biometric_text_norm = f"Max Heart Ratet: {max_heart_rate}"
        biometric_text_sd = f"Most Common Sleep Form: {most_common_value}"

        biometric_rest_rendered = font.render(biometric_text_rest, True, (255, 255, 255))
        biometric_norm_rendered = font.render(biometric_text_norm, True, (255, 255, 255))
        biometric_sd_rendered = font.render(biometric_text_sd, True, (255, 255, 255))

        biometric_rest_rect = biometric_rest_rendered.get_rect(bottomleft=(700, 200))
        display.blit(biometric_rest_rendered, biometric_rest_rect)

        biometric_norm_rect = biometric_norm_rendered.get_rect(bottomleft=(700, 300))
        display.blit(biometric_norm_rendered, biometric_norm_rect)

        biometric_sd_rect = biometric_sd_rendered.get_rect(bottomleft=(700, 400))
        display.blit(biometric_sd_rendered, biometric_sd_rect)
