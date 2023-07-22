from fentoboardimage import fenToImage, loadPiecesFolder
import random
import berserk
from stockfish import Stockfish
import chess


session = berserk.TokenSession("lip_RkiI0f9NGkIIR7QjvZMY")
client = berserk.Client(session=session)
board = [[" " for x in range(8)] for y in range(8)]
piece_list = ["R", "N", "B", "Q", "P"]


def get_moves_to_fen(initial_fen, target_fen):
    initial_board = chess.Board(initial_fen)
    target_board = chess.Board()
    target_moves = target_fen.split()[-2:]
    for move in target_moves:
        target_board.push_uci(move)

    moves_to_target = []
    for move in target_board.move_stack:
        if move in initial_board.legal_moves:
            moves_to_target.append(move)
            initial_board.push(move)
        else:
            raise ValueError("The target position is not reachable from the initial position.")

    return moves_to_target


def place_kings(brd):
	while True:
		rank_white, file_white, rank_black, file_black = random.randint(0,7), random.randint(0,7), random.randint(0,7), random.randint(0,7)
		diff_list = [abs(rank_white - rank_black),  abs(file_white - file_black)]
		if sum(diff_list) > 2 or set(diff_list) == set([0, 2]):
			brd[rank_white][file_white], brd[rank_black][file_black] = "K", "k"
			break

def populate_board(brd, wp, bp):
    white_piece_counts = {"Q": 0, "B": 0, "N": 0, "R": 0, "P": 0}
    black_piece_counts = {"q": 0, "b": 0, "n": 0, "r": 0, "p": 0}

    for x in range(2):
        if x == 0:
            piece_amount = wp
            pieces = piece_list
            piece_counts = white_piece_counts
        else:
            piece_amount = bp
            pieces = [s.lower() for s in piece_list]
            piece_counts = black_piece_counts

        while piece_amount > 0:
            piece_rank, piece_file = random.randint(0, 7), random.randint(0, 7)
            piece = random.choice(pieces)

            if brd[piece_rank][piece_file] == " " and pawn_on_promotion_square(piece, piece_rank) is False:
                if piece in piece_counts and piece_counts[piece] < 2:
                    brd[piece_rank][piece_file] = piece
                    piece_counts[piece] += 1
                    piece_amount -= 1


def fen_from_board(brd):
	fen = ""
	for x in brd:
		n = 0
		for y in x:
			if y == " ":
				n += 1
			else:
				if n != 0:
					fen += str(n)
				fen += y
				n = 0
		if n != 0:
			fen += str(n)
		fen += "/" if fen.count("/") < 7 else ""
	fen += " w - - 0 1\n"
	return fen

def pawn_on_promotion_square(pc, pr):
	if pc == "P" and pr == 0:
		return True
	elif pc == "p" and pr == 7:
		return True
	return False
def save_board_image_as_png(fen, random_move):
    board_image = fenToImage(
        fen=fen,
        squarelength=100,
        pieceSet=loadPiecesFolder("/Users/abhinavgoel/PycharmProjects/Fen-To-Board-Imge/piece_set_name"),
        darkColor="#D18B47",
        lightColor="#FFCE9E"
    )

    # Save the image with a filename that includes the random move
    image_filename = f"board_{random_move}.png"
    board_image.save(image_filename, "PNG")


def start():
    piece_amount_white, piece_amount_black = 6, 6
    place_kings(board)
    populate_board(board, piece_amount_white, piece_amount_black)
    fen = fen_from_board(board)
    print("Random FEN:", fen)
    initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    save_board_image_as_png(fen, "original")
    moves_to_target = get_moves_to_fen(initial_fen, fen)
    for move in moves_to_target:
        print(move.uci())

#entry point
start()