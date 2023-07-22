import berserk
import chess
from chess import Board

# Replace 'YOUR_API_TOKEN' with your actual Lichess API token
session = berserk.TokenSession("lip_g25x0ULdqVHyKXfxRWeG")
client = berserk.Client(session=session)


# Function to play moves with the bot
def play_moves_with_bot(game_id, moves):
    board = chess.Board()

    for move in moves:
        # Make the move on the local board
        board.push_uci(move)

        # Send the move to the bot via the Lichess API
        response = client.board.post_game_move(game_id, move)
        if not response['ok']:
            print(f"Failed to make move {move} with the bot.")
            return

        print(f"Your move: {move}")

        # Wait for the bot to make its move and get the updated game state
        game_state = client.board.get_game_state(game_id)
        if game_state['status'] != 'started':
            print(f"The game with the bot is finished. Result: {game_state['status']}")
            return

        bot_move = game_state['moves'].split()[-1]
        board.push_uci(bot_move)
        print(f"Bot move: {bot_move}")


# Sample list of moves (replace with your desired moves)
moves_to_play = ["e2e4", "e7e5", "g1f3", "b8c6"]

# Start the game with your bot and get the game ID
response = client.board.make_bot_game('ChessBrainHealthBot', 'white')
if response['ok']:
    game_id = response['game']['id']
    play_moves_with_bot(game_id, moves_to_play)
else:
    print("Failed to start the game with the bot.")
