def is_drawn_game(block):
    return block.strip().endswith("1/2-1/2")

def extract_blocks(file_path):
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()

    blocks = []
    current_block = []
    for line in lines:
        if line.strip() == "":
            blocks.append("\n".join(current_block))
            current_block = []
        else:
            current_block.append(line)

    return blocks

def remove_drawn_games(file_path, output_file_path):
    blocks = extract_blocks(file_path)

    with open(output_file_path, 'w') as output_file:
        for i in range(len(blocks)):
            if not is_drawn_game(blocks[i]):
                output_file.write(blocks[i] + "\n\n")

input_file_path = "Anand_NoDraws.pgn"
output_file_path = "Anand_NoDraws_NoDraws.pgn"
remove_drawn_games(input_file_path, output_file_path)
print("Blocks with drawn games removed. New file saved as 'Anand_NoDraws_NoDraws.pgn'.")
