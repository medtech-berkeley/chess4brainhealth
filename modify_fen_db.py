import pandas as pd
import chess

def drop_non_matein1_rows(input_file_path, output_file_path):
    # Read the CSV file
    df = pd.read_csv(input_file_path)

    # Create an empty DataFrame to store filtered rows
    df_filtered = pd.DataFrame(columns=df.columns)

    # Filter rows that contain the word "matein1" in col7 and add to df_filtered
    for index, row in df.iterrows():
        col7_values = str(row["col7"]).split()  # Split the values in col7 by spaces
        if any("mateIn1" in word for word in col7_values):
            df_filtered = pd.concat([df_filtered, row.to_frame().T], ignore_index=True)

    # Save the filtered data to a new CSV file
    df_filtered.to_csv(output_file_path, index=False)


def add_na_to_10_value_rows(input_file_path, output_file_path):
    # Manually specify filler column names
    filler_column_names = ['col' + str(i) for i in range(11)]

    # Read the CSV file and label the columns with filler names
    df = pd.read_csv(input_file_path, names=filler_column_names)

    # Check for rows with only 10 values
    mask = df.apply(lambda row: len(row) == 10, axis=1)
    rows_with_10_values = df[mask]

    # Add "N/A" as the last value in the last column for rows with 10 values
    df.loc[mask, 'col10'] = df.loc[mask, 'col10'] + ',N/A'

    # Save the modified data to a new CSV file
    df.to_csv(output_file_path, index=False)

def keep_selected_columns(input_file_path, output_file_path):
    # Read the CSV file
    df = pd.read_csv(input_file_path)

    # Keep only columns "col1", "col2", and "col7"
    df_selected = df[["col1"]]

    # Save the selected columns to a new CSV file
    df_selected.to_csv(output_file_path, index=False)
def update_fen_with_move(input_file_path, output_file_path):
    # Read the CSV file
    df = pd.read_csv(input_file_path)

    # Apply the move in move1 to the FEN in col1
    for index, row in df.iterrows():
        board = chess.Board(row['col1'])
        move = chess.Move.from_uci(row['move1'])
        board.push(move)
        updated_fen = board.fen()
        df.at[index, 'col1'] = updated_fen

    # Save the updated data to a new CSV file
    df.to_csv(output_file_path, index=False)

# Example usage:
update_fen_with_move("lichess_db_puzzle_filtered6.csv", "lichess_db_puzzle_filtered7.csv")
keep_selected_columns("lichess_db_puzzle_filtered7.csv", "lichess_db_puzzle_filtered7.csv")
