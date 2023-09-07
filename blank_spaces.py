import pandas as pd
import numpy as np
infile = 'lichess_db_puzzle_filtered7.csv'

df = pd.read_csv(infile)
print(df)
blank_spaces = []
for board in df['col1'].tolist():
    blanks = 0
    for c in board:
        if c >= '1' and c <= '9':
            blanks += int(c)
    blank_spaces.append(blanks)
df['Blank Spaces'] = blank_spaces
print(df)
print(np.percentile(blank_spaces, 25))
print(np.percentile(blank_spaces, 50))
print(np.percentile(blank_spaces, 75))



    


