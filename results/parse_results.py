import csv
import re
import pandas as pd
import numpy as np



if __name__ == "__main__":
    model = "gemma:7b"
    output_path = f"/home/arpg-miles/code/battle_of_wits/{model}_output.csv"
    df = pd.read_csv(output_path)

    # Patterns for char/word matches
    pattern_A = r'\bA\b'
    pattern_B = r'\bB\b'
    pattern_ansB = r'\b(ans(?:wer)?[:\s]*(?:is\s*)?B)\b'
    pattern_ansA = r'\b(ans(?:wer)?[:\s]*(?:is\s*)?A)\b'
    # Count occurrences
    df['A_count'] = df['Answer'].str.count(pattern_A)
    df['B_count'] = df['Answer'].str.count(pattern_B)
    df['A_or_B_count'] = df['A_count'] + df['B_count']
    df['onlyA'] = (df['A_count'] == df['A_or_B_count'])
    df['onlyB'] = (df['B_count'] == df['A_or_B_count'])
    # Check if pattern matches
    df['matches_ansA'] = df['Answer'].str.contains(pattern_ansA, flags=re.IGNORECASE)
    df['matches_ansB'] = df['Answer'].str.contains(pattern_ansB, flags=re.IGNORECASE)
    df['matches_any'] = (df['matches_ansA'] | df['matches_ansB'])
    df['NO_ANS'] = ~df[['onlyA','onlyB','matches_any']].any(axis = 1)
    # Final answer
    df['Ground_Truth'] = ['A'] * df.shape[0]
    df['Final_Answer'] = np.where(df['matches_ansA'] | df['onlyA'], 'A', np.where(df['matches_ansB'] | df['onlyB'], 'B', ''))
    df['Correct'] = (df['Ground_Truth'] == df['Final_Answer'])

    print()
    match_any = df['matches_any'].sum()
    onlyA = df['onlyA'].sum()
    onlyB = df['onlyB'].sum()
    print(f"ANS format matches: {match_any} out of {df.shape[0]}")
    print(f"Only A: {onlyA}, Only B: {onlyB}")
    print(f"Total Known ANS: {df[['onlyA','onlyB','matches_any']].any(axis = 1).sum()}")
    print()
    print(f"Correct % (remove non-answers): {df['Correct'].sum()/(df.shape[0] - (df['Final_Answer'] == '').sum())}")
    print(f"Correct %: {df['Correct'].sum()/df.shape[0]}")
#     print()

    df[['Iteration', 
    'Execution Time (s)', 
    'Final_Answer',
    'A_count',
    'B_count',''
    'A_or_B_count',
    'onlyA',
    'onlyB',
    'matches_ansA',
    'matches_ansB',
    'matches_any',
    'NO_ANS',
    'Answer']].to_csv(f"{model}_parsed_answer.csv")









# import re
# import pandas as pd

# df = pd.DataFrame({
#     'answers': [
#         'Answer: A is my choice.',
#         'Answer A is correct.',
#         'ANSWER A is selected.',
#         'ANSWER: A works.',
#         'ans A is fine.',
#         'ans: A confirmed.',
#         'No answer here.',
#         'ans: C not A.'
#         'Answer is A.',
#         'answer is A',
#         'ANSWER is A.',
#         'ans: A is correct.',
#         'ans is A',
#         'No answer here.',
#         'ans: C not A.'
#     ]
# })

# pattern = r'\b(ans(?:wer)?[:]?)[ ]*A\b'
# pattern_p2 = r'\b(ans(?:wer)?[:\s]*is[:\s]*A)\b'
# pattern_both = r'\b(ans(?:wer)?[:\s]*(?:is\s*)?A)\b'


# # Check if pattern matches
# df['matches_A'] = df['answers'].str.contains(pattern, flags=re.IGNORECASE)

# # Check if pattern matches
# df['matches_A_p2'] = df['answers'].str.contains(pattern_p2, flags=re.IGNORECASE)

# df['matches_A_both'] = df['answers'].str.contains(pattern_both, flags=re.IGNORECASE)


# print(df)
