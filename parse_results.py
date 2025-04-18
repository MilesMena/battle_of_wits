import csv
import re
import pandas as pd
import numpy as np
import os
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class ResultParser:
    def __init__(self, model):
        rel_path = os.path.dirname(__file__)
        self.results_path = os.path.join(rel_path, f"results/{model}")
        self.df_output = pd.read_csv(f"{self.results_path}/{model}_output.csv")

    def parse_ans(self):
        # Patterns for char/word matches
        pattern_A = r'\bA\b'
        pattern_B = r'\bB\b'
        pattern_ansB = r'\b(ans(?:wer)?[:\s]*(?:is\s*)?B)\b'
        pattern_ansA = r'\b(ans(?:wer)?[:\s]*(?:is\s*)?A)\b'
        # Count occurrences
        self.df_output['A_count'] = self.df_output['Answer'].str.count(pattern_A)
        self.df_output['B_count'] = self.df_output['Answer'].str.count(pattern_B)
        self.df_output['A_or_B_count'] = self.df_output['A_count'] + self.df_output['B_count']
        self.df_output['onlyA'] = (self.df_output['A_count'] == self.df_output['A_or_B_count'])
        self.df_output['onlyB'] = (self.df_output['B_count'] == self.df_output['A_or_B_count'])
        # Check if pattern matches
        self.df_output['matches_ansA'] = self.df_output['Answer'].str.contains(pattern_ansA, flags=re.IGNORECASE)
        self.df_output['matches_ansB'] = self.df_output['Answer'].str.contains(pattern_ansB, flags=re.IGNORECASE)
        self.df_output['matches_any'] = (self.df_output['matches_ansA'] | self.df_output['matches_ansB'])
        self.df_output['NO_ANS'] = ~self.df_output[['onlyA','onlyB','matches_any']].any(axis = 1)
        # Final answer
        self.df_output['Ground_Truth'] = ['A'] * self.df_output.shape[0]
        self.df_output['Final_Answer'] = np.where(self.df_output['matches_ansA'] | self.df_output['onlyA'], 'A', np.where(self.df_output['matches_ansB'] | self.df_output['onlyB'], 'B', ''))
        self.df_output['Correct'] = (self.df_output['Ground_Truth'] == self.df_output['Final_Answer'])

        print()
        match_any = self.df_output['matches_any'].sum()
        onlyA = self.df_output['onlyA'].sum()
        onlyB = self.df_output['onlyB'].sum()
        print(f"ANS format matches: {match_any} out of {self.df_output.shape[0]}")
        print(f"Only A: {onlyA}, Only B: {onlyB}")
        print(f"Total Known ANS: {self.df_output[['onlyA','onlyB','matches_any']].any(axis = 1).sum()}")
        print()
        print(f"Correct % (remove non-answers): {self.df_output['Correct'].sum()/(self.df_output.shape[0] - (self.df_output['Final_Answer'] == '').sum())}")
        print(f"Correct %: {self.df_output['Correct'].sum()/self.df_output.shape[0]}")
    

    def write_parse_results(self):
        self.df_output[['Iteration',
                        'Correct',
                        'Location',
                        'Final_Answer',
                        'Defending Disposition',
                        'Asking Belief',
                        'Questions',
                        'Response',
                        'Answer',
                        'Execution Time (s)']].to_csv(f"{self.results_path}/{model}_parsed_answer.csv", index = False)
        print(f"Results written")

    def plot_heatmap(self):
        # labels = ['Truthful', 'Deceitful']
        # df = pd.DataFrame(index=labels, columns=labels)

        # Assuming self.df_output is your DataFrame
        df = self.df_output.copy()
        result = df.groupby(['Defending Disposition', 'Asking Belief'])['Correct'].sum().reset_index()
        result.rename(columns={'Correct': 'Correct_Count'}, inplace=True)

        print(result)

        pass


if __name__ == "__main__":
    model = "gemma:7b"

    res_parser = ResultParser(model)
    res_parser.parse_ans()
    res_parser.write_parse_results()
    res_parser.plot_heatmap()

    
#     print()

    









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
