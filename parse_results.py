import csv
import re
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class ResultParser:
    def __init__(self, model, date_time, shot):
        self.shot = shot
        self.model = model
        self.date_time = date_time
        rel_path = os.path.dirname(__file__)
        self.results_path = os.path.join(rel_path, f"results/{model}/prompt_shot_{shot}/{date_time}/")
        self.df_output = pd.read_csv(f"{self.results_path}/{model}_output.csv")

        os.makedirs(os.path.join(rel_path, f"{rel_path}/results/{model}/parsed_results/{date_time}"), exist_ok = True)

        self.write_path = f"{rel_path}/results/{model}/parsed_results/{date_time}/shot_{shot}.csv"

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
                        'Execution Time (s)']].to_csv(self.write_path, index = False)
        print(f"Results written")

    def plot_single_heatmap(self, model, date_time, shot):
        # labels = ['Truthful', 'Deceitful']
        # df = pd.DataFrame(index=labels, columns=labels)

        # Assuming self.df_output is your DataFrame
        df = self.df_output.copy()
        result = df.groupby(['Defending Disposition', 'Asking Belief'])['Correct'].sum().reset_index()
        result.rename(columns={'Correct': 'Correct_Count'}, inplace=True)

        # Pivot the grouped DataFrame (this was the incorrect line)
        pivot = result.pivot(index='Asking Belief', columns='Defending Disposition', values='Correct_Count')

        # Plot
        sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt='d')
        plt.title(f"{model}: Correct Count by Defending Disposition vs Asking Belief")
        plt.xlabel("Asking Belief")
        plt.ylabel("Defending Disposition")
        plt.tight_layout()
        os.makedirs(f"results/{model}/parsed_results/{date_time}/plots", exist_ok=True)
        plt.savefig(f"results/{model}/parsed_results/{date_time}/plots/prompt_shot_{shot}.png")
        plt.close()  # Close the figure after saving


        print(result)
    
def plot_bars(parsers):
 

    all_data = []

    for parse in parsers:
        df = parse.df_output.copy()
        shot = parse.shot
        model = parse.model  # assuming model name is stored in each parse
        date_time = parse.date_time  # assuming a consistent timestamp is available

        # Group and pivot like in plot_single_heatmap
        result = df.groupby(['Defending Disposition', 'Asking Belief'])['Correct'].sum().reset_index()
        result['Shot'] = shot  # include shot for grouping
        all_data.append(result)

    # Combine all results into a single DataFrame
    combined_df = pd.concat(all_data)

    # Aggregate total correct count per Shot
    summary_df = combined_df.groupby(['Shot', 'Defending Disposition', 'Asking Belief'])['Correct'].sum().reset_index()

    # Optional: Create a column for grouped labels
    summary_df['Condition'] = summary_df['Asking Belief'] + ' → ' + summary_df['Defending Disposition']

    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=summary_df, x='Condition', y='Correct', hue='Shot')
    plt.title(f"Correct Count by Condition (Asking → Defending) Across Shots ({model})")
    plt.xlabel("Condition")
    plt.ylabel("Correct Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save figure
    os.makedirs(f"results/{model}/parsed_results/{date_time}/plots", exist_ok=True)
    plt.savefig(f"results/{model}/parsed_results/{date_time}/plots/bar_plot.png")
    plt.close()

    print(summary_df)
    


if __name__ == "__main__":

    prompt_shots = 2
    model = "gemma:7b"
    date_time = "30-47-09-22-04-2025" # copy from the results
    parsers = []
    for shot in range(prompt_shots +1):
        res_parser = ResultParser(model, date_time, shot)
        res_parser.parse_ans()
        res_parser.write_parse_results()
        res_parser.plot_single_heatmap(model, date_time, shot)
        parsers.append(res_parser)
    
    plot_bars(parsers)
 