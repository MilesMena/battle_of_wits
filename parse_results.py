import csv
import re
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pointbiserialr




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

        print("==========Known ANS============")
        match_any = self.df_output['matches_any'].sum()
        onlyA = self.df_output['onlyA'].sum()
        onlyB = self.df_output['onlyB'].sum()
        print(f"ANS format matches: {match_any} out of {self.df_output.shape[0]}")
        print(f"Only A: {onlyA}, Only B: {onlyB}")
        print(f"Total Known ANS: {self.df_output[['onlyA','onlyB','matches_any']].any(axis = 1).sum()}")
        print()
        print(f"Correct % (remove non-answers): {self.df_output['Correct'].sum()/(self.df_output.shape[0] - (self.df_output['Final_Answer'] == '').sum())}")
        print(f"Correct %: {self.df_output['Correct'].sum()/self.df_output.shape[0]}")
        print()

    def count_questions(self):
        self.df_output['Question_Mark_Count'] = self.df_output['Questions'].astype(str).apply(lambda x: len(re.findall(r'\?', x)))
        # print(self.df_output['Qe'])

    def write_parse_results(self):
        self.df_output[['Iteration',
                        'Correct',
                        'Location',
                        'Final_Answer',
                        'Defending Disposition',
                        'Asking Belief',
                        'Questions',
                        'Response',
                        'Answer','Question_Mark_Count',
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

    def chi_squared_test(self):
        """
        Applies a Chi-squared test of independence to the pivot table of
        Correct counts grouped by Asking Belief and Defending Disposition.
        """
        # Create contingency table (pivot)
        df = self.df_output.copy()
        result = df.groupby(['Defending Disposition', 'Asking Belief'])['Correct'].sum().reset_index()
        pivot = result.pivot(index='Asking Belief', columns='Defending Disposition', values='Correct').fillna(0)

        # Apply chi-squared test
        chi2, p, dof, expected = chi2_contingency(pivot)

        # Print results
        print()
        print("=== Chi-Squared Test ===")
        print(f"Chi-squared statistic: {chi2:.3f}")
        print(f"Degrees of freedom: {dof}")
        print(f"p-value: {p:.5f}")
        print("\nObserved counts:")
        print(pivot)
        print("\nExpected counts:")
        print(pd.DataFrame(expected, index=pivot.index, columns=pivot.columns))

        # Optionally return values for logging or further processing
        return {
            "chi2": chi2,
            "p_value": p,
            "dof": dof,
            "observed": pivot,
            "expected": pd.DataFrame(expected, index=pivot.index, columns=pivot.columns)
        }
    
    def analyze_logistic_relationship(self):
        """
        Performs a logistic regression to determine if the number of questions
        predicts whether the answer was correct.
        """
        df = self.df_output.dropna(subset=['Correct', 'Question_Mark_Count']).copy()
        df['Correct'] = df['Correct'].astype(int)

        X = sm.add_constant(df['Question_Mark_Count'])  # Add intercept
        y = df['Correct']

        model = sm.Logit(y, X).fit(disp=False)
        print("=== Logistic Regression Results ===")
        print(model.summary())

        return model
    
    def visualize_question_count_by_correctness(self, save_path=None):
        """
        Creates a violin plot to visualize the number of questions asked
        grouped by whether the answer was correct.
        
        Parameters:
        - save_path (str): Optional. If provided, saves the plot to this path.
        """
        df = self.df_output.dropna(subset=['Correct', 'Question_Mark_Count']).copy()

        plt.figure(figsize=(8, 5))
        sns.violinplot(data=df, x='Correct', y='Question_Mark_Count', palette='Set2')
 
        plt.title(f"Number of Questions by Correctness {self.model, self.shot}")
        plt.xlabel("Correct (0 = False, 1 = True)")
        plt.ylabel("Question Mark Count")
        plt.tight_layout()

        os.makedirs(f"results/{model}/parsed_results/{date_time}/plots", exist_ok=True)
        plt.savefig(f"results/{model}/parsed_results/{date_time}/plots/shot_{self.shot}_count_by_correct.png")

        plt.close()

    def plot_logistic_regression(self, save_path=None):
        """
        Fits a logistic regression model and plots the predicted probability
        of a correct answer as a function of the number of question marks.
        
        Parameters:
        - save_path (str): Optional path to save the plot.
        """
        df = self.df_output.dropna(subset=['Correct', 'Question_Mark_Count']).copy()
        df['Correct'] = df['Correct'].astype(int)

        # Fit logistic regression
        X = sm.add_constant(df['Question_Mark_Count'])
        y = df['Correct']
        model = sm.Logit(y, X).fit(disp=False)

        # Generate range of X values and predict probabilities
        x_vals = np.linspace(df['Question_Mark_Count'].min(), df['Question_Mark_Count'].max(), 100)
        X_pred = sm.add_constant(x_vals)
        y_pred = model.predict(X_pred)

        # Plot
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x='Question_Mark_Count', y='Correct', alpha=0.6, label="Actual Data")
        plt.plot(x_vals, y_pred, color='red', linewidth=2, label="Logistic Regression")

        plt.title("Probability of Correctness by Number of Questions")
        plt.xlabel("Question Mark Count")
        plt.ylabel("Probability of Correct Answer")
        plt.ylim(-0.05, 1.05)
        plt.legend()
        plt.tight_layout()

        os.makedirs(f"results/{self.model}/parsed_results/{self.date_time}/plots", exist_ok=True)
        plt.savefig(f"results/{self.model}/parsed_results/{self.date_time}/plots/shot_{self.shot}_logistic_regression.png")

        plt.close()

    def plot_pointbiserial_relationship(self, save_path=None):
        """
        Visualizes the relationship captured by the point-biserial correlation
        between number of questions and correctness using a boxplot.
        
        Parameters:
        - save_path (str): Optional path to save the plot.
        """
        df = self.df_output.dropna(subset=['Correct', 'Question_Mark_Count']).copy()
        df['Correct'] = df['Correct'].astype(int)

        # Compute correlation
        corr, p = pointbiserialr(df['Correct'], df['Question_Mark_Count'])

        # Plot
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x='Correct', y='Question_Mark_Count', palette='Set2')
        sns.stripplot(data=df, x='Correct', y='Question_Mark_Count', color='black', alpha=0.4, jitter=0.2)

        plt.title(f"Point-Biserial Relationship\nCorr = {corr:.2f}, p = {p:.4f}")
        plt.xlabel("Correct (0 = False, 1 = True)")
        plt.ylabel("Question Mark Count")
        plt.tight_layout()

        os.makedirs(f"results/{self.model}/parsed_results/{self.date_time}/plots", exist_ok=True)
        plt.savefig(f"results/{self.model}/parsed_results/{self.date_time}/plots/shot_{self.shot}_point_bi.png")

        plt.close()

    
    
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
        res_parser.count_questions()
        res_parser.analyze_logistic_relationship()
        res_parser.visualize_question_count_by_correctness()
        res_parser.plot_logistic_regression()
        res_parser.plot_pointbiserial_relationship()
        res_parser.write_parse_results()
        res_parser.plot_single_heatmap(model, date_time, shot)
        res_parser.chi_squared_test()
        parsers.append(res_parser)
    
    plot_bars(parsers)
 