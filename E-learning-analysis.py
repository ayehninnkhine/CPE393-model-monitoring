import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the CSV dataset
df = pd.read_csv("e_learning_quiz_scores.csv")

# Step 2: Separate the two groups
group_a_scores = df[df["Group"] == "A"]["Quiz_Score"]
group_b_scores = df[df["Group"] == "B"]["Quiz_Score"]

# Step 3: Perform independent two-sample t-test
t_stat, p_val = stats.ttest_ind(group_a_scores, group_b_scores)

# Step 4: Print the results
print("Group A Mean:", group_a_scores.mean())
print("Group B Mean:", group_b_scores.mean())
print("T-statistic:", t_stat)
print("P-value:", p_val)

# Step 5: Interpretation
if p_val < 0.05:
    print("Result is statistically significant. The new video content (Group B) is likely more effective.")
else:
    print("Result is not statistically significant. No strong evidence of a difference.")

# Step 6: Optional - Visualize the distribution
sns.boxplot(x="Group", y="Quiz_Score", data=df)
plt.title("Quiz Scores by Group")
plt.ylabel("Quiz Score")
plt.xlabel("Content Group")
plt.grid(True)
plt.show()
