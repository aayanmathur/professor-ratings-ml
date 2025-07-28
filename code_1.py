import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import roc_curve, roc_auc_score
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2_contingency

# Seed RNG with N-number 
np.random.seed(12972469)

# Load both datasets with column names
df_num = pd.read_csv('data/rmpCapstoneNum.csv', names=[
    'average_rating', 'average_difficulty', 'number_of_ratings',
    'has_pepper', 'would_take_again', 'online_ratings', 'is_male', 'is_female'
])

df_qual = pd.read_csv('data/rmpCapstoneQual.csv', names=[
    'major', 'university', 'state'
])

# Remove rows with NaN in critical columns
critical_columns = [
    'average_rating', 'average_difficulty', 'number_of_ratings',
    'has_pepper', 'online_ratings', 'is_male', 'is_female'
]

print("\nPreprocessing:")
print(f"\nRows before removing NaN: {len(df_num)}")
df_num = df_num.dropna(subset=critical_columns)
print(f"Rows after removing NaN: {len(df_num)}")

# Apply minimum ratings threshold
print(f"Rows before applying threshold: {len(df_num)}")
mask = df_num['number_of_ratings'] >= 8
df_num = df_num[mask]
print(f"Rows after applying threshold: {len(df_num)}")

# Apply same filtering to qualitative data
df_qual = df_qual.loc[df_num.index]


# Start of EDA for Question 1 
print("\n Question 1 EDA ")

# Remove invalid gender combinations
valid_gender = ~((df_num['is_male'] == 0) & (df_num['is_female'] == 0) | 
                 (df_num['is_male'] == 1) & (df_num['is_female'] == 1))
df_gender = df_num[valid_gender].copy()

# Create male and female rating groups
male_ratings = df_gender[df_gender['is_male'] == 1]['average_rating']
female_ratings = df_gender[df_gender['is_female'] == 1]['average_rating']

# Print basic statistics
print("\nSample Sizes:")
print(f"Male professors: {len(male_ratings)}")
print(f"Female professors: {len(female_ratings)}")

print("\nMale Ratings Statistics:")
print(f"Mean: {male_ratings.mean():.3f}")
print(f"Standard Deviation: {male_ratings.std():.3f}")

print("\nFemale Ratings Statistics:")
print(f"Mean: {female_ratings.mean():.3f}")
print(f"Standard Deviation: {female_ratings.std():.3f}")

# Create histograms
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(male_ratings, bins=30, alpha=0.7, label='Male')
plt.title('Distribution of Male Professor Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(female_ratings, bins=30, alpha=0.7, label='Female')
plt.title('Distribution of Female Professor Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig("Question_1_EDA")
plt.close()


# QUESTION 1 

# Create male and female rating groups
male_ratings = df_gender[df_gender['is_male'] == 1]['average_rating']
female_ratings = df_gender[df_gender['is_female'] == 1]['average_rating']

# Perform Mann-Whitney U test
statistic, p_value = stats.mannwhitneyu(male_ratings, female_ratings, 
                                      alternative='two-sided')

# Calculate effect size (r = Z / sqrt(N))
n1, n2 = len(male_ratings), len(female_ratings)
z_score = statistic - (n1 * n2 / 2)  # Convert U to Z-score
z_score = z_score / np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
effect_size = abs(z_score) / np.sqrt(n1 + n2)

print("\nQuestion 1")
print("\nMann-Whitney U Test Results")
print(f"Male professors (n={len(male_ratings)}):")
print(f"Mean rating: {male_ratings.mean():.3f}")
print(f"Median rating: {male_ratings.median():.3f}")

print(f"\nFemale professors (n={len(female_ratings)}):")
print(f"Mean rating: {female_ratings.mean():.3f}")
print(f"Median rating: {female_ratings.median():.3f}")

print("\nTest Results:")
print(f"U-statistic: {statistic:.0f}")
print(f"p-value: {p_value:.6f}")
print(f"Effect size (r): {effect_size:.3f}")

# Create boxplot for visualization
plt.figure(figsize=(8, 6))
sns.boxplot(data=pd.DataFrame({
    'Rating': pd.concat([male_ratings, female_ratings]),
    'Gender': ['Male']*len(male_ratings) + ['Female']*len(female_ratings)
}), x='Gender', y='Rating')
plt.title('Professor Ratings by Gender')
plt.savefig("Question_1_box_plot")
plt.close()
# EDA FOR QUESTION 2

# Create scatter plot to check linearity and outliers
plt.figure(figsize=(10, 6))
plt.scatter(df_num['number_of_ratings'], df_num['average_rating'], alpha=0.5)
plt.xlabel('Number of Ratings (Experience)')
plt.ylabel('Average Rating (Quality)')
plt.title('Teaching Quality vs Experience')

# Create histograms to check normality
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Distribution of number of ratings
ax1.hist(df_num['number_of_ratings'], bins=30)
ax1.set_title('Distribution of Number of Ratings')
ax1.set_xlabel('Number of Ratings')
ax1.set_ylabel('Frequency')

# Distribution of average ratings
ax2.hist(df_num['average_rating'], bins=30)
ax2.set_title('Distribution of Average Ratings')
ax2.set_xlabel('Average Rating')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig("Question_2_EDA")
plt.close()


# Print basic statistics
print("\nSummary Statistics:")
print("\nNumber of Ratings:")
print(df_num['number_of_ratings'].describe())
print("\nAverage Rating:")
print(df_num['average_rating'].describe())



# QUESTION 2

print("\nQuestion 2")

# Calculate Spearman correlation
correlation, p_value = stats.spearmanr(df_num['number_of_ratings'], 
                                     df_num['average_rating'])

print("Spearman Correlation Results:")
print(f"Correlation coefficient: {correlation:.3f}")
print(f"P-value: {p_value:.6f}")

# Calculate effect size (r²)
effect_size = correlation ** 2
print(f"Effect size (r²): {effect_size:.3f}")

# Create scatter plot with trend line for visualization
plt.figure(figsize=(10, 6))
sns.regplot(data=df_num, x='number_of_ratings', y='average_rating', 
            scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
plt.xlabel('Number of Ratings (Experience)')
plt.ylabel('Average Rating (Quality)')
plt.title('Teaching Quality vs Experience\nwith Trend Line')
plt.savefig("Question_2_scatterplot")
plt.close()

# EDA for Question 3 

print("\nEDA for question 3")

# Create visualizations on same plot
fig, ((ax1, ax2), (ax3, ax3_twin)) = plt.subplots(2, 2, figsize=(15, 10))

# Histogram of ratings
ax1.hist(df_num['average_rating'], bins=30)
ax1.set_title('Distribution of Average Ratings')
ax1.set_xlabel('Rating')
ax1.set_ylabel('Frequency')

# Histogram of difficulty
ax2.hist(df_num['average_difficulty'], bins=30)
ax2.set_title('Distribution of Average Difficulty')
ax2.set_xlabel('Difficulty')
ax2.set_ylabel('Frequency')

# Scatter plot
ax3.scatter(df_num['average_difficulty'], df_num['average_rating'], alpha=0.5)
ax3.set_xlabel('Average Difficulty')
ax3.set_ylabel('Average Rating')
ax3.set_title('Rating vs Difficulty')

# Add linear trend line
z = np.polyfit(df_num['average_difficulty'], df_num['average_rating'], 1)
p = np.poly1d(z)
ax3.plot(df_num['average_difficulty'], p(df_num['average_difficulty']), "r--", alpha=0.8)

plt.tight_layout()
plt.savefig("Question_3_EDA")
plt.close()

# Print summary statistics
print("\nSummary Statistics:")
print("\nAverage Rating:")
print(df_num['average_rating'].describe())
print("\nAverage Difficulty:")
print(df_num['average_difficulty'].describe())

# QUESTION 3 

print("\nQUESTION 3")

# Calculate Spearman correlation
correlation, p_value = stats.spearmanr(df_num['average_rating'], 
                                     df_num['average_difficulty'])

print("Spearman Correlation Results:")
print(f"Correlation coefficient: {correlation:.3f}")
print(f"P-value: {p_value:.6f}")

# Calculate effect size (r²)
effect_size = correlation ** 2
print(f"Effect size (r²): {effect_size:.3f}")

# Create scatter plot with trend line
plt.figure(figsize=(10, 6))
sns.regplot(data=df_num, x='average_difficulty', y='average_rating', 
            scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
plt.xlabel('Average Difficulty')
plt.ylabel('Average Rating')
plt.title('Rating vs Difficulty\nwith Trend Line')
plt.savefig("Question_3_scatterplot")
plt.close()

# Question 4 EDA

print("Question 4 EDA")

# Calculate proportion of online ratings
df_num['online_ratio'] = df_num['online_ratings'] / df_num['number_of_ratings']

# Plot distribution of online ratio
plt.figure(figsize=(10, 6))
plt.hist(df_num['online_ratio'], bins=50)
plt.title('Distribution of Online Teaching Ratio')
plt.xlabel('Proportion of Online Ratings')
plt.ylabel('Frequency')
plt.savefig("Question_4_EDA")
plt.close()

# Print summary statistics
print("\nOnline Teaching Statistics:")
print("\nRaw online ratings:")
print(df_num['online_ratings'].describe())
print("\nOnline ratio (proportion):")
print(df_num['online_ratio'].describe())

# Print counts of professors with different online proportions
thresholds = [0, 0.25, 0.5, 0.75, 1]
for i in range(len(thresholds)-1):
    count = len(df_num[(df_num['online_ratio'] > thresholds[i]) & 
                       (df_num['online_ratio'] <= thresholds[i+1])])
    print(f"\nProfessors with {thresholds[i]*100}%-{thresholds[i+1]*100}% online ratings: {count}")


# Create the two groups
online = df_num[df_num['online_ratings'] > 0]['average_rating']
no_online = df_num[df_num['online_ratings'] == 0]['average_rating']

# Create histograms
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(online, bins=30)
ax1.set_title('Ratings Distribution\nProfessors with Online Teaching')
ax1.set_xlabel('Rating')
ax1.set_ylabel('Frequency')

ax2.hist(no_online, bins=30)
ax2.set_title('Ratings Distribution\nProfessors without Online Teaching')
ax2.set_xlabel('Rating')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig("Question_4_EDA_2")
plt.close()

# Print basic statistics
print("\nOnline Teaching Group:")
print(online.describe())
print("\nNo Online Teaching Group:")
print(no_online.describe())

# Question 4 

print("\nQuestion 4")

# Create groups
online = df_num[df_num['online_ratings'] > 0]['average_rating']
no_online = df_num[df_num['online_ratings'] == 0]['average_rating']

# Perform Mann-Whitney U test
statistic, p_value = stats.mannwhitneyu(online, no_online, alternative='two-sided')

# Calculate effect size (r = Z / sqrt(N))
n1, n2 = len(online), len(no_online)
z_score = statistic - (n1 * n2 / 2)
z_score = z_score / np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
effect_size = abs(z_score) / np.sqrt(n1 + n2)

# Print results
print("Mann-Whitney U Test Results:")
print(f"\nOnline Teaching Group (n={len(online)}):")
print(f"Mean rating: {online.mean():.3f}")
print(f"Median rating: {online.median():.3f}")

print(f"\nNo Online Teaching Group (n={len(no_online)}):")
print(f"Mean rating: {no_online.mean():.3f}")
print(f"Median rating: {no_online.median():.3f}")

print("\nTest Results:")
print(f"U-statistic: {statistic:.0f}")
print(f"p-value: {p_value:.6f}")
print(f"Effect size (r): {effect_size:.3f}")

# Create box plot
plt.figure(figsize=(8, 6))
sns.boxplot(data=pd.DataFrame({
    'Rating': pd.concat([online, no_online]),
    'Group': ['Online']*len(online) + ['No Online']*len(no_online)
}), x='Group', y='Rating')
plt.title('Professor Ratings by Teaching Mode')
plt.savefig("Question_4_boxplot")
plt.close()


# QUESTION 5 EDA
print("\nQ5 EDA")


# First check NaN situation
total_profs = len(df_num)
nan_profs = df_num['would_take_again'].isna().sum()
valid_profs = total_profs - nan_profs

print("Data Availability:")
print(f"Total professors: {total_profs}")
print(f"Professors with missing 'would take again' data: {nan_profs}")
print(f"Professors with valid 'would take again' data: {valid_profs}")
print(f"Percentage of data available: {(valid_profs/total_profs)*100:.1f}%")

# Create analysis using only valid data
df_clean = df_num.dropna(subset=['would_take_again'])

# Create visualizations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Distribution of ratings (for professors with valid would_take_again)
ax1.hist(df_clean['average_rating'], bins=30)
ax1.set_title('Distribution of Ratings\n(Professors with Would Take Again data)')
ax1.set_xlabel('Average Rating')
ax1.set_ylabel('Frequency')

# Distribution of would_take_again
ax2.hist(df_clean['would_take_again'], bins=30)
ax2.set_title('Distribution of Would Take Again')
ax2.set_xlabel('Proportion Who Would Take Again')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig("Question_5_EDA")
plt.close()


# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df_clean['would_take_again'], df_clean['average_rating'], alpha=0.5)
plt.xlabel('Proportion Who Would Take Again')
plt.ylabel('Average Rating')
plt.title('Rating vs Would Take Again')
plt.savefig("Question_5_EDA_2")
plt.close()

# Print statistics for valid data
print("\nSummary Statistics for Valid Data:")
print("\nAverage Rating:")
print(df_clean['average_rating'].describe())
print("\nWould Take Again (%):")
print(df_clean['would_take_again'].describe())

# QUESTION 5
print("\nQuestion 5")

# Calculate Spearman correlation
correlation, p_value = stats.spearmanr(df_clean['would_take_again'], 
                                     df_clean['average_rating'])

# Calculate effect size (r²)
effect_size = correlation ** 2

print("Spearman Correlation Results:")
print(f"Correlation coefficient (ρ): {correlation:.3f}")
print(f"P-value: {p_value:.6f}")
print(f"Effect size (r²): {effect_size:.3f}")

# Create scatter plot with trend line
plt.figure(figsize=(10, 6))
sns.regplot(data=df_clean, x='would_take_again', y='average_rating', 
            scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
plt.xlabel('Proportion Who Would Take Again (%)')
plt.ylabel('Average Rating')
plt.title('Rating vs Would Take Again\nwith Trend Line')
plt.savefig("Question_5_scatterplot")
plt.close()


# QUESTION 6 EDA 

print("\nQuestion 6 EDA")


# Create the two groups
hot = df_num[df_num['has_pepper'] == 1]['average_rating']
not_hot = df_num[df_num['has_pepper'] == 0]['average_rating']

# Print sample sizes
print("Sample Sizes:")
print(f"Hot professors: {len(hot)}")
print(f"Not hot professors: {len(not_hot)}")

# Create histograms
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(hot, bins=30)
ax1.set_title('Ratings Distribution\nHot Professors')
ax1.set_xlabel('Rating')
ax1.set_ylabel('Frequency')

ax2.hist(not_hot, bins=30)
ax2.set_title('Ratings Distribution\nNot Hot Professors')
ax2.set_xlabel('Rating')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig("Question_6_EDA")
plt.close()

# Print basic statistics
print("\nRating Statistics:")
print("\nHot Professors:")
print(hot.describe())
print("\nNot Hot Professors:")
print(not_hot.describe())

# QUESTION 6

print("\nQuestion 6")

# Create groups
hot = df_num[df_num['has_pepper'] == 1]['average_rating']
not_hot = df_num[df_num['has_pepper'] == 0]['average_rating']

# Perform Mann-Whitney U test
statistic, p_value = stats.mannwhitneyu(hot, not_hot, alternative='two-sided')

# Calculate effect size
n1, n2 = len(hot), len(not_hot)
z_score = statistic - (n1 * n2 / 2)
z_score = z_score / np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
effect_size = abs(z_score) / np.sqrt(n1 + n2)

# Print results
print("Mann-Whitney U Test Results:")
print(f"\nHot professors (n={len(hot)}):")
print(f"Mean rating: {hot.mean():.3f}")
print(f"Median rating: {hot.median():.3f}")

print(f"\nNot hot professors (n={len(not_hot)}):")
print(f"Mean rating: {not_hot.mean():.3f}")
print(f"Median rating: {not_hot.median():.3f}")

print("\nTest Results:")
print(f"U-statistic: {statistic:.0f}")
print(f"p-value: {p_value:.6f}")
print(f"Effect size (r): {effect_size:.3f}")

# Create box plot
plt.figure(figsize=(8, 6))
sns.boxplot(data=pd.DataFrame({
    'Rating': pd.concat([hot, not_hot]),
    'Group': ['Hot']*len(hot) + ['Not Hot']*len(not_hot)
}), x='Group', y='Rating')
plt.title('Professor Ratings by Hotness')
plt.savefig("Question_6_boxplot")
plt.close()

# QUESTION 7 

print("\nQuestion 7")
# Prepare data
X = df_num[['average_difficulty']]
y = df_num['average_rating']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12972469)

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate metrics on test set
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print results
print("Test Set Results:")
print(f"R²: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"Intercept: {model.intercept_:.3f}")
print(f"Coefficient: {model.coef_[0]:.3f}")

# Scatter plot with regression line
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.5)
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.xlabel('Average Difficulty')
plt.ylabel('Average Rating')
plt.title('Rating vs Difficulty (Test Set)')
plt.savefig("Question_7_scatterplot")
plt.close()


# QUESTION 8 

print("\nQuestion 8 EDA")
# Prepare features
features = ['average_difficulty', 'number_of_ratings', 'has_pepper', 
           'would_take_again', 'online_ratings', 'is_male', 'is_female']

# Calculate correlation matrix
correlation_matrix = df_clean[features].corr()

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, fmt='.3f')
plt.title('Correlation Matrix of Predictors')
plt.tight_layout()
plt.savefig("Question_8_correlation_matrix")
plt.close()

print("\nCorrelation Matrix:")
print(correlation_matrix)

print("\nQuestion 8")

# Prepare features
features = ['average_difficulty', 'number_of_ratings', 'has_pepper', 
           'would_take_again', 'online_ratings', 'is_male', 'is_female']

X = df_clean[features]
y = df_clean['average_rating']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12972469)

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print results
print("Test Set Results:")
print(f"R²: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")
print("\nFeature Coefficients:")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:.3f}")
print(f"Intercept: {model.intercept_:.3f}")

# Compare with simple model on same test set
X_simple = X_test[['average_difficulty']]
simple_model = LinearRegression()
simple_model.fit(X_train[['average_difficulty']], y_train)
y_pred_simple = simple_model.predict(X_simple)
r2_simple = r2_score(y_test, y_pred_simple)
rmse_simple = np.sqrt(mean_squared_error(y_test, y_pred_simple))

print("\nSimple Model Test Results:")
print(f"R²: {r2_simple:.3f}")
print(f"RMSE: {rmse_simple:.3f}")

# Actual vs Predicted plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual vs Predicted Ratings')
plt.tight_layout()
plt.savefig("Question_8_plot")
plt.close()


# QUESTION 9

print("\nQuestion 9")

# Prepare data
X = df_num[['average_rating']]
y = df_num['has_pepper']

# Check class distribution
print("Class Distribution:")
print(y.value_counts(normalize=True) * 100)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12972469)

# Fit model
model = LogisticRegression()
model.fit(X_train, y_train)

# Get predictions on test set
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate AUC-ROC on test set
auc_roc = roc_auc_score(y_test, y_pred_proba)

# Get ROC curve points
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc_roc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig("Question_9_ROC_curve")
plt.close()

# Print results
print("\nModel Results:")
print(f"AUC-ROC: {auc_roc:.3f}")
print(f"Coefficient: {model.coef_[0][0]:.3f}")
print(f"Intercept: {model.intercept_[0]:.3f}")

# Question 10 

print("\nQuestion 10")

# Prepare features
features = ['average_rating', 'average_difficulty', 'number_of_ratings',
           'would_take_again', 'online_ratings', 'is_male', 'is_female']

X = df_clean[features]
y = df_clean['has_pepper']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12972469)

# Fit full model
model = LogisticRegression()
model.fit(X_train, y_train)

# Get predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate AUC-ROC
auc_roc = roc_auc_score(y_test, y_pred_proba)

# Fit simple model for comparison
X_simple = df_clean[['average_rating']]
X_simple_train, X_simple_test, y_simple_train, y_simple_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=12972469)

model_simple = LogisticRegression()
model_simple.fit(X_simple_train, y_simple_train)
y_pred_proba_simple = model_simple.predict_proba(X_simple_test)[:, 1]
auc_roc_simple = roc_auc_score(y_simple_test, y_pred_proba_simple)

# Plot ROC curves
plt.figure(figsize=(8, 6))
# Full model curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'Full Model (AUC = {auc_roc:.3f})')

# Simple model curve
fpr_simple, tpr_simple, _ = roc_curve(y_simple_test, y_pred_proba_simple)
plt.plot(fpr_simple, tpr_simple, '--', label=f'Rating Only (AUC = {auc_roc_simple:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Full vs Simple Model')
plt.legend()
plt.savefig("Question_10_ROC_curve")
plt.close()

# Print results
print("\nFull Model Results:")
print("\nFeature Coefficients:")
for feature, coef in zip(features, model.coef_[0]):
    print(f"{feature}: {coef:.3f}")
print(f"Intercept: {model.intercept_[0]:.3f}")
print(f"AUC-ROC: {auc_roc:.3f}")

print("\nSimple Model AUC-ROC:", auc_roc_simple)


# Extra Credit 
print("EXTRA CREDIT")

# Create DataFrame with major and pepper info
df_major_hot = pd.DataFrame({
    'major': df_qual['major'],
    'has_pepper': df_num['has_pepper']
})

# Calculate pepper percentage for each major
major_stats = df_major_hot.groupby('major').agg({
    'has_pepper': ['count', 'mean']
}).sort_values(('has_pepper', 'count'), ascending=False)

# Filter for majors with at least 50 professors
major_stats = major_stats[major_stats[('has_pepper', 'count')] >= 50]

# Sort by pepper percentage
major_stats = major_stats.sort_values(('has_pepper', 'mean'), ascending=False)

# Convert mean to percentage
major_stats[('has_pepper', 'percentage')] = major_stats[('has_pepper', 'mean')] * 100

print("Pepper Distribution by Major (minimum 50 professors):")
print("\nTop 10 Majors by Pepper Percentage:")
print(major_stats[('has_pepper', 'percentage')].head(10))
print("\nBottom 10 Majors by Pepper Percentage:")
print(major_stats[('has_pepper', 'percentage')].tail(10))

# Create bar plot of percentages
plt.figure(figsize=(12, 6))
plt.bar(range(len(major_stats)), major_stats[('has_pepper', 'percentage')])
plt.xticks(range(len(major_stats)), major_stats.index, rotation=45, ha='right')
plt.title('Percentage of Professors with Pepper by Major')
plt.ylabel('Percentage with Pepper')
plt.tight_layout()
plt.savefig("Extra_credit_barplot")
plt.close()


# Create contingency table
contingency = pd.crosstab(df_qual['major'], df_num['has_pepper'])

# Filter for majors with at least 50 professors
major_counts = contingency.sum(axis=1)
contingency = contingency[major_counts >= 50]

# Perform chi-square test
chi2, p_value, dof, expected = chi2_contingency(contingency)

print("\nChi-square Test Results:")
print(f"Chi-square statistic: {chi2:.3f}")
print(f"p-value: {p_value:.6f}")
print(f"Degrees of freedom: {dof}")

