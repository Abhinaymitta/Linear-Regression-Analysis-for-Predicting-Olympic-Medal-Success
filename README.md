# Linear-Regression-Analysis-for-Predicting-Olympic-Medal-Success
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score


# Load the data
teams = pd.read_excel("OlympicFinal.xlsx")

# Ensure there are no NaN or infinite values in the entire DataFrame
teams = teams.replace([pd.NA, pd.NaT, float('inf'), -float('inf')], 0)

# Check for NaN values in the entire DataFrame
print("NaN values in the dataset:")
print(teams.isna().sum())

# Ensure there are no NaN values in the target column and predictors
print("Checking NaN values in predictors:")
print(teams[["Athletes", "Previous Medals"]].isna().sum())
print("Checking NaN values in target:")
print(teams["Medals"].isna().sum())

# Drop rows with NaN values in the predictors or target column
teams = teams.dropna(subset=["Athletes", "Previous Medals", "Medals"])

# Calculate the correlation matrix for the numeric columns
numeric_cols = teams.select_dtypes(include=['number'])
correlation_matrix = numeric_cols.corr()
print(correlation_matrix["Medals"])

# Plotting
sns.lmplot(x="Athletes", y="Medals", data=teams, fit_reg=True, ci=None)
teams.plot.hist(y="Medals")  # Shows that many countries are getting fewer medals; only a few getting a high number of medals

# Checking for null values
new_var = teams[teams.isnull().any(axis=1)]
print(new_var)  # Verify there are no remaining null values

# Split the data into train and test sets based on the 'Year' column
train = teams[teams["Year"] < 2020].copy()
test = teams[teams["Year"] >= 2020].copy()

# Display the shape of the train and test sets
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Initialize and fit the Linear Regression model
reg = LinearRegression()
predictors = ["Athletes", "Previous Medals"]
target = "Medals"

# Fit the model
reg.fit(train[predictors], train[target])

# Make predictions
predictions = reg.predict(test[predictors])
test["predictions"] = predictions

# Calculate the mean absolute error
error = mean_absolute_error(test[target], test["predictions"])
print("Mean Absolute Error:", error)

# Display statistics of Medals
print(teams.describe()["Medals"])

# Display test results for a specific team
print(test[test["Team"] == "USA"])

# Calculate errors and error ratios
errors = (test[target] - test["predictions"]).abs()
error_by_team = errors.groupby(test["Team"]).mean()
medals_by_team = test[target].groupby(test["Team"]).mean()
error_ratio = error_by_team / medals_by_team

# Print results
print("Error by Team:")
print(error_by_team)
print("Error Ratio:")
print(error_ratio)

# Cross-validation with 5 folds
cv_scores = cross_val_score(reg, teams[predictors], teams[target], cv=5, scoring='neg_mean_absolute_error')
print("Cross-Validation Scores:", -cv_scores)
print("Mean Cross-Validation Score:", -cv_scores.mean())
error = mean_absolute_error(test[target], test["predictions"])
r2 = r2_score(test[target], test["predictions"])

print("Mean Absolute Error:", error)
print("R-squared:", r2)
