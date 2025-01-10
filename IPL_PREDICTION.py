import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
file_path = '/content/ipl_colab.csv'
dataset = pd.read_csv(file_path)
features = ['runs', 'overs', 'runs_last_5']
target = 'total'
dataset_filtered = dataset.dropna(subset=features + [target])
X = dataset_filtered[features]
y = dataset_filtered[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("\nError Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print("\nModel Evaluation:")
print(f"Training Score (R²): {train_score:.2f}")
print(f"Test Score (R²): {test_score:.2f}")
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
numeric_data = dataset.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.subplot(2, 2, 2)
sns.histplot(dataset['runs'], bins=20, kde=True, color='blue')
plt.title("Distribution of Runs")
plt.subplot(2, 2, 3)
final_scores = dataset.groupby('batting_team')['total'].max().reset_index()
sns.barplot(x='batting_team', y='total', data=final_scores, palette='viridis')
plt.title("Batting Teams and Their Final Scores")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
print("\nEnter the values for prediction:")
runs = float(input("Runs: "))
overs = float(input("Overs: "))
runs_last_5 = float(input("Runs in last 5 overs: "))
user_input = pd.DataFrame([[runs, overs, runs_last_5]], columns=features)
predicted_total = model.predict(user_input)
batting_teams = dataset['batting_team'].unique()
bowling_teams = dataset['bowling_team'].unique()
print(f"\nSample Batting Teams: {batting_teams[:5]}")
print(f"Sample Bowling Teams: {bowling_teams[:5]}")
batting_team = batting_teams[0]
bowling_team = bowling_teams[0]
print(f"\nPredicted Batting Team: {batting_team}")
print(f"Predicted Bowling Team: {bowling_team}")
print(f"\nPredicted Total Score: {predicted_total[0]:.2f}")
