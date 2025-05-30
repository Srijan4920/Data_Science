import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create a directory to save plots
os.makedirs("plots", exist_ok=True)

# Step 1: Load the Data
df = pd.read_csv('weather.csv')



# Step 2: Data Exploration
print(df.head())
print(df.info())
print(df.describe())

# Step 3: Data Visualization
sns.pairplot(df[['MinTemp', 'MaxTemp', 'Rainfall']])
plt.savefig("plots/pairplot_temp_rainfall.png")
plt.show()

# Step 4: Feature Engineering (if needed)
# Handle missing 'Date' column
if 'Date' not in df.columns:
    df['Date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Month'] = df['Date'].dt.month

# Assign season based on month
def get_season(month):
    if month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    else:
        return 'Spring'

df['Season'] = df['Month'].apply(get_season)

# Step 5: Data Analysis (analyze each term)
monthly_avg_max_temp = df.groupby('Month')['MaxTemp'].mean()
print(monthly_avg_max_temp)

# Step 6: Data Visualization (Part 2)
plt.figure(figsize=(10, 5))
plt.plot(monthly_avg_max_temp.index, monthly_avg_max_temp.values, marker='o')
plt.xlabel('Month')
plt.ylabel('Average Max Temperature')
plt.title('Monthly Average Max Temperature')
plt.grid(True)
plt.savefig("plots/monthly_avg_max_temp.png")
plt.show()

# Step 7: Advanced Analysis (e.g., predict Rainfall)
X = df[['MinTemp', 'MaxTemp']]
y = df['Rainfall']

# Drop missing values for modeling
data = pd.concat([X, y], axis=1).dropna()
X = data[['MinTemp', 'MaxTemp']]
y = data['Rainfall']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error for Rainfall Prediction: {mse}')


# Step 8: Conclusions and Insights (analyze each term)

# Calculate average rainfall per month
monthly_avg_rainfall = df.groupby('Month')['Rainfall'].mean()

# Find the month with the highest and lowest rainfall
highest_rainfall_month = monthly_avg_rainfall.idxmax()
lowest_rainfall_month = monthly_avg_rainfall.idxmin()

# Calculate average max temperature per month (if not already calculated)
monthly_avg_max_temp = df.groupby('Month')['MaxTemp'].mean()

# Find the hottest and coldest months
max_temp_month = monthly_avg_max_temp.idxmax()
min_temp_month = monthly_avg_max_temp.idxmin()

# Display the insights
print("\n--- CONCLUSIONS AND INSIGHTS ---")
print("1. Monthly Average Rainfall:")
print(monthly_avg_rainfall)

print(f"\n2. The month with the highest average rainfall is: Month {highest_rainfall_month}")
print(f"3. The month with the lowest average rainfall is: Month {lowest_rainfall_month}")

print(f"4. The hottest month (highest average max temperature): Month {max_temp_month}")
print(f"5. The coldest month (lowest average max temperature): Month {min_temp_month}")

print("6. These patterns help in understanding seasonal weather changes.")
print("   Such insights are useful for applications in agriculture, water resource planning, and tourism.")



# Step 9: Communication (Optional)
plt.figure(figsize=(8, 5))
sns.boxplot(x='Season', y='Rainfall', data=df)
plt.title('Rainfall Distribution by Season')
plt.savefig("plots/rainfall_by_season.png")
plt.show()

# Step 10: Future Work (Optional)
# Could include forecasting, seasonal modeling, etc.

# Save the results and export updated dataset
df.to_csv('weather_with_features.csv', index=False)
