# Weather Data Analysis Project

This is a comprehensive beginner-friendly data analysis project in Python using a weather dataset. It covers steps from data exploration to visualization, feature engineering, linear regression, and concluding trends in rainfall and temperature.

---

## 📁 Project Structure

```
weather_data_analysis/
├── weather.csv
├── weather_with_features.csv
├── weather_data_analysis.py
└── plots/
    ├── pairplot_temp_rainfall.png
    ├── monthly_avg_max_temp.png
    └── rainfall_by_season.png
```

---

## ✅ Step-by-Step Breakdown

### 🔹 Step 1: Load the Data

```python
df = pd.read_csv('weather.csv')
```
Loads the weather dataset from a CSV file into a pandas DataFrame.

---

### 🔹 Step 2: Data Exploration

```python
print(df.head())
print(df.info())
print(df.describe())
```
Displays the first few rows, data types, and statistical summary of the dataset.

---

### 🔹 Step 3: Data Visualization

```python
sns.pairplot(df[['MinTemp', 'MaxTemp', 'Rainfall']])
plt.savefig("plots/pairplot_temp_rainfall.png")
plt.show()
```
Visualizes relationships between minimum temp, maximum temp, and rainfall using a pair plot.

---

### 🔹 Step 4: Feature Engineering

```python
if 'Date' not in df.columns:
    df['Date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Month'] = df['Date'].dt.month
```

Ensures a 'Date' column exists, then extracts the month for seasonal grouping.

```python
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
```
Defines and applies a function to categorize each month into a season.

---

### 🔹 Step 5: Data Analysis

```python
monthly_avg_max_temp = df.groupby('Month')['MaxTemp'].mean()
print(monthly_avg_max_temp)
```
Calculates and prints the average maximum temperature for each month.

---

### 🔹 Step 6: Monthly Max Temperature Plot

```python
plt.figure(figsize=(10, 5))
plt.plot(monthly_avg_max_temp.index, monthly_avg_max_temp.values, marker='o')
plt.xlabel('Month')
plt.ylabel('Average Max Temperature')
plt.title('Monthly Average Max Temperature')
plt.grid(True)
plt.savefig("plots/monthly_avg_max_temp.png")
plt.show()
```
Plots monthly average max temperatures and saves the image.

---

### 🔹 Step 7: Rainfall Prediction using Linear Regression

```python
X = df[['MinTemp', 'MaxTemp']]
y = df['Rainfall']
data = pd.concat([X, y], axis=1).dropna()
X = data[['MinTemp', 'MaxTemp']]
y = data['Rainfall']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error for Rainfall Prediction: {mse}')
```
Builds a linear regression model to predict rainfall based on min and max temperature.

---

### 🔹 Step 8: Conclusions and Insights

```python
monthly_avg_rainfall = df.groupby('Month')['Rainfall'].mean()
highest_rainfall_month = monthly_avg_rainfall.idxmax()
lowest_rainfall_month = monthly_avg_rainfall.idxmin()
max_temp_month = monthly_avg_max_temp.idxmax()
min_temp_month = monthly_avg_max_temp.idxmin()
```

Performs key statistical analysis and prints insights like hottest/coldest and wettest/driest months.

```python
print("1. Monthly Average Rainfall:")
print(monthly_avg_rainfall)
print(f"2. Highest average rainfall month: {highest_rainfall_month}")
print(f"3. Lowest average rainfall month: {lowest_rainfall_month}")
print(f"4. Hottest month: {max_temp_month}")
print(f"5. Coldest month: {min_temp_month}")
```

---

### 🔹 Step 9: Seasonal Rainfall Visualization

```python
plt.figure(figsize=(8, 5))
sns.boxplot(x='Season', y='Rainfall', data=df)
plt.title('Rainfall Distribution by Season')
plt.savefig("plots/rainfall_by_season.png")
plt.show()
```
Uses a boxplot to show how rainfall varies across different seasons.

---

### 🔹 Step 10: Future Work

Ideas to expand the project:
- Time series modeling of temperature or rainfall
- Weather forecasting using ARIMA or LSTM
- Integrate external APIs for real-time updates

---

## 📦 Output Files

- `weather_with_features.csv`: Cleaned and processed dataset
- PNGs in `plots/`: Saved charts and graphs

---

## 🧠 Skills Learned

- Pandas and data manipulation
- Exploratory Data Analysis (EDA)
- Seaborn and Matplotlib visualizations
- Linear Regression modeling
- Insight extraction from real-world data

---
## ▶️ How to Run the Project

```bash
pip install pandas matplotlib seaborn scikit-learn
python weather_data_analysis.py
```

---

## 🙋 Author

**SRIJAN PAUL**  
This project was completed as part of an internship and academic submission.

## 📜 License

MIT License
