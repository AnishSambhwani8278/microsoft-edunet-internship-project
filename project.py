import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

data = pd.read_csv(r"C:\Users\User\Desktop\weather.csv")
data.columns = ['date', 'temperature']
data['date'] = pd.to_datetime(data['date'])

data['day'] = data['date'].dt.day
data['month'] = data['date'].dt.month
data['dayofweek'] = data['date'].dt.dayofweek
data['is_weekend'] = data['dayofweek'].isin([5, 6]).astype(int)
data['prev_temp'] = data['temperature'].shift(1)
data['temp_diff'] = data['temperature'] - data['prev_temp']
data['rolling_avg_3'] = data['temperature'].rolling(window=3).mean()
data['rolling_avg_7'] = data['temperature'].rolling(window=7).mean()
data['rolling_std_3'] = data['temperature'].rolling(window=3).std()
data['rolling_std_7'] = data['temperature'].rolling(window=7).std()
data['target_temp'] = data['temperature'].shift(-1)

def get_season(month):
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    else:
        return 3  # Fall

data['season'] = data['month'].apply(get_season)
data = data.dropna()

features = ['temperature', 'prev_temp', 'temp_diff', 'rolling_avg_3', 'rolling_avg_7',
            'rolling_std_3', 'rolling_std_7', 'day', 'month', 'dayofweek', 'is_weekend', 'season']
X = data[features]
y = data['target_temp']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

models = {
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, max_depth=5, random_state=42),
    "Linear Regression": LinearRegression()
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append((name, mse, mae, r2))

    plt.figure(figsize=(10, 4))
    plt.plot(y_test.values[:50], label='Actual')
    plt.plot(y_pred[:50], label='Predicted')
    plt.title(f'{name} - Temperature Prediction')
    plt.xlabel('Samples')
    plt.ylabel('Temperature')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot.png")
    plt.show()

accuracy_df = pd.DataFrame(results, columns=["Model", "MSE", "MAE", "R² Score"])
print("Model Performance Comparison:\n")
print(accuracy_df.round(3))

plt.figure(figsize=(10, 6))
corr = data[features + ['target_temp']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(data['target_temp'], bins=30, kde=True)
plt.title("Target Temperature Distribution")
plt.xlabel("Temperature")
plt.tight_layout()
plt.savefig("temperature_distribution.png")
plt.show()