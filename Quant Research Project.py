import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

parquet_file_path = 'E:/True Beacon Quant Research Assigntment/data.parquet'
df = pd.read_parquet(parquet_file_path)
df.reset_index(inplace=True)
print(df.head())
data = df.copy()
report = pd.DataFrame(columns=['Model', 'Sharpe Ratio', 'Total PnL', 'Max Drawdown', 'Length'])
min_tf = 30 # 30 minutes
max_tf = 1880 #5 Days
train_size = 0.7
factor = 0.95

def data_pre_process(data):
    data = data.copy()

    print("Initial = ", len(data))

    data.rename(columns={'time': 'datetime'}, inplace=True)
    data['time'] = data['datetime'].astype(str)
    data['time'] = pd.to_datetime(data['time'])
    data['time'] = data['time'].dt.time
    start_time = pd.to_datetime('09:15:00').time()
    end_time = pd.to_datetime('15:30:00').time()
    data = data[(data['time'] >= start_time) & (data['time'] <= end_time)]

    data['date'] = data['datetime'].astype(str)
    data['date'] = pd.to_datetime(data['date'])
    data['date'] = data['date'].dt.date
    data['day'] = pd.to_datetime(data['date']).dt.day_name()

    print("Minutes after filtering trade time = ", len(data), (len(data) / 376))
    data = data[(data['day'] != 'Saturday') & (data['day'] != 'Sunday')]
    print("Minutes after filtering trade time = ", len(data), (len(data) / 376))

    data = data.ffill()

    data['Spread'] = data['banknifty'] - data['nifty']

    # negative_spread_rows = data[data['Spread'] < 0]
    # print(negative_spread_rows)
    # negative_spread_rows.to_excel("E:/True Beacon Quant Research Assigntment/negative.xlsx", index=False)

    data['consecutive_counts'] = (data['Spread'] == data['Spread'].shift()).astype(int)
    counter = 0
    i = 0
    while i < len(data) - 375:
        if data['consecutive_counts'].iloc[i] == 1 and all(data['consecutive_counts'].iloc[i:i + 376] == 1):
            if str(data['time'].iloc[i]) == '09:15:00' and str(data['time'].iloc[i + 375]) == '15:30:00':
                counter += 1
                print(
                    f"{counter}. Consecutive 1's found from row {i + 2} to {i + 2 + 375} {data['time'].iloc[i]} {data['time'].iloc[i + 375]} {data['date'].iloc[i]} {data['date'].iloc[i + 375]}")
                data = data.drop(data.index[i:i + 375 + 1])
                i += 376
            else:
                i += 1
        else:
            i += 1

    print("removing non-trade days = ", len(data))

    tte_check = data.groupby('date')['tte'].nunique()
    rows_with_different_tte = tte_check[tte_check > 1]
    if not rows_with_different_tte.empty:
        print("There are rows with different 'tte' values for the same date:")
        print(rows_with_different_tte)
    else:
        print("All 'tte' values are the same for rows with the same date.")

    data['first_tte'] = data.groupby('date')['tte'].transform('first')
    data['tte'] = data['first_tte']

    data = data.drop(columns=['first_tte'])
    data.reset_index(inplace=True)
    print("Initial = ", len(data))
    return data

def base_model(data, window_size, train_size):

    rolling_mean = data['Spread'].rolling(window=window_size).mean()
    rolling_std = data['Spread'].rolling(window=window_size).std()

    data['RMean'] = rolling_mean
    data['RStd'] = rolling_std
    data['ZScore'] = zscore(data['Spread'])

    start_index = int(len(data) * train_size)

    data.loc[:29, 'ZScore'] = 0

    data['Long_Signal'] = np.where((data['ZScore'] > data['RMean']), 1, 0)
    data['Short_Signal'] = np.where((data['ZScore'] < -data['RMean']), 1, 0)

    data.loc[:start_index, 'Long_Signal'] = 0
    data.loc[:start_index, 'Short_Signal'] = 0

    data = timeframe(data)
    data = evaluation("Base Model", data)
    print(data['ZScore'].describe())

def linear_regression(data, window_size, train_size):

    data['Spread_Lagged_1'] = data['Spread'].shift(1)
    data['Spread_Lagged_2'] = data['Spread'].shift(2)

    data[['Spread_Lagged_1', 'Spread_Lagged_2']] = data[['Spread_Lagged_1', 'Spread_Lagged_2']].fillna(0)

    data['Future_Spread'] = data['Spread'].shift(-1)
    data[['Future_Spread']] = data[['Future_Spread']].fillna(0)

    X = data[['Spread_Lagged_1', 'Spread_Lagged_2']]
    y = data['Future_Spread']

    split_index = int(train_size * len(data)) - window_size
    X_train, X_test, y_train, y_test = X[:split_index], X[split_index:], y[:split_index], y[split_index:]

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    data['Predicted_Spread'] = 0
    data.loc[X_test.index, 'Predicted_Spread'] = lr_model.predict(X_test)

    data['Dynamic_Threshold'] = data['Predicted_Spread'].rolling(window=window_size).mean().shift(-1)
    threshold = data['Dynamic_Threshold'] * factor

    data['Long_Signal'] = np.where(data['Predicted_Spread'] > threshold, 1, 0)
    data['Short_Signal'] = np.where(data['Predicted_Spread'] < -threshold, 1, 0)

    data = timeframe(data)
    data = evaluation("Linear Regression", data)
    print(data.loc[X_test.index, 'Predicted_Spread'].describe())
    return data

def gradient_boosting(data, window_size, train_size):
    # Create lagged features
    data['Spread_Lagged_1'] = data['Spread'].shift(1)
    data['Spread_Lagged_2'] = data['Spread'].shift(2)

    # Fill NaN values in lagged features with 0
    data[['Spread_Lagged_1', 'Spread_Lagged_2']] = data[['Spread_Lagged_1', 'Spread_Lagged_2']].fillna(0)

    # Create the target variable (y): The future spread
    data['Future_Spread'] = data['Spread'].shift(-1)
    data[['Future_Spread']] = data[['Future_Spread']].fillna(0)

    # Prepare features (X) and target variable (y)
    X = data[['Spread_Lagged_1', 'Spread_Lagged_2']]
    y = data['Future_Spread']

    # Split data into training and testing sets
    split_index = int(train_size * len(data)) - window_size
    X_train, X_test, y_train, y_test = X[:split_index], X[split_index:], y[:split_index], y[split_index:]

    # Define parameter grid for Gradient Boosting
    param_grid_gb = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5]
    }

    # Initialize GridSearchCV for Gradient Boosting
    gb_model = GradientBoostingRegressor(random_state=42)
    grid_search_gb = GridSearchCV(gb_model, param_grid_gb, cv=3)
    grid_search_gb.fit(X_train, y_train)

    # Get the best model from the grid search
    best_gb_model = grid_search_gb.best_estimator_

    # Make predictions on the test set
    data['Predicted_Spread'] = 0
    data.loc[X_test.index, 'Predicted_Spread'] = best_gb_model.predict(X_test)

    # Set dynamic threshold based on the rolling standard deviation of predicted spread
    data['Dynamic_Threshold'] = data['Predicted_Spread'].rolling(window=window_size).mean().shift(-1)
    data['Uncertainty_Threshold'] = data['Dynamic_Threshold'] * factor

    # Generate trading signals based on predicted spread in the test set and dynamic threshold
    data['Long_Signal'] = np.where(data['Predicted_Spread'] > data['Uncertainty_Threshold'], 1, 0)
    data['Short_Signal'] = np.where(data['Predicted_Spread'] < -data['Uncertainty_Threshold'], 1, 0)

    data = timeframe(data)
    data = evaluation("Gradient Boosting", data)
    print(data.loc[X_test.index, 'Predicted_Spread'].describe())

    return data

def random_forest(data, window_size, train_size):
    # Create lagged features
    data['Spread_Lagged_1'] = data['Spread'].shift(1)
    data['Spread_Lagged_2'] = data['Spread'].shift(2)

    # Fill NaN values in lagged features with 0
    data[['Spread_Lagged_1', 'Spread_Lagged_2']] = data[['Spread_Lagged_1', 'Spread_Lagged_2']].fillna(0)

    # Create the target variable (y): The future spread
    data['Future_Spread'] = data['Spread'].shift(-1)
    data[['Future_Spread']] = data[['Future_Spread']].fillna(0)

    # Prepare features (X) and target variable (y)
    X = data[['Spread_Lagged_1', 'Spread_Lagged_2']]
    y = data['Future_Spread']

    # Split data into training and testing sets
    split_index = int(train_size * len(data)) - window_size
    X_train, X_test, y_train, y_test = X[:split_index], X[split_index:], y[:split_index], y[split_index:]

    # Initialize the random forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model on the training set
    rf_model.fit(X_train, y_train)

    # Make predictions on the test set
    data['Predicted_Spread'] = 0
    data.loc[X_test.index, 'Predicted_Spread'] = rf_model.predict(X_test)

    data['Dynamic_Threshold'] = data['Predicted_Spread'].rolling(window=window_size).mean().shift(-1)

    # Set threshold values for trading signals using the multiplier
    threshold = data['Dynamic_Threshold'] * factor

    # Generate trading signals based on predicted spread in the test set
    data['Long_Signal'] = np.where(data['Predicted_Spread'] > threshold, 1, 0)
    data['Short_Signal'] = np.where(data['Predicted_Spread'] < -threshold, 1, 0)

    data = timeframe(data)
    data = evaluation("Random Forest", data)

    # Display statistics of the predicted spread in the test set
    print(data.loc[X_test.index, 'Predicted_Spread'].describe())

    return data

def support_vector_machine(data, window_size, train_size):

    data['Spread_Lagged_1'] = data['Spread'].shift(1)
    data['Spread_Lagged_2'] = data['Spread'].shift(2)
    data[['Spread_Lagged_1', 'Spread_Lagged_2']] = data[['Spread_Lagged_1', 'Spread_Lagged_2']].fillna(0)

    data['Future_Spread'] = data['Spread'].shift(-1)
    data[['Future_Spread']] = data[['Future_Spread']].fillna(0)

    X = data[['Spread_Lagged_1', 'Spread_Lagged_2']]
    y = data['Future_Spread']

    split_index = int(train_size * len(data)) - window_size
    X_train, X_test, y_train, y_test = X[:split_index], X[split_index:], y[:split_index], y[split_index:]

    svm_model = SVR()
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)

    predictions_df = pd.DataFrame({'Predicted_Spread_SVM': svm_predictions}, index=X_test.index)

    data = data.join(predictions_df, how='left')

    data['Dynamic_Threshold_SVM'] = data['Predicted_Spread_SVM'].rolling(window=window_size).mean().shift(-1)
    data['Uncertainty_Threshold_SVM'] = data['Dynamic_Threshold_SVM'] * factor

    data['Long_Signal'] = np.where(data['Predicted_Spread_SVM'] > data['Uncertainty_Threshold_SVM'], 1, 0)
    data['Short_Signal'] = np.where(data['Predicted_Spread_SVM'] < -data['Uncertainty_Threshold_SVM'], 1, 0)

    data = timeframe(data)
    data = evaluation("SVM", data)
    print(data.loc[X_test.index, 'Predicted_Spread_SVM'].describe())
    return data

def timeframe(data):
    global min_tf, max_tf

    data['Position_Hold_Time'] = 0
    data['Signal Type'] = ''
    data['Posi'] = 0
    data['Position'] = 0
    position_hold_time = 0
    current_signal_type = None
    position = None

    for i in range(len(data)):
        if data['Long_Signal'].iloc[i] == 1:
            current_signal_type = 'long'
        elif data['Short_Signal'].iloc[i] == 1:
            current_signal_type = 'short'
        else:
            current_signal_type = None

        if position_hold_time > max_tf:
            position_hold_time = 0
            current_signal_type = None
            position = None

        if position_hold_time <= min_tf:
            if position_hold_time == 0 and position is None:
                if data["Long_Signal"].iloc[i] == 1:
                    position_hold_time = 0
                    position = 'long'
                elif data["Short_Signal"].iloc[i] == 1:
                    position_hold_time = 0
                    position = 'short'
            else:
                position_hold_time += 1
        else:
            if current_signal_type == 'long' and position == 'short':
                position_hold_time = 1
                position = 'long'
            elif current_signal_type == 'short' and position == 'long':
                position_hold_time = 1
                position = 'short'
            elif current_signal_type == position:
                position_hold_time += 1
            else:
                position_hold_time += 1

        data.loc[data.index[i], 'Position_Hold_Time'] = position_hold_time
        data.loc[data.index[i], 'Signal Type'] = current_signal_type
        data.loc[data.index[i], 'Position'] = position

        if data['Position'].iloc[i] == 'long' or data['Position'].iloc[i] == 'short':
            data.loc[data.index[i], 'Posi'] = int(1)

    data['PnL'] = data['Spread'] * (data['tte'] ** 0.7) * data['Posi']

    data['PnL'] = data['PnL'].fillna(0)
    data['Position'] = data['Position'].shift(1)
    data['Position_Hold_Time'] = data['Position_Hold_Time'].shift(1)
    data['Cumulative_PnL'] = data['PnL'].cumsum()

    return data

def evaluation(text, data):
    global report

    sharpe = data['PnL'].mean() / data['PnL'].std()
    total_pnl = data['PnL'].sum()
    max_dd = (data['Cumulative_PnL'].cummax() - data['Cumulative_PnL']).max()
    length = len(data)

    new_row = {'Model': text,
               'Sharpe Ratio': sharpe,
               'Total PnL': total_pnl,
               'Max Drawdown': max_dd,
               'Length': length}

    report = pd.concat([report, pd.DataFrame([new_row])], ignore_index=True)
    # excel_file_path = f'E:/True Beacon Quant Research Assigntment/models/{text}.xlsx'
    # data.to_excel(excel_file_path, index=False)

    return data

data = data_pre_process(data)
base_model(data, min_tf, 0)
base_model(data, min_tf, train_size)
linear_regression(data, min_tf, train_size)
gradient_boosting(data, min_tf, train_size)
random_forest(data, min_tf, train_size)
support_vector_machine(data, min_tf, train_size)

print(report)
print("Factor = ", factor)