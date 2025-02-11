import pandas as pd
import numpy as np

# Define custom functions
def feature_engineering(df):
    df = df.copy()
    df['date of reservation'] = pd.to_datetime(df['date of reservation'])
    df['year'] = df['date of reservation'].dt.year
    df['month'] = df['date of reservation'].dt.month
    df['day'] = df['date of reservation'].dt.day
    df['total_stay_night'] = df['number of week nights'] + df['number of weekend nights']
    df['total_visiors'] = df['number of adults'] + df['number of children']
    df['percent_canceled'] = df['P-C'] / (df['P-C'] + df['P-not-C'])
    df['percent_canceled'] = df['percent_canceled'].fillna(-1)
    return df

def encode_categorical(df):
    df = df.copy()
    meal_dict = {"Not Selected": 1, 'Meal Plan 1': 2, 'Meal Plan 2': 3, 'Meal Plan 3': 4}
    room_dict = {
        'Room_Type 1': 1, 'Room_Type 2': 2, 'Room_Type 3': 3, 'Room_Type 4': 4, 
        'Room_Type 5': 5, 'Room_Type 6': 6, 'Room_Type 7': 7
    }
    df['type of meal'] = df['type of meal'].map(meal_dict)
    df['room type'] = df['room type'].map(room_dict)
    market_segment_categories = ["Aviation", "Complementary", "Corporate", "Offline", "Online"]
    for category in market_segment_categories:
        column_name = f"market_segment_{category}"
        df[column_name] = (df["market segment type"] == category).astype(int)
    return df

def Drop_unnecessary(df):
    df = df.copy()
    df = df.drop(columns=["Booking_ID", "date of reservation", "market segment type"])
    return df

def log_transform(df):
    df = df.copy()
    df[:, [8, 12]] = np.log1p(df[:, [8, 12]])
    return df
