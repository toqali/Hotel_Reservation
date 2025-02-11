from flask import Flask, render_template, request
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from pre_steps import *


# Load the model and scaler
model = joblib.load('rf_model_hotel_reservation.pkl')  
scaler = joblib.load('scaler.pkl')  

# Define the preprocessing pipeline
preprocessor = Pipeline([
    ("feature_engineering", FunctionTransformer(feature_engineering)),
    ("encoding", FunctionTransformer(encode_categorical)),
    ("Drop_unnecessary", FunctionTransformer(Drop_unnecessary)),
    ("scaling", scaler),
    ("log_transform", FunctionTransformer(log_transform))
])



app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def booking_form():
    ordered_cols = ['Booking_ID', 'number of adults', 'number of children',
       'number of weekend nights', 'number of week nights', 'type of meal',
       'car parking space', 'room type', 'lead time', 'market segment type',
       'repeated', 'P-C', 'P-not-C', 'average price', 'special requests',
       'date of reservation']
    fields = [
        ("Booking_ID", "text", "Enter booking ID"),
        ("number of adults", "number", "Enter number of adults"),
        ("number of children", "number", "Enter number of children"),
        ("number of weekend nights", "number", "Enter number of weekend nights"),
        ("number of week nights", "number", "Enter number of week nights"),
        ("car parking space", "number", "0: No car, 1: Has car"),
        ("lead time", "number", "Enter lead time in days"),
        ("repeated", "number", "Enter 1 if repeated guest, 0 otherwise"),
        ("P-C", "number", "Enter P-C value"),
        ("P-not-C", "number", "Enter P-not-C value"),
        ("average price", "number", "Enter average price"),
        ("special requests", "number", "Enter number of special requests"),
        ("date of reservation", "date", "Select reservation date"),
    ]

    field_datalist = [
        ("room type", ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7']),
        ("type of meal", ["Not Selected", 'Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3']),
        ("market segment type", ["Online", "Offline", "Corporate", "Complementary", "Aviation"]),
    ]
    is_canceled = None
    if request.method == "POST":
        form_data = {}
        for field, field_type, _ in fields:
            key = field
            val = request.form.get(key)  

            if val is not None:  
                val = val.strip()  
                if field_type == "number":
                    if val.isdigit():  
                        val = int(val)  

            form_data[key] = val  

        # Handle field_datalist data
        for field in field_datalist:
            key = field[0]
            form_data[key] = request.form.get(key)
        # Convert form data to DataFrame
        input_data = pd.DataFrame([form_data])
        print('==========================================================')
        print(input_data.columns)
        # Transform data using the pipeline
        input_data = input_data[ordered_cols]
        transformed_data = preprocessor.transform(input_data)
        is_canceled = "Not_Canceled" if model.predict(transformed_data )==1 else "Canceled"
    return render_template("ui.html", fields=fields, field_datalist=field_datalist, prediction_result = is_canceled)

if __name__ == "__main__":
        app.run(debug=True)
            
