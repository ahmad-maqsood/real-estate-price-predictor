import numpy as np
import joblib
import pandas as pd

model = joblib.load("model/random_forest_model.pkl")
ohe = joblib.load("model/ohe_encoder.pkl")
# print(ohe.categories_)

in_area = float(input("Total Area in Marla : "))
in_bedrooms = float(input("Total Bedrooms : "))
in_baths = float(input("Total Baths : "))

while(True):
    in_type = input("Property Type (House, Flat, Shop, etc.) : ")
    in_city = input("City : ")

    valid_types = ohe.categories_[0].tolist()
    valid_cities = ohe.categories_[1].tolist()

    if in_type not in valid_types:
        print(f"Invalid type. Choose from: {valid_types}")
    elif in_city not in valid_cities:
        print(f"Invalid city. Choose from: {valid_cities}")
    else:
        break

input_data = pd.DataFrame([{
    "area_in_marla": in_area,
    "bedroom": in_bedrooms,
    "bath": in_baths
}])

cat_input = pd.DataFrame([{
    "type": in_type,
    "location_city": in_city
}])

encoded_cat_input = ohe.transform(cat_input)
encoded_input_df = pd.DataFrame(encoded_cat_input, columns=ohe.get_feature_names_out(["type", "location_city"]))

final_input = pd.concat([input_data, encoded_input_df], axis=1)

log_price = model.predict(final_input)
actual_price = np.expm1(log_price)

print(f"Predicted Price: PKR {actual_price[0]:,.2f}")