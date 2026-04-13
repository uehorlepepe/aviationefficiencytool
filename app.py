from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import os
app = Flask(__name__)
CORS(app)
airports_df = pd.read_csv("globalairportsdata.csv")
airports_df = airports_df[airports_df["IATA"].notna()].drop_duplicates(subset="IATA", keep="first")
airport_coords = airports_df.set_index("IATA")[["GeoPointLat", "GeoPointLong"]].to_dict("index")
def calculate_distance(origin_iata, dest_iata):
    try:
        origin = airport_coords[origin_iata]
        dest = airport_coords[dest_iata]
    except KeyError:
        raise ValueError(f"Airport code {origin_iata} or {dest_iata} not found in database.")
    lat1, lon1 = radians(origin["GeoPointLat"]), radians(origin["GeoPointLong"])
    lat2, lon2 = radians(dest["GeoPointLat"]), radians(dest["GeoPointLong"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return 6371 * c
aircraft_ranges = {
    "Boeing 787-9": 13950, "Airbus A330-200": 13200, "Airbus A350-900": 15000,
    "Boeing 777-300ER": 13450, "Boeing 787-8": 13100, "Airbus A380-800": 15400,
    "Airbus A330-900": 12800, "Boeing 777-200ER": 12400, "Airbus A330-300": 11300,
    "Boeing 787-10": 11200, "Boeing 767-300ER": 10400, "Airbus A321neo": 6800,
    "Airbus A319ceo": 5900, "Boeing 737 MAX 9": 6100, "Boeing 737 MAX 8": 6100,
    "Airbus A320neo": 5800, "Boeing 757-200": 5800, "Airbus A220-300": 5700,
    "Airbus A320ceo": 5400, "Airbus A321ceo": 4900, "Boeing 737-800": 5000,
    "Boeing 737-900ER": 5100, "Boeing 737-700": 4600, "Embraer E190": 114,
    "Embraer E175": 3300, "Bombardier CRJ-900": 2400, "Embraer ERJ-145": 2100,
    "Airbus A319neo": 6400, "Boeing 737 MAX 7": 6500, "Boeing 737 MAX 10": 5400
}

aircraft_max_passengers = {
    "Boeing 787-9": 290, "Airbus A330-200": 246, "Airbus A350-900": 325,
    "Boeing 777-300ER": 396, "Boeing 787-8": 242, "Airbus A380-800": 555,
    "Airbus A330-900": 287, "Boeing 777-200ER": 314, "Airbus A330-300": 277,
    "Boeing 787-10": 330, "Boeing 767-300ER": 218, "Airbus A321neo": 244,
    "Airbus A319ceo": 156, "Boeing 737 MAX 9": 220, "Boeing 737 MAX 8": 210,
    "Airbus A320neo": 195, "Boeing 757-200": 200, "Airbus A220-300": 160,
    "Airbus A320ceo": 180, "Airbus A321ceo": 185, "Boeing 737-800": 189,
    "Boeing 737-900ER": 215, "Boeing 737-700": 149, "Embraer E190": 114,
    "Embraer E175": 88, "Bombardier CRJ-900": 90, "Embraer ERJ-145": 50,
    "Airbus A319neo": 156, "Boeing 737 MAX 7": 172, "Boeing 737 MAX 10": 230
}
df = pd.read_csv("yifannewmodelling.csv")

le_aircraft = LabelEncoder()
df["Aircraft_Code"] = le_aircraft.fit_transform(df["Aircraft_Type"])
le_origin = LabelEncoder()
df["Origin_Code"] = le_origin.fit_transform(df["Origin"])
le_dest = LabelEncoder()
df["Destination_Code"] = le_dest.fit_transform(df["Destination"])

X = df[["Distance_km", "Passengers", "Cargo_Load_pct", "Wind_kts", "Cruise_Temperature_C",
        "Aircraft_Code", "Origin_Code", "Destination_Code"]]
y = df["Fuel_Burn_L"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
model.fit(X_train, y_train)

train_efficiency = (X_train["Passengers"] * X_train["Distance_km"]) / y_train
typical_efficiency = train_efficiency.mean()
threshold_efficiency = 0.9 * typical_efficiency
def check_flight_optimization(flight_input, range_buffer=1.2):
    fi = flight_input.copy()
    origin, destination = fi["Origin"], fi["Destination"]
    ac_type, passengers = fi["Aircraft_Type"], fi["Passengers"]

    try:
        fi["Distance_km"] = calculate_distance(origin, destination)
    except ValueError as e:
        return f"Error: {str(e)}"

    ac_range = aircraft_ranges.get(ac_type, 0)
    max_pass = aircraft_max_passengers.get(ac_type, 1000)
    if fi["Distance_km"] > ac_range:
        return f"Warning: {ac_type} cannot fly {fi['Distance_km']:.0f} km."
    if passengers > max_pass:
        return f"Warning: {ac_type} cannot carry {passengers} passengers (max {max_pass})."

    fi["Aircraft_Code"] = le_aircraft.transform([ac_type])[0] if ac_type in le_aircraft.classes_ else -1
    fi["Origin_Code"] = le_origin.transform([origin])[0] if origin in le_origin.classes_ else -1
    fi["Destination_Code"] = le_dest.transform([destination])[0] if destination in le_dest.classes_ else -1

    df_temp = pd.DataFrame([fi])[X_train.columns].fillna(-1)
    predicted_fuel = model.predict(df_temp)[0]
    efficiency = (passengers * fi["Distance_km"]) / predicted_fuel

    recommendations = []

    if efficiency < threshold_efficiency:
        scale = efficiency / threshold_efficiency
        if fi.get("Cargo_Load_pct", 0) > 0:
            cargo_new = fi["Cargo_Load_pct"] * scale
            recommendations.append(f"Reduce cargo by ~{fi['Cargo_Load_pct'] - cargo_new:.1f}% (to {cargo_new:.1f}%)")
        if passengers > 0:
            pass_new = int(passengers * scale)
            recommendations.append(f"Reduce passengers by ~{passengers - pass_new} (to {pass_new})")

    alt_recs = []
    for ac in le_aircraft.classes_:
        if ac == ac_type:
            continue
        ac_range_alt = aircraft_ranges.get(ac, 0)
        ac_pass_alt = aircraft_max_passengers.get(ac, 0)
        if ac_range_alt < fi["Distance_km"] * range_buffer or ac_pass_alt < passengers:
            continue
        temp_fi = fi.copy()
        temp_fi["Aircraft_Code"] = le_aircraft.transform([ac])[0]
        temp_df = pd.DataFrame([temp_fi])[X_train.columns].fillna(-1)
        temp_fuel = model.predict(temp_df)[0]
        temp_efficiency = (passengers * fi["Distance_km"]) / temp_fuel
        if temp_efficiency > efficiency:
            alt_recs.append((ac, temp_efficiency))
    alt_recs = sorted(alt_recs, key=lambda x: x[1], reverse=True)[:3]
    for ac, eff in alt_recs:
        recommendations.append(f"Try {ac} for higher efficiency ({eff:.2f} pkm/L)")

    if not recommendations:
        recommendations.append("Passengers and cargo load are already optimal, and aircraft is efficient.")

    summary = f"Predicted Fuel Burn: {predicted_fuel:.2f} L\nEfficiency: {efficiency:.2f} pkm/L\n"
    return summary + "Recommendations:\n" + "\n".join(recommendations)
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    flight_input = {
        "Origin": data["origin"].upper(),
        "Destination": data["destination"].upper(),
        "Aircraft_Type": data["aircraft"],
        "Passengers": int(data["passengers"]),
        "Cargo_Load_pct": float(data["cargo"]),
        "Wind_kts": float(data["wind"]),
        "Cruise_Temperature_C": float(data["cruise_temp"])
    }
    
    result = check_flight_optimization(flight_input)
    lines = result.split("\n")
    fuel_line = lines[0] if len(lines) > 0 else ""
    efficiency_line = lines[1] if len(lines) > 1 else ""
    recommendations = [l for l in lines[3:] if l.strip()]

    return jsonify({
    "fuel": fuel_line,
    "efficiency": efficiency_line,
    "recommendations": recommendations
})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
