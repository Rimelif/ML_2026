import numpy as np
import pandas as pd


POWER_RAW_PATH = "data/household_power_consumption.txt"
WEATHER_RAW_PATH = "data/MENSQ_92_previous-1950-2024.csv"

ALL_DAILY_PATH = "data/daily_with_nearest_weather.csv"
TRAIN_DAILY_PATH = "data/train_daily.csv"
TEST_DAILY_PATH = "data/test_daily.csv"
WEATHER_MONTHLY_PATH = "data/weather_monthly_nearest_station.csv"
WEATHER_SOURCE_PATH = "data/weather_monthly_station_sources.csv"

SCEAUX_LAT = 48.7786
SCEAUX_LON = 2.2906

WEATHER_COLS = ["RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]


def write_csv_with_fallback(df, path, **kwargs):
    try:
        df.to_csv(path, **kwargs)
        return path
    except PermissionError:
        fallback = path.replace(".csv", "_new.csv")
        df.to_csv(fallback, **kwargs)
        print(f"Could not overwrite {path}; saved fallback file {fallback}")
        return fallback


def haversine_km(lat, lon, ref_lat=SCEAUX_LAT, ref_lon=SCEAUX_LON):
    lat1 = np.radians(ref_lat)
    lon1 = np.radians(ref_lon)
    lat2 = np.radians(lat.astype(float))
    lon2 = np.radians(lon.astype(float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def read_power_data(path=POWER_RAW_PATH):
    data = pd.read_csv(
        path,
        sep=";",
        na_values="?",
        low_memory=False,
    )
    data["DateTime"] = pd.to_datetime(
        data["Date"] + " " + data["Time"],
        format="%d/%m/%Y %H:%M:%S",
    )
    data = data.drop(columns=["Date", "Time"]).set_index("DateTime").sort_index()

    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data.ffill(inplace=True)
    return data


def aggregate_power_daily(power):
    power = power.copy()
    power["sub_metering_remainder"] = (
        power["Global_active_power"] * 1000 / 60
        - (
            power["Sub_metering_1"]
            + power["Sub_metering_2"]
            + power["Sub_metering_3"]
        )
    )

    agg_rules = {
        "Global_active_power": "sum",
        "Global_reactive_power": "sum",
        "Voltage": "mean",
        "Global_intensity": "mean",
        "Sub_metering_1": "sum",
        "Sub_metering_2": "sum",
        "Sub_metering_3": "sum",
        "sub_metering_remainder": "sum",
    }
    daily = power.resample("D").agg(agg_rules)
    daily.index.name = "DateTime"
    return daily


def read_weather_data(path=WEATHER_RAW_PATH):
    cols = ["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "AAAAMM"] + WEATHER_COLS
    weather = pd.read_csv(path, sep=";", usecols=cols)

    numeric_cols = ["NUM_POSTE", "LAT", "LON", "AAAAMM"] + WEATHER_COLS
    for col in numeric_cols:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")

    weather = weather.dropna(subset=["NUM_POSTE", "LAT", "LON", "AAAAMM"])
    weather["NUM_POSTE"] = weather["NUM_POSTE"].astype("int64")
    weather["AAAAMM"] = weather["AAAAMM"].astype("int64")
    weather["distance_to_sceaux_km"] = haversine_km(weather["LAT"], weather["LON"])
    return weather


def build_monthly_weather(weather, months):
    monthly_rows = []
    source_rows = []

    for month in sorted(months):
        month_weather = weather[weather["AAAAMM"] == month].sort_values("distance_to_sceaux_km")
        if month_weather.empty:
            raise ValueError(f"No weather station data found for month {month}.")

        monthly_row = {"AAAAMM": month}
        for col in WEATHER_COLS:
            valid = month_weather[month_weather[col].notna()]
            if valid.empty:
                monthly_row[col] = 0.0
                source_rows.append(
                    {
                        "AAAAMM": month,
                        "weather_feature": col,
                        "value": 0.0,
                        "NUM_POSTE": pd.NA,
                        "NOM_USUEL": "filled_zero_no_station_value",
                        "LAT": pd.NA,
                        "LON": pd.NA,
                        "distance_to_sceaux_km": pd.NA,
                    }
                )
                continue

            selected = valid.iloc[0]
            monthly_row[col] = selected[col]
            source_rows.append(
                {
                    "AAAAMM": month,
                    "weather_feature": col,
                    "value": selected[col],
                    "NUM_POSTE": int(selected["NUM_POSTE"]),
                    "NOM_USUEL": selected["NOM_USUEL"],
                    "LAT": selected["LAT"],
                    "LON": selected["LON"],
                    "distance_to_sceaux_km": selected["distance_to_sceaux_km"],
                }
            )

        monthly_rows.append(monthly_row)

    monthly = pd.DataFrame(monthly_rows)
    sources = pd.DataFrame(source_rows)
    return monthly, sources


def attach_weather(daily, monthly_weather):
    daily = daily.copy()
    daily["AAAAMM"] = daily.index.year * 100 + daily.index.month
    daily = daily.reset_index().merge(monthly_weather, on="AAAAMM", how="left")

    if daily[WEATHER_COLS].isna().any().any():
        missing_months = daily.loc[daily[WEATHER_COLS].isna().any(axis=1), "AAAAMM"].unique()
        raise ValueError(f"Weather merge produced missing values for months: {missing_months}")

    daily = daily.drop(columns=["AAAAMM"]).set_index("DateTime")
    return daily


def split_train_test(daily):
    train = daily.loc[: "2008-12-31"]
    test = daily.loc["2009-01-01":]
    return train, test


def main():
    power = read_power_data()
    daily_power = aggregate_power_daily(power)

    months = (daily_power.index.year * 100 + daily_power.index.month).unique()
    weather = read_weather_data()
    monthly_weather, source_log = build_monthly_weather(weather, months)
    daily = attach_weather(daily_power, monthly_weather)
    train, test = split_train_test(daily)

    daily_path = write_csv_with_fallback(daily, ALL_DAILY_PATH)
    train_path = write_csv_with_fallback(train, TRAIN_DAILY_PATH)
    test_path = write_csv_with_fallback(test, TEST_DAILY_PATH)
    weather_path = write_csv_with_fallback(monthly_weather, WEATHER_MONTHLY_PATH, index=False)
    source_path = write_csv_with_fallback(source_log, WEATHER_SOURCE_PATH, index=False)

    print(f"Saved {daily_path}: {daily.shape}, {daily.index.min().date()} to {daily.index.max().date()}")
    print(f"Saved {train_path}: {train.shape}, {train.index.min().date()} to {train.index.max().date()}")
    print(f"Saved {test_path}: {test.shape}, {test.index.min().date()} to {test.index.max().date()}")
    print(f"Saved {weather_path}: {monthly_weather.shape}")
    print(f"Saved {source_path}: {source_log.shape}")


if __name__ == "__main__":
    main()
