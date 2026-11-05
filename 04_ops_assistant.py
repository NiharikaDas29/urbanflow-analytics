"""Rule-based operations assistant backed only by the trained forecast models."""

import argparse
import os
import re

import numpy as np
import pandas as pd

from common import BUFFER, OUT_DIR, PEAK_HOURS, predict_quantiles

import joblib

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
MONTH_NAMES = {
    name: index for index, name in enumerate(
        ["january", "february", "march", "april", "may", "june",
         "july", "august", "september", "october", "november", "december"], 1
    )
}
WEEKDAYS = {
    name: index for index, name in enumerate(
        ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
    )
}
PERIODS = {
    "morning rush": [7, 8, 9],
    "evening rush": [16, 17, 18, 19],
    "rush hour": PEAK_HOURS,
    "morning": [6, 7, 8, 9, 10, 11],
    "midday": [11, 12, 13, 14],
    "afternoon": [12, 13, 14, 15, 16, 17],
    "evening": [17, 18, 19, 20],
    "night": [22, 23, 0, 1, 2],
    "overnight": [0, 1, 2, 3, 4, 5],
}
WEATHER_NAMES = {1: "clear", 2: "mist/cloud", 3: "light rain/snow", 4: "severe weather"}


class Forecaster:
    """Create estimates from the saved calendar/weather quantile model."""

    def __init__(self, model_path=None):
        path = model_path or os.path.join(OUT_DIR, "plan_model.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError("Run 03_fleet_model.py to train outputs/plan_model.joblib first.")
        artifacts = joblib.load(path)
        self.plan = artifacts["plan"]
        self.weather_by_month = artifacts["weather_by_month"]
        self.weather_all = artifacts["weather_all"]
        self.season_by_month = artifacts["season_by_month"]
        self.assumptions = artifacts["assumptions"]

    def _weather(self, month, weather):
        try:
            return self.weather_by_month.loc[(month, weather)]
        except KeyError:
            try:
                return self.weather_all.loc[weather]
            except KeyError as error:
                raise ValueError(f"No historical weather context exists for code {weather}.") from error

    def run(
        self,
        daytype=None,
        hours=None,
        weather=1,
        month=None,
        growth=1.0,
        forecast_date=None,
        weekday=None,
        holiday=False,
    ):
        """Return P50/P90/P95 estimated average concurrent use by hour.

        For a generic workday or weekend, forecasts average across weekdays in
        that type; a supplied date or weekday produces a specific-day scenario.
        """
        if weather not in WEATHER_NAMES:
            raise ValueError("Weather must be one of the source dataset codes 1 through 4.")
        if growth <= 0:
            raise ValueError("Demand growth multiplier must be greater than zero.")
        selected_hours = list(range(24)) if hours is None else list(hours)
        if not selected_hours or any(hour not in range(24) for hour in selected_hours):
            raise ValueError("Hours must be a non-empty list of integers from 0 to 23.")

        date = pd.Timestamp(forecast_date) if forecast_date is not None else None
        if date is None and weekday is None and daytype is None:
            daytype = "workday"
        if date is not None:
            month = int(date.month)
            weekday = int((date.day_of_week + 1) % 7)
        month = int(month or (date.month if date is not None else 6))
        if month not in range(1, 13):
            raise ValueError("Month must be between 1 and 12.")

        if weekday is not None:
            weekdays = [int(weekday)]
            inferred_daytype = "holiday" if holiday else (
                "workday" if weekday not in (0, 6) else "weekend"
            )
            chosen_daytype = daytype or inferred_daytype
        elif date is not None:
            weekdays = [int(date.day_of_week + 1) % 7]
            chosen_daytype = daytype or (
                "workday" if weekdays[0] not in (0, 6) else "weekend"
            )
        elif daytype == "workday":
            weekdays, chosen_daytype = [1, 2, 3, 4, 5], "workday"
        elif daytype == "weekend":
            weekdays, chosen_daytype = [0, 6], "weekend"
        elif daytype == "sunday":
            weekdays, chosen_daytype = [0], "sunday"
        elif daytype == "holiday":
            weekdays, chosen_daytype = [1], "holiday"
            holiday = True
        else:
            raise ValueError("Day type must be workday, weekend, sunday, or holiday.")

        if chosen_daytype == "holiday":
            holiday = True
        weather_context = self._weather(month, weather)
        rows = []
        for day in weekdays:
            is_workingday = int(day not in (0, 6) and not holiday)
            X = pd.DataFrame({"hr": selected_hours})
            X["hr_sin"] = np.sin(2 * np.pi * X["hr"] / 24)
            X["hr_cos"] = np.cos(2 * np.pi * X["hr"] / 24)
            X["weekday"] = day
            X["workingday"] = is_workingday
            X["holiday"] = int(holiday)
            X["mnth"] = month
            X["season"] = int(self.season_by_month.get(month, 1))
            X["temp_c"] = float(weather_context["temp_c"])
            X["hum_pct"] = float(weather_context["hum_pct"])
            X["wind_kmh"] = float(weather_context["wind_kmh"])
            X["weathersit"] = weather
            X["rain_or_snow"] = int(weather >= 3)
            X["commute_peak"] = (X["workingday"].eq(1) & X["hr"].isin(PEAK_HOURS)).astype(int)
            X["weekend_daytime"] = (
                X["workingday"].eq(0) & X["hr"].between(10, 17)
            ).astype(int)
            predictions = predict_quantiles(self.plan, X)
            for q, values in predictions.items():
                X[f"p{int(q * 100)}"] = values * growth
            rows.append(X[["hr"] + [f"p{int(q * 100)}" for q in predictions]])

        combined = pd.concat(rows, ignore_index=True)
        output = combined.groupby("hr", as_index=False).mean()
        output = output.rename(columns={
            "hr": "hour", "p50": "median_in_use", "p90": "p90_in_use", "p95": "p95_in_use",
        })
        output["fleet_needed_p95"] = np.ceil(output["p95_in_use"] * (1 + BUFFER))
        output.attrs["assumed"] = {
            "day_type": chosen_daytype,
            "weekday_scope": [_weekday_name(day) for day in weekdays],
            "forecast_date": date.date().isoformat() if date is not None else None,
            "month": MONTHS[month - 1],
            "weather": WEATHER_NAMES[weather],
            "historical_weather_context": {
                "temperature_c": round(float(weather_context["temp_c"]), 1),
                "humidity_percent": round(float(weather_context["hum_pct"]), 1),
                "wind_kmh": round(float(weather_context["wind_kmh"]), 1),
            },
            "demand_growth_multiplier": growth,
            "trip_durations_minutes_registered_casual": (
                f"{self.assumptions['registered_trip_min']}/"
                f"{self.assumptions['casual_trip_min']}"
            ),
            "fleet_buffer": self.assumptions["fleet_buffer"],
            "model_reference_period": (
                "Training data through May 2012; no future year trend is modelled"
            ),
            "estimate_definition": (
                "Model prediction of average concurrent bikes in use using assumed "
                "Little's Law trip durations; not observed station availability."
            ),
        }
        return output


def _weekday_name(day):
    return list(WEEKDAYS)[day]


def _parse_hour(token, meridiem=None):
    if ":" in token:
        hour = int(token.split(":")[0])
    else:
        hour = int(token)
    if meridiem:
        if hour < 1 or hour > 12:
            raise ValueError("12-hour clock values must be from 1 to 12.")
        hour %= 12
        if meridiem.lower() == "pm":
            hour += 12
    if hour not in range(24):
        raise ValueError("Hour must be between 0 and 23.")
    return hour


def _parse_date(text):
    """Parse unambiguous ISO, US numeric, or month-name dates from a question."""
    iso = re.search(r"\b(20\d{2}-\d{1,2}-\d{1,2})\b", text)
    if iso:
        return pd.Timestamp(iso.group(1))
    numeric = re.search(r"\b(\d{1,2}/\d{1,2}/20\d{2})\b", text)
    if numeric:
        return pd.to_datetime(numeric.group(1), format="%m/%d/%Y", errors="raise")
    named = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|"
        r"october|november|december)\s+(\d{1,2})(?:,?\s+(20\d{2}))?\b",
        text,
    )
    if named:
        year = int(named.group(3) or 2012)
        return pd.Timestamp(year=year, month=MONTH_NAMES[named.group(1)], day=int(named.group(2)))
    if re.search(r"\b(today|tomorrow)\b", text):
        base = pd.Timestamp.today().normalize()
        return base + pd.Timedelta(days=int("tomorrow" in text))
    return None


def parse_rules(question):
    """Convert supported natural-language requests to explicit forecast arguments."""
    text = question.lower().strip()
    request = {
        "intent": "forecast", "daytype": None, "hours": None, "weather": 1,
        "month": None, "growth": 1.0, "compare": False, "weather_specified": False, "date": None,
        "date_year_assumed": False, "weekday": None, "holiday": False,
        "current_fleet": None, "error": None,
    }
    if not text:
        request["error"] = "Enter a question about bike demand, weather, time, or fleet scenarios."
        return request
    try:
        request["date"] = _parse_date(text)
    except (ValueError, OverflowError) as error:
        request["error"] = f"I could not interpret that date: {error}"
        return request
    named_without_year = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|"
        r"october|november|december)\s+\d{1,2}\b(?!,?\s+20\d{2})",
        text,
    )
    request["date_year_assumed"] = named_without_year is not None

    if re.search(
        r"\b(buy|purchase|more bikes|add bikes|fleet size|enough bikes|shortage)\b"
        r"|\b(?:current\s+)?fleet\s+(?:of\s+|size\s+)?\d+\b",
        text,
    ):
        request["intent"] = "fleet"
    elif re.search(
        r"\b(idle|unused|underused|underutili[sz]ed|sitting|low use|utili[sz]ation)\b",
        text,
    ):
        request["intent"] = "idle"
    elif re.search(r"\b(when|which hours?|busiest|peak hours?|highest|lowest)\b", text):
        request["intent"] = "peak_hours"

    if re.search(r"\b(weekends?|saturdays?)\b", text):
        request["daytype"] = "weekend"
        request["weekday"] = 6 if "saturday" in text else None
    elif re.search(r"\bsundays?\b", text):
        request["daytype"], request["weekday"] = "sunday", 0
    elif re.search(r"\bholidays?\b", text):
        request["daytype"], request["holiday"] = "holiday", True
    elif re.search(r"\b(weekdays?|workdays?|working days?)\b", text):
        request["daytype"] = "workday"

    for name, day in WEEKDAYS.items():
        if re.search(rf"\b{re.escape(name)}s?\b", text):
            request["weekday"] = day
            request["daytype"] = "workday" if day not in (0, 6) else "weekend"
            break
    if request["date"] is not None:
        date_weekday = int((request["date"].day_of_week + 1) % 7)
        if request["weekday"] is not None and request["weekday"] != date_weekday:
            request["error"] = "The named weekday conflicts with the supplied date."
            return request
        is_weekend = date_weekday in (0, 6)
        if (
            (request["daytype"] == "workday" and is_weekend)
            or (request["daytype"] == "weekend" and not is_weekend)
            or (request["daytype"] == "sunday" and date_weekday != 0)
        ):
            request["error"] = "The supplied date conflicts with the requested day type."
            return request

    time_text = text
    if request["date"] is not None:
        time_text = re.sub(
            r"\b20\d{2}-\d{1,2}-\d{1,2}\b|\b\d{1,2}/\d{1,2}/20\d{2}\b"
            r"|\b(?:january|february|march|april|may|june|july|august|september|"
            r"october|november|december)\s+\d{1,2}(?:,?\s+20\d{2})?\b",
            " ",
            time_text,
        )
    range_match = re.search(
        r"\b(?:(?:between|from)\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?"
        r"\s*(?:and|to|-)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b",
        time_text,
    )
    if range_match:
        try:
            start = _parse_hour(
                range_match.group(1) + (f":{range_match.group(2)}" if range_match.group(2) else ""),
                range_match.group(3),
            )
            end = _parse_hour(
                range_match.group(4) + (f":{range_match.group(5)}" if range_match.group(5) else ""),
                range_match.group(6),
            )
        except ValueError as error:
            request["error"] = str(error)
            return request
        if end <= start:
            request["error"] = "Use an increasing same-day time range, such as 9am to 11am."
            return request
        request["hours"] = list(range(start, end))
    else:
        single = re.search(
            r"\b(?:at|around)?\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", time_text
        )
        if single:
            token = single.group(1) + (f":{single.group(2)}" if single.group(2) else "")
            try:
                request["hours"] = [_parse_hour(token, single.group(3))]
            except ValueError as error:
                request["error"] = str(error)
                return request
        else:
            twenty_four = re.search(
                r"\b(?:at|around)?\s*([01]?\d|2[0-3]):([0-5]\d)\b", time_text
            )
            if twenty_four:
                request["hours"] = [int(twenty_four.group(1))]
            else:
                ambiguous_hour = re.search(r"\b(?:at|around)\s+(\d{1,2})\b", time_text)
                if ambiguous_hour:
                    request["error"] = (
                        "That hour is ambiguous; use a time such as 8am, 5pm, or 17:00."
                    )
                    return request
                for period in sorted(PERIODS, key=len, reverse=True):
                    if re.search(rf"\b{re.escape(period)}\b", text):
                        request["hours"] = PERIODS[period]
                        break

    if re.search(r"\b(heavy rain|storm|thunderstorm|severe weather|extreme weather)\b", text):
        request["weather"] = 4
        request["weather_specified"] = True
    elif re.search(r"\b(rain|rainy|snow|wet|drizzle)\w*\b", text):
        request["weather"] = 3
        request["weather_specified"] = True
    elif re.search(r"\b(mist|cloud|fog|overcast)\w*\b", text):
        request["weather"] = 2
        request["weather_specified"] = True
    elif re.search(r"\b(clear|sunny)\b", text):
        request["weather"] = 1
        request["weather_specified"] = True
    numeric_weather = re.search(r"\bweather(?:sit)?\s*(?:code)?\s*(\d+)\b", text)
    if numeric_weather:
        code = int(numeric_weather.group(1))
        if code not in WEATHER_NAMES:
            request["error"] = "Dataset weather codes are 1, 2, 3, or 4."
            return request
        request["weather"] = code
        request["weather_specified"] = True

    for full_name, month in MONTH_NAMES.items():
        short = full_name[:3]
        if re.search(rf"\b(?:{full_name}|{short})\b", text):
            request["month"] = month
            break
    if request["month"] is None:
        season_month = re.search(r"\b(spring|summer|autumn|fall|winter)\b", text)
        if season_month:
            request["month"] = {
                "winter": 1, "spring": 4, "summer": 7, "autumn": 10, "fall": 10,
            }[season_month.group(1)]

    growth_match = re.search(
        r"\b(?:growth|grow(?:s|n|ing)?)\s+(?:(?:by|of)\s+)?"
        r"(\d+(?:\.\d+)?)\s*%",
        text,
    )
    if growth_match is None:
        growth_match = re.search(
            r"\b(\d+(?:\.\d+)?)\s*%\s*(?:growth|more|increase|higher)\b", text
        )
    if growth_match:
        request["growth"] = 1 + float(growth_match.group(1)) / 100
    else:
        decline_match = re.search(
            r"\b(\d+(?:\.\d+)?)\s*%\s*(?:decline|decrease|less|drop)\b", text
        )
        if decline_match:
            request["growth"] = 1 - float(decline_match.group(1)) / 100
            if request["growth"] <= 0:
                request["error"] = "Demand reduction must be less than 100%."
                return request

    request["compare"] = bool(re.search(r"\b(what if|versus|vs\.?|compare|comparison)\b", text))
    fleet_match = re.search(r"\b(?:current\s+)?fleet(?:\s+of|\s+size)?\s*(\d+)\b", text)
    if fleet_match:
        request["current_fleet"] = int(fleet_match.group(1))
    recognized = (
        request["date"] is not None
        or request["daytype"] is not None
        or request["weekday"] is not None
        or request["hours"] is not None
        or request["month"] is not None
        or request["weather"] != 1
        or request["growth"] != 1.0
        or request["intent"] != "forecast"
        or bool(re.search(r"\b(bike|bikes|demand|rentals?|forecast|weather)\b", text))
    )
    if not recognized:
        request["error"] = (
            "I can forecast bike use by date/day type, hour, month, weather, or growth, "
            "and compare a fleet size. Please include one of those details."
        )
    return request


def _fmt_hours(hours):
    if len(hours) == 1:
        return f"{hours[0]:02d}:00"
    return ", ".join(f"{hour:02d}:00" for hour in hours)


def answer(question, forecaster, parse=parse_rules, current_fleet=None):
    request = parse(question)
    if request.get("error"):
        return request["error"]
    fleet_value = request.get("current_fleet") or current_fleet
    date = request.get("date")
    daytype = request.get("daytype")
    if date is not None and daytype is None:
        daytype = "workday" if date.day_of_week < 5 else "weekend"
    try:
        forecast = forecaster.run(
            daytype=daytype or "workday",
            hours=request["hours"],
            weather=request["weather"],
            month=request["month"],
            growth=request["growth"],
            forecast_date=date,
            weekday=request["weekday"],
            holiday=request["holiday"],
        )
    except (ValueError, KeyError) as error:
        return f"I could not produce that forecast: {error}"
    assumptions = forecast.attrs["assumed"]
    selected = forecast

    if request["intent"] == "fleet":
        full_day = forecaster.run(
            daytype=daytype or "workday", weather=request["weather"],
            month=request["month"], growth=request["growth"], forecast_date=date,
            weekday=request["weekday"], holiday=request["holiday"],
        )
        peak = full_day.loc[full_day["fleet_needed_p95"].idxmax()]
        required = float(peak["fleet_needed_p95"])
        text = (
            f"Modelled peak-hour scenario: about {required:.0f} bikes at "
            f"{int(peak['hour']):02d}:00 (P95 estimated in-use "
            f"{peak['p95_in_use']:.0f}, plus {BUFFER:.0%} buffer). "
        )
        if fleet_value:
            if fleet_value < required:
                text += (
                    f"The stated fleet ({fleet_value}) is below this system-level scenario. "
                    "This does not prove a shortage or justify buying: validate actual inventory "
                    "and failed attempts, and check redistribution feasibility first."
                )
            else:
                text += (
                    f"The stated fleet ({fleet_value}) is at/above this aggregate scenario. "
                    "This does not rule out station-level stock-outs; redistribution cannot be "
                    "evaluated without station data."
                )
        else:
            text += "Provide the current fleet count to compare it with this scenario."
    elif request["intent"] == "idle":
        full_day = forecaster.run(
            daytype=daytype or "workday", weather=request["weather"],
            month=request["month"], growth=request["growth"], forecast_date=date,
            weekday=request["weekday"], holiday=request["holiday"],
        )
        peak_p95 = full_day["p95_in_use"].max()
        threshold = 0.3 * peak_p95
        low_use_hours = full_day.loc[full_day["median_in_use"] < threshold, "hour"].astype(int).tolist()
        text = (
            f"The modelled median in-use estimate is below 30% of the day's P95 peak "
            f"estimate in {len(low_use_hours)} hours: {_fmt_hours(low_use_hours)}. "
            "This is a demand-profile comparison, not observed idle-bike inventory."
        )
    elif request["intent"] == "peak_hours":
        top = selected.sort_values("p95_in_use", ascending=False).head(3)
        text = "Highest modelled P95 hours: " + ", ".join(
            f"{int(row.hour):02d}:00 (~{row.p95_in_use:.0f} bikes in use)"
            for row in top.itertuples()
        )
    else:
        text = (
            f"{assumptions['day_type'].replace('_', ' ').title()} "
            f"{_fmt_hours(selected['hour'].astype(int).tolist())}: maximum hourly estimate "
            f"P50 {selected['median_in_use'].max():.0f}, P90 "
            f"{selected['p90_in_use'].max():.0f}, P95 "
            f"{selected['p95_in_use'].max():.0f}; plan for about "
            f"{selected['fleet_needed_p95'].max():.0f} bikes under the assumed buffer."
        )

    if (
        request["compare"] and request["weather_specified"]
        and request["intent"] in ("forecast", "peak_hours")
    ):
        alternate_weather = 3 if request["weather"] != 3 else 1
        alternate = forecaster.run(
            daytype=daytype or "workday", hours=request["hours"],
            weather=alternate_weather, month=request["month"],
            growth=request["growth"], forecast_date=date,
            weekday=request["weekday"], holiday=request["holiday"],
        )
        base_need = max(float(selected["fleet_needed_p95"].max()), 1)
        difference = float(alternate["fleet_needed_p95"].max()) / base_need - 1
        text += (
            f" Under {WEATHER_NAMES[alternate_weather]}, modelled P95 fleet scenario "
            f"is {alternate['fleet_needed_p95'].max():.0f} ({difference:+.0%} vs. selected weather)."
        )
    elif (
        request["compare"] and request["growth"] != 1.0
        and request["intent"] in ("forecast", "peak_hours")
    ):
        baseline = forecaster.run(
            daytype=daytype or "workday", hours=request["hours"],
            weather=request["weather"], month=request["month"], growth=1.0,
            forecast_date=date, weekday=request["weekday"], holiday=request["holiday"],
        )
        base_need = max(float(baseline["fleet_needed_p95"].max()), 1)
        difference = float(selected["fleet_needed_p95"].max()) / base_need - 1
        text += (
            f" Versus the same scenario without growth, P95 fleet need changes by "
            f"{difference:+.0%} (baseline {base_need:.0f})."
        )

    text += "\nAssumptions: " + "; ".join(
        f"{key}={value}" for key, value in assumptions.items()
    )
    if request.get("date_year_assumed"):
        text += "; omitted date year assumed to be 2012"
    return text


DEMO = [
    "How many bikes do we need on a weekday morning rush?",
    "What if it rains on a weekday at 8am?",
    "When is demand highest on weekends in July?",
    "When are bikes likely to be underused on a workday?",
    "Compare a fleet of 300 with P95 need on a rainy Sunday in July.",
    "How many bikes at 5pm on a rainy workday in October?",
    "Forecast for 2012-10-15 at 17:00, clear weather.",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Run example questions and save a transcript.")
    parser.add_argument("--fleet", type=int, default=None, help="Current system fleet count for scenario comparison.")
    args = parser.parse_args()
    if args.fleet is not None and args.fleet <= 0:
        parser.error("--fleet must be a positive integer.")
    forecaster = Forecaster()
    if args.demo:
        lines = ["# Rule-based operations assistant demo\n"]
        for question in DEMO:
            response = answer(question, forecaster, current_fleet=args.fleet)
            print(f"Q: {question}\nA: {response}\n")
            lines.extend([f"**Q:** {question}\n", f"**A:** {response}\n"])
        with open(os.path.join(OUT_DIR, "assistant_demo.md"), "w", encoding="utf-8") as file:
            file.write("\n".join(lines))
    else:
        print("Ask about demand (blank line to quit). All numeric forecasts use the saved ML model.")
        while True:
            question = input("> ").strip()
            if not question:
                break
            print(answer(question, forecaster, current_fleet=args.fleet), "\n")
