# Rule-based operations assistant demo

**Q:** How many bikes do we need on a weekday morning rush?

**A:** Workday 07:00, 08:00, 09:00: maximum hourly estimate P50 121, P90 155, P95 159; plan for about 184 bikes under the assumed buffer.
Assumptions: day_type=workday; weekday_scope=['monday', 'tuesday', 'wednesday', 'thursday', 'friday']; forecast_date=None; month=jun; weather=clear; historical_weather_context={'temperature_c': 28.7, 'humidity_percent': 56.5, 'wind_kmh': 11.0}; demand_growth_multiplier=1.0; trip_durations_minutes_registered_casual=12/30; fleet_buffer=0.15; model_reference_period=Training data through May 2012; no future year trend is modelled; estimate_definition=Model prediction of average concurrent bikes in use using assumed Little's Law trip durations; not observed station availability.

**Q:** What if it rains on a weekday at 8am?

**A:** Workday 08:00: maximum hourly estimate P50 92, P90 137, P95 141; plan for about 162 bikes under the assumed buffer. Under clear, modelled P95 fleet scenario is 184 (+14% vs. selected weather).
Assumptions: day_type=workday; weekday_scope=['monday', 'tuesday', 'wednesday', 'thursday', 'friday']; forecast_date=None; month=jun; weather=light rain/snow; historical_weather_context={'temperature_c': 26.2, 'humidity_percent': 78.0, 'wind_kmh': 13.0}; demand_growth_multiplier=1.0; trip_durations_minutes_registered_casual=12/30; fleet_buffer=0.15; model_reference_period=Training data through May 2012; no future year trend is modelled; estimate_definition=Model prediction of average concurrent bikes in use using assumed Little's Law trip durations; not observed station availability.

**Q:** When is demand highest on weekends in July?

**A:** Highest modelled P95 hours: 13:00 (~211 bikes in use), 17:00 (~208 bikes in use), 14:00 (~207 bikes in use)
Assumptions: day_type=weekend; weekday_scope=['sunday', 'saturday']; forecast_date=None; month=jul; weather=clear; historical_weather_context={'temperature_c': 31.2, 'humidity_percent': 56.0, 'wind_kmh': 11.0}; demand_growth_multiplier=1.0; trip_durations_minutes_registered_casual=12/30; fleet_buffer=0.15; model_reference_period=Training data through May 2012; no future year trend is modelled; estimate_definition=Model prediction of average concurrent bikes in use using assumed Little's Law trip durations; not observed station availability.

**Q:** When are bikes likely to be underused on a workday?

**A:** The modelled median in-use estimate is below 30% of the day's P95 peak estimate in 9 hours: 00:00, 01:00, 02:00, 03:00, 04:00, 05:00, 06:00, 10:00, 23:00. This is a demand-profile comparison, not observed idle-bike inventory.
Assumptions: day_type=workday; weekday_scope=['monday', 'tuesday', 'wednesday', 'thursday', 'friday']; forecast_date=None; month=jun; weather=clear; historical_weather_context={'temperature_c': 28.7, 'humidity_percent': 56.5, 'wind_kmh': 11.0}; demand_growth_multiplier=1.0; trip_durations_minutes_registered_casual=12/30; fleet_buffer=0.15; model_reference_period=Training data through May 2012; no future year trend is modelled; estimate_definition=Model prediction of average concurrent bikes in use using assumed Little's Law trip durations; not observed station availability.

**Q:** Compare a fleet of 300 with P95 need on a rainy Sunday in July.

**A:** Modelled peak-hour scenario: about 198 bikes at 12:00 (P95 estimated in-use 171, plus 15% buffer). The stated fleet (300) is at/above this aggregate scenario. This does not rule out station-level stock-outs; redistribution cannot be evaluated without station data.
Assumptions: day_type=weekend; weekday_scope=['sunday']; forecast_date=None; month=jul; weather=light rain/snow; historical_weather_context={'temperature_c': 28.7, 'humidity_percent': 79.0, 'wind_kmh': 11.0}; demand_growth_multiplier=1.0; trip_durations_minutes_registered_casual=12/30; fleet_buffer=0.15; model_reference_period=Training data through May 2012; no future year trend is modelled; estimate_definition=Model prediction of average concurrent bikes in use using assumed Little's Law trip durations; not observed station availability.

**Q:** How many bikes at 5pm on a rainy workday in October?

**A:** Workday 17:00: maximum hourly estimate P50 70, P90 124, P95 131; plan for about 151 bikes under the assumed buffer.
Assumptions: day_type=workday; weekday_scope=['monday', 'tuesday', 'wednesday', 'thursday', 'friday']; forecast_date=None; month=oct; weather=light rain/snow; historical_weather_context={'temperature_c': 19.7, 'humidity_percent': 88.0, 'wind_kmh': 15.0}; demand_growth_multiplier=1.0; trip_durations_minutes_registered_casual=12/30; fleet_buffer=0.15; model_reference_period=Training data through May 2012; no future year trend is modelled; estimate_definition=Model prediction of average concurrent bikes in use using assumed Little's Law trip durations; not observed station availability.

**Q:** Forecast for 2012-10-15 at 17:00, clear weather.

**A:** Workday 17:00: maximum hourly estimate P50 123, P90 157, P95 168; plan for about 193 bikes under the assumed buffer.
Assumptions: day_type=workday; weekday_scope=['monday']; forecast_date=2012-10-15; month=oct; weather=clear; historical_weather_context={'temperature_c': 18.9, 'humidity_percent': 67.0, 'wind_kmh': 9.0}; demand_growth_multiplier=1.0; trip_durations_minutes_registered_casual=12/30; fleet_buffer=0.15; model_reference_period=Training data through May 2012; no future year trend is modelled; estimate_definition=Model prediction of average concurrent bikes in use using assumed Little's Law trip durations; not observed station availability.
