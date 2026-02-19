import heapq
import numpy as np

# -------------------------------------------------
# Convert LSTM numeric output to crowd label
# -------------------------------------------------
def interpret_crowd(value):
    if value < 0.9:
        return "Low"
    elif value < 1.6:
        return "Medium"
    else:
        return "High"


# -------------------------------------------------
# Forecast future crowd using trained LSTM
# -------------------------------------------------
def forecast_crowd_lstm(model, last_sequence, arrival_hour):
    """
    model: trained LSTM model
    last_sequence: last 3 crowd levels
    arrival_hour: hour of arrival at station
    """
    seq = np.array(last_sequence).reshape((1, 3, 1))
    prediction = model.predict(seq, verbose=0)[0][0]

    # Peak-hour bias (morning & evening)
    if 8 <= arrival_hour <= 10 or 17 <= arrival_hour <= 20:
        prediction += 0.7
    elif 11 <= arrival_hour <= 16:
        prediction += 0.2
    else:
        prediction -= 0.2

    return interpret_crowd(prediction)



# -------------------------------------------------
# Dijkstra + LSTM-integrated route planner
# (CORRECT arrival time using real edge weights)
# -------------------------------------------------
def get_route_with_future_crowd(
    graph,
    source,
    destination,
    start_hour,
    model,
    station_crowd_history
):
    # ---------- DIJKSTRA INITIALIZATION ----------
    pq = []
    heapq.heappush(pq, (0, source, [source]))

    visited = set()
    min_time = {station: float("inf") for station in graph}
    min_time[source] = 0

    # ---------- RUN DIJKSTRA ----------
    while pq:
        current_time, station, path = heapq.heappop(pq)

        if station in visited:
            continue

        visited.add(station)

        if station == destination:
            break

        for neighbor, travel_time in graph[station]:
            new_time = current_time + travel_time
            if new_time < min_time[neighbor]:
                min_time[neighbor] = new_time
                heapq.heappush(
                    pq,
                    (new_time, neighbor, path + [neighbor])
                )

    route = path
    total_travel_time = min_time[destination]

    # ---------- ARRIVAL TIME + FUTURE CROWD ----------
    station_results = []

    current_time_minutes = start_hour * 60

    for i, station in enumerate(route):
        # Arrival time formatting
        arrival_hour = (current_time_minutes // 60) % 24
        arrival_minute = current_time_minutes % 60
        arrival_time = f"{arrival_hour:02d}:{arrival_minute:02d}"

        # Crowd prediction
        last_3 = station_crowd_history.get(station, [1, 1, 1])
        crowd = forecast_crowd_lstm(model, last_3, arrival_hour)


        station_results.append({
            "station": station,
            "arrival_time": arrival_time,
            "predicted_crowd": crowd
        })

        # Add real travel time to next station
        if i < len(route) - 1:
            current_station = route[i]
            next_station = route[i + 1]

            for neighbor, travel_time in graph[current_station]:
                if neighbor == next_station:
                    current_time_minutes += travel_time
                    break

    return {
        "route": route,
        "total_time": total_travel_time,
        "station_crowd": station_results
    }
