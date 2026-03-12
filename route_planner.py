import heapq
import numpy as np

# -------------------------------------------------
# Convert LSTM numeric output to crowd label
# -------------------------------------------------
def interpret_crowd(value):
    if value < 0.5:
        return "Low"
    elif value < 1.5:
        return "Medium"
    else:
        return "High"


# -------------------------------------------------
# Forecast future crowd using trained LSTM
# -------------------------------------------------
def forecast_crowd_lstm(model, last_sequence):
    """
    model: trained LSTM model
    last_sequence: list of last 3 crowd values (0, 1, 2)
    """
    seq = np.array(last_sequence).reshape((1, 3, 1))
    prediction = model.predict(seq, verbose=0)
    return interpret_crowd(prediction[0][0])


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
        crowd = forecast_crowd_lstm(model, last_3)

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