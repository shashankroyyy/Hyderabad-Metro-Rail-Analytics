import numpy as np

def interpret_crowd(value):
    if value < 0.5:
        return "Low"
    elif value < 1.5:
        return "Medium"
    else:
        return "High"


def forecast_crowd_lstm(model, last_sequence):
    """
    model: trained LSTM model
    last_sequence: list of last 3 crowd values
    """
    seq = np.array(last_sequence).reshape((1, 3, 1))
    prediction = model.predict(seq, verbose=0)
    return interpret_crowd(prediction[0][0])
