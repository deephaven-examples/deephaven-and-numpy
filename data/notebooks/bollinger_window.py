from deephaven.constants import NULL_DOUBLE
from deephaven import DynamicTableWriter
from deephaven.time import millis, now
from deephaven import dtypes as dht
from deephaven.learn import gather
from deephaven import learn

import numpy as np, random, threading, time

table_writer = DynamicTableWriter(
    {"Millis": dht.double, "Ticker": dht.string, "Price": dht.double}
)

t = table_writer.table

def write_to_t():
    syms = ["A", "B", "C", "D", "E"]
    for i in range(45):
        start = time.time()
        current_millis = float(millis(now()))
        current_ticker = syms[random.randrange(0, 5)]
        current_price = random.uniform(10, 100)
        mask = np.random.uniform(0, 1)
        if mask <= 0.15:
            current_price = NULL_DOUBLE
        table_writer.write_row(current_millis, current_ticker, current_price)
        end = time.time()
        time.sleep(1 - (end - start))

thread = threading.Thread(target = write_to_t)
thread.start()

symbols = {}
n_symbols = 0
def get_sym_number(symbol):
    global symbols, n_symbols
    if symbol not in symbols:
        symbols[symbol] = float(n_symbols)
        n_symbols += 1
    return symbols[symbol]

t = t.update(formulas = ["TickerIndex = (double)get_sym_number(Ticker)"])

# Set constants
n_seconds = 30
n_millis = n_seconds * 1000
n_valid_trades = 5
tickers = list(symbols.keys())
n_tickers = 5

window_length = 10

price_windows = -1 * np.ones((window_length, n_tickers), dtype = np.double)
time_windows = -1 * np.ones((window_length, n_tickers), dtype = np.double)

def validate_calc_bollinger_mean(data):
    global price_windows, time_windows, n_millis

    current_millis = data[0][0]
    price = data[0][1]
    sym_index = int(data[0][2])

    if price and price > 0:

        price_windows[:, sym_index] = np.roll(price_windows[:, sym_index], -1)
        price_windows[-1, sym_index] = price
        time_windows[:, sym_index] = np.roll(time_windows[:, sym_index], -1)
        time_windows[-1, sym_index] = current_millis

        num_valid_prices = np.count_nonzero(price_windows[:, sym_index] > 0)

        first_valid_index = np.argmax(time_windows[:, sym_index] > (current_millis - n_millis))
        first_valid_millis = time_windows[first_valid_index, sym_index]
        # first_valid_millis = time_windows[-n_valid_trades, sym_index]

        # Have enough trades happened?
        if np.count_nonzero(price_windows[first_valid_index:, sym_index] > 0) >= n_valid_trades:

            # Return the mean of the window (ignore NaNs)
            return np.array([np.mean(price_windows[first_valid_index:, sym_index])])

    # Otherwise, return null
    return np.array([NULL_DOUBLE])

def table_to_numpy_double(rows, columns):
    return gather.table_to_numpy_2d(rows, columns, np_type = np.double)

def numpy_to_table(data, index):
    return data

t2 = learn.learn(
    table = t,
    model_func = validate_calc_bollinger_mean,
    inputs = [learn.Input(["Millis", "Price", "TickerIndex"], table_to_numpy_double)],
    outputs = [learn.Output("Bollinger_Mean", numpy_to_table, "double")],
    batch_size = 1
)

t2a = t2.where(filters = ["Ticker == `A`"])
t2b = t2.where(filters = ["Ticker == `B`"])
t2c = t2.where(filters = ["Ticker == `C`"])
t2d = t2.where(filters = ["Ticker == `D`"])
t2e = t2.where(filters = ["Ticker == `E`"])