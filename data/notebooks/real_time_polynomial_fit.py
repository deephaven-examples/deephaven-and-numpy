from deephaven import DynamicTableWriter
from deephaven.learn import gather
import deephaven.Types as dht
from deephaven import learn
from deephaven import Plot

import threading, time
import numpy as np

sleep_time = 1

table_writer = DynamicTableWriter(
    ["X", "Y"],
    [dht.double, dht.double]
)

data_table_live = table_writer.getTable()

def write_noisy_signal():
    x0 = 0
    x1 = 1
    step_size = 0.01
    for i in range(30):
        start = time.time()
        x = np.arange(x0, x1, step_size)
        y = 3.5 * np.sin(x) + 1.5 * np.sin(x) + 0.75 * np.sin(3.5 * x) + np.random.normal(0, 1, 100)
        for i in range(len(y)):
            table_writer.logRow(x[i], y[i])
        x0 += 1
        x1 += 1
        end = time.time()
        elapsed = end - start
        time.sleep(sleep_time - elapsed)
    
thread = threading.Thread(target = write_noisy_signal)
thread.start()

data_plot_live = Plot.plot("Raw Signal", data_table_live, "X", "Y").show()

def table_to_numpy_double(rows, columns):
    return gather.table_to_numpy_2d(rows, columns, dtype = np.double)

def numpy_to_table(data, index):
    return data[index]

def polynomial_fit(data):
    x = data[:, 0]
    y = data[:, 1]
    poly_order = 3
    z = np.polyfit(x, y, poly_order)
    p = np.poly1d(z)
    return p(x)

data_table_live_polyfitted = learn.learn(
    table = data_table_live,
    model_func = polynomial_fit, 
    inputs = [learn.Input(["X", "Y"], table_to_numpy_double)],
    outputs = [learn.Output("Fitted_Y", numpy_to_table, "double")],
    batch_size = 100
)

data_plot_live_polyfitted = Plot.plot("Raw Signal", data_table_live, "X", "Y").plot("Polyfitted Signal", data_table_live_polyfitted, "X", "Fitted_Y").show()
