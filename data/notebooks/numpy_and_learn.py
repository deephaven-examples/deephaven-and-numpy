# Deephaven imports
from deephaven.learn import gather
from deephaven import empty_table
from deephaven import learn

# Python imports
import numpy as np

# First table: 1 row, 1 column, int type
t1 = empty_table(1).update(formulas = ["X = i"])

# Define data exchange - Deephaven to Python
# Integer data types
def table_to_numpy_int(rows, cols):
    return gather.table_to_numpy_2d(rows, cols, np_type = np.intc)

# Float data types
def table_to_numpy_float(rows, cols):
    return gather.table_to_numpy_2d(rows, cols, np_type = np.single)

# First model function: Print the single input object
def print_array(data):
    print(f"{data.shape[0]} rows, {data.shape[1]} columns:")
    print(data)

print("\tLEARN CALL 1: Single input of 1 row and 1 column")

# The first learn call
learn.learn(                                            # No return value
    table = t1,                                         # Our input table t1
    model_func = print_array,                           # The function we will use to "process" data
    inputs = [learn.Input("X", table_to_numpy_int)],    # Inputs - must be an iterable of learn.Inputs
    outputs = None,                                     # Outputs - None (print_array returns nothing)
    batch_size = 1                                      # Batch size - the max # of rows to process at once
)

# Second table: 2 rows, 2 columns, float type
t2 = empty_table(2).update(formulas = ["X = (float)i", "Y = (float)(2 * i)"])

print("\tLEARN CALL 2: Single input of two rows and two columns")

# The second learn call
learn.learn(                                                  # No return value
    table = t2,                                               # Our input table t2
    model_func = print_array,                                 # The function we will use to "process" data
    inputs = [learn.Input(["X", "Y"], table_to_numpy_float)], # Inputs - must be an iterable of learn.Inputs
    outputs = None,                                           # Outputs - None (print_array returns nothing)
    batch_size = 2                                            # Batch size - the max # of rows to process at once
)

# Second model function: print both input objects
def print_arrays(data1, data2):
    for data in [data1, data2]:
        print(f"{data.shape[0]} rows, {data.shape[1]} columns:")
        print(data)

print("\tLEARN CALL 3: Multiple inputs - one column each")

# The third learn call
learn.learn(                                           # No return value
    table = t2,                                        # Our input table t2
    model_func = print_arrays,                         # The function we will use to "process" data
    inputs = [learn.Input("X", table_to_numpy_float),  # Input one - the X column
              learn.Input("Y", table_to_numpy_float)], # Input two - the Y column
    outputs = None,                                    # Outputs - None (print_arrays returns nothing)
    batch_size = 2                                     # Batch size - the max # of rows to process at once
)

# Third model function: return the average of each row
def row_average(col_one, col_two):
    n_vals = col_one.shape[0]
    return_array = np.zeros((2, 1))
    for i in range(col_one.shape[0]):
        return_array[i] = (col_one[i] + col_two[i]) / 2

    return return_array

# Define data exchange - Python to Deephaven
def numpy_to_table(data, index):
    return data[index]

# The fourth learn call
t3 = learn.learn(                                           # Will return values into a new table
    table = t2,                                             # Our input table t2
    model_func = row_average,                               # The function we will use to "process" data
    inputs = [learn.Input("X", table_to_numpy_float),       # Input one - the X column
              learn.Input("Y", table_to_numpy_float)],      # Input two - the Y column
    outputs = [learn.Output("Z", numpy_to_table, "float")], # Outputs - Syntax similar to inputs
    batch_size = 2                                          # Batch size - the max # of rows to process at once
)
