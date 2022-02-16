# NumPy and Deephaven

This repository contains code that shows how users can combine [NumPy](https://numpy.org/) and Deephaven via three different scripts.  These scripts process data in Deephaven Tables using [NumPy](https://numpy.org/).  Both static and real-time operations are covered.

## Components

### General

- [`docker-compose.yml`](./docker-compose.yml): The Docker Compose file that defines the Docker images to be used.
- [`Dockerfile`](./Dockerfile): The Dockerfile for this application.  It extends Deephaven's base images with dependencies and scripts.
- [`requirements.txt`](./requirements.txt'): The Python package dependencies required to run code in this repository.

### Python scripts

- [`numpy_and_learn.py`](./Python_Scripts/numpy_and_learn.py): Shows how the data transfer between Deephaven tables and [NumPy ndarrays](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html?highlight=ndarray#numpy.ndarray) works with deephaven.learn.
- [`bollinger_window.py`](./Python_Scripts/bollinger_window.py): Performs a moving Bollinger mean calculation for five different "stock tickers".
- [`real_time_polynomial_fit.py`](./Python_Scripts/real_time_polynomial_fit.py): Performs polynomial fitting in real time.

## Overview

This repository is meant to show how [NumPy](https://numpy.org/) and Deephaven can be used together to perform data processing in real-time.  The scripts in the `Python_Scripts` folder each perform some signal analysis.  Standard Python code would likely do these operations on static data, but with Deephaven, the extension to real-time is made easy through the use of `deephaven.learn`.

These scripts are meant to be educational in nature.  The first, `numpy_and_learn.py`, is used to show how data is transferred between Deephaven and Python via the `deephaven.learn` module in both static and real-time applications.  The second, `bollinger_window.py` performs real-time windowing using [NumPy](https://numpy.org/), where the calculations on the windowed data are very simple.  The last, `real_time_polynomial_fit.py`, is used to show how [NumPy](https://numpy.org/) can do data smoothing in real-time.  The latter two scripts could be improved:

- `bollinger_window.py` only performs simple Bollinger band calculations.  Additionally, it does it on fake data.  Code already exists that does these operations on the `trades` table in Amanda Martin's [redpanda_dxfeed_financial_data](https://github.com/deephaven-examples/redpanda-dxfeed-financial-data) application.  Could you extend the code in the `bollinger_window.py` script to work on that as well?

- `real_time_polynomial_fit.py` does real-time signal processing to remove noise from a noisy signal.  It does, however, have a problem due to the batch size and the real-time nature of the data.  How can you fix this such that the discontinuities between batches (windows) disappear?  Additionally, if you want to extend signal processing code such as this, [SciPy](https://scipy.org/) is built on top of [NumPy](https://numpy.org/) and contains a wealth of signal processing methods.  If you're interested in using SciPy, it's been added to this build by default, so you can try it for yourself!
