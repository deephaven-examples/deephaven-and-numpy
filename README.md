# NumPy and Deephaven

This repository contains code that shows how users can combine [NumPy](https://numpy.org/) and Deephaven via three different scripts.  These scripts process data in Deephaven Tables using [NumPy](https://numpy.org/).  Both static and real-time operations are covered.

## Components

### General

- [`docker-compose.yml`](./docker-compose.yml): The Docker Compose file that defines the Docker images to be used.

### Python scripts

- [`numpy_and_learn.py`](./data/notebooks/numpy_and_learn.py): Shows how the data transfer between Deephaven tables and [NumPy ndarrays](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html?highlight=ndarray#numpy.ndarray) works with deephaven.learn.
- [`real_time_polynomial_fit.py`](./data/notebooks/real_time_polynomial_fit.py): Performs polynomial fitting in real time.

## Overview

This repository is meant to show how [NumPy](https://numpy.org/) and Deephaven can be used together to perform data processing in real-time.  The scripts in `./data/notebooks` each perform some signal analysis.  Standard Python code would likely do these operations on static data, but with Deephaven, the extension to real-time is made easy through the use of `deephaven.learn`.

These scripts are meant to be educational in nature.  The first, `numpy_and_learn.py`, is used to show how data is transferred between Deephaven and Python via the `deephaven.learn` module in both static and real-time applications.  The last, `real_time_polynomial_fit.py`, is used to show how [NumPy](https://numpy.org/) can do data smoothing in real-time.  The latter two scripts could be improved:

- `real_time_polynomial_fit.py` does real-time signal processing to remove noise from a noisy signal.  It does, however, have a problem due to the batch size and the real-time nature of the data.  How can you fix this such that the discontinuities between batches (windows) disappear?  Additionally, if you want to extend signal processing code such as this, [SciPy](https://scipy.org/) is built on top of [NumPy](https://numpy.org/) and contains a wealth of signal processing methods.  If you're interested in using SciPy, it's been added to this build by default, so you can try it for yourself!

## Get started

To use the code in this notebook, clone the repository to a location of your choice on your computer, and run `docker compose up` from the cloned repository.  If you're unsure if you have all of the dependencies, go to [deephaven-core](https://github.com/deephaven/deephaven-core) and follow the steps in the README to get started with Deephaven.

## Note

This application has been updated to be compatible with Deephaven Community Core v0.28.0 and v0.28.1. No backwards or forwards compatibility is guaranteed.
