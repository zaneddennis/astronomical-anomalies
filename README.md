# CNN Autoencoder for Astronomical Transient Outlier Detection

## Data:

Raw data: `data_110620.npz`

Preprocessed data: `notebooks/fulldata.npz`

## To Run:

1. Interpolate the data to dense, regularly sampled multichannel time series

`python3 gp_interpolating.py notebooks/fulldata.npz 20000`

Arguments:

* filepath to data file, as created in PreprocessFullData.ipynb.

* number of events to sample and process

2. Encode the data into autoencoded features, using the output of the interpolation step

`python3 cnn_encoding.py interpolated.csv interpolated_labels.csv`

Arguments:

* interpolated light curve data file

* class labels file

Note that both these files are created automatically (with these exact names) by the interpolator script

3. Cluster events and evaluate

`python3 cluster_events_v2.py encodings.csv labels.csv 0.01 0.10`

Arguments:

* encodings file (automatically created by encoder script)

* labels file (automatically created by encoder script)

* cluster cutoff (X, where all clusters with size <X% of total should be considered tagged outliers)

* label cutoff (X, where all labels with size <X% of total should be considered true outliers)
