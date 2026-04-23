# Scalable Dataset Generation and Feature Extraction for Improved Robustness in High-Traffic Environment Personal Tracker Identification

This is the Master Thesis of Stefan Richard Saxer, written in the spring semester of 2026 with the Communication Systems Group at the University of Zurich (UZH).

The related Bachelor Thesis can be found [here](https://github.com/stsaxe/Bachelor-Thesis-Stefan-Richard-Saxer).

The repositories' directories contain the following files:
- **ble**: The BLE Parser Framework responsible for parsing, masking and generating packets. 
- **bpe**: The custom BytePair Encoder for BLE packets. 
- **data_generation**: Jupyter Notebooks responsible for generating synthetic BLE data and analyzing it as well as the corresponding .yaml config files.
- **data_masking**: Jupyter Notebooks responsible for masking BLE Packets as well as the corresponding .yaml config files.
- **data_preprocessing**: Various Python functions and Notebooks to perform various pre-processing steps. These processed datasets can be found under data\csv\Processed Files.
- **evaluation_framework**: The Evaluation Framework for performing open- and closed-set evaluation. 
- **executors**: Custom executors used in data pipelines built on Task-Group-Framework. 
- **modeling**: The Implementation of the adaptive **HydraBLE** Transformer Model and various notebooks for model training, logits extraction and evaluation of the result. 
- **nrf**: The source code from Nordic Semi Conductor for BLE Packet Sniffing with the NRF DK-58420 (slightly updated so that various bgs are fixed). 
- **open_set_example**: An Example Notebook showing Open-Set Example on various MNIST datasets.
- **out**: The outputs of the process and main artifacts of work, mostly plots, pickle objects and tables. Note that the model checkpoints and logits are not in there as they are too large for GitHub.
- **plotting**: Functions and pipelines for plotting.
- **tgf**: The Implementation of my Task-Group Framework. 

To most important files and notebooks for model training are in the **modeling** folder. There are three Jupyter notebooks for each experiment. One for training the model, one for extracting the logits into 
PyTorch tensors stored on disk, and one for evaluating with various evaluation metrics. It is also necessary to create the corresponding folders for checkpoints and logits. This are always provided in paths at the top
of the notebooks and can also be seen in the gitignore file.


The folder containing the data (including PCAP, CSV and processed Parquet files) can be found on the link below on Kaggle, as it is too large for GitHub. Include the "data" folder as is on the top level directory to execute the Jupyter Notebooks locally. 

Link to data: https://www.kaggle.com/datasets/stefansaxer/ble-packets-from-tracking-devices-extended/data

**Note**: Local execution requires Python Version **3.12** or later. The requirements for the conda environment can be found in the environment.yaml file.

