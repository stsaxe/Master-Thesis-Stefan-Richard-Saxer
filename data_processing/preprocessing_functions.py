import copy
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from typing import Tuple
import numpy as np

from ble.pseudomization.pseudomizer_config import PseudomizerConfig
from executors import *
from tgf import *
from ble import BleStream, GenConfig, MaskConfig, Packet
from tgf import Pipeline, AbstractFlag, TaskGroup, Flag

import random
random.seed(0)


def process_csv_file(file: str, flags: list[AbstractFlag | str], labelColumn: str, pipeline_part_1, pipeline_part_2,
                     pickle_path: str, dataPathCSV: str) -> pd.Series:
    myPipeline = Pipeline().setPath(dataPathCSV + file + ".csv").loadData()

    new_dataset = myPipeline.setTask(copy.deepcopy(pipeline_part_1)).run()

    parent_flags = []

    for flag in flags:
        if isinstance(flag, AbstractFlag):
            parent_flags.append(flag)

        elif isinstance(flag, str):
            with open(pickle_path + flag + ".pickle", 'rb') as flag_file:
                loaded_flag = pickle.load(flag_file)

            flag_file.close()

            assert isinstance(loaded_flag, AbstractFlag)
            parent_flags.append(loaded_flag)

    processing_flag = Flag("Processing Flag", parents=parent_flags)

    new_dataset_labeled = copy.deepcopy(pipeline_part_2).process(new_dataset, flag=processing_flag)

    return new_dataset_labeled[labelColumn]


def process_pcap_file(file: str, dataPathPCAP: str) -> pd.DataFrame:
    stream = BleStream()

    stream.from_pcap_file(dataPathPCAP + file + ".pcapng", parse_mode='tolerant', update=True, ignore_error=True,
                          fill_empty_packet=True)

    return stream.to_pandas()


def process_file(file: str, flags: list[AbstractFlag | str], labelColumn: str, pipeline_part_1: TaskGroup,
                 pipeline_part_2: TaskGroup, pickle_path: str, dataPathCSV: str,
                 dataPathPCAP: str):
    labels = process_csv_file(file, flags, labelColumn, pipeline_part_1, pipeline_part_2, pickle_path, dataPathCSV)
    df = process_pcap_file(file, dataPathPCAP)

    assert len(labels) == len(
        df), f"mismatch affected file: {file},\n Length Df: {len(df)},\n,  Length Series: {len(labels)}"

    df[labelColumn] = labels
    df = df[df['Hex Data'].str.len() > 0]
    df['File'] = str(file)

    return df


def filesToDataFrame(filesDict: dict[str, list[AbstractFlag | str]],
                     pipeline_part_1: TaskGroup,
                     pipeline_part_2: TaskGroup,
                     picklePath: str,
                     dataPathCSV: str,
                     dataPathPCAP: str,
                     labelColumn: str = 'Label',
                     dropLabels: list[str] = None, ) -> pd.DataFrame:
    if dropLabels is None:
        dropLabels = []

    dfs = []

    max_workers = min(cpu_count(), len(filesDict))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_file, fileName, flags, labelColumn, pipeline_part_1, pipeline_part_2, picklePath,
                            dataPathCSV, dataPathPCAP)
            for fileName, flags in filesDict.items()
        ]

        for future in as_completed(futures):
            dfs.append(future.result())

    dataset = pd.concat(dfs)

    for label in dropLabels:
        dataset = dataset[~dataset[labelColumn].str.contains(label)]

    dataset = dataset[~dataset[labelColumn].isin(dropLabels)]

    dataset.reset_index(drop=True, inplace=True)

    return dataset


def balance_per_label_across_files_redistribute(
        df: pd.DataFrame,
        n: int,
        label_col: str,
        file_col: str,
        time_col: str,
        random_state: int,
) -> pd.DataFrame:
    df = df.sort_values([label_col, file_col, time_col]).copy()
    selected_parts = []

    rng = np.random.default_rng(random_state)

    for label, label_df in df.groupby(label_col, sort=False):
        file_groups = [(file, g) for file, g in label_df.groupby(file_col, sort=False)]
        number_of_file_groups = len(file_groups)

        capacities = [len(g) for _, g in file_groups]
        total_available = sum(capacities)
        target = n

        if total_available < n:
            raise Exception(f"insufficient number of samples for class label {label}")

        if number_of_file_groups == 0 or target == 0:
            continue

        # initial even allocation
        alloc = [target // number_of_file_groups] * number_of_file_groups
        for i in range(target % number_of_file_groups):
            alloc[i] += 1

        # cap by file capacity
        alloc = [min(a, cap) for a, cap in zip(alloc, capacities)]

        # redistribute leftover quota
        assigned = sum(alloc)
        leftover = target - assigned

        while leftover > 0:
            re_allocated = False
            for i in range(number_of_file_groups):
                if alloc[i] < capacities[i]:
                    alloc[i] += 1
                    leftover -= 1
                    re_allocated = True
                    if leftover == 0:
                        break
            if not re_allocated:
                break

        assert sum(alloc) == target, f"sum of allocations does not add up to target value {target}"

        # take allocation rows per file
        for (_, g), take_rows in zip(file_groups, alloc):
            length = len(g)

            assert length >= take_rows, "allocation exceeds dataframe length"

            if length == take_rows:
                selected_parts.append(g)
            else:
                start = rng.integers(0, length - take_rows + 1)
                selected_parts.append(g.iloc[start:start + take_rows])

    return pd.concat(selected_parts, ignore_index=True)


def split_by_file_packet_percentages(
        df: pd.DataFrame,
        file_col: str,
        time_col: str,
        train_frac: float,
        val_frac: float,
        test_frac: float,
) -> pd.DataFrame:
    if not np.isclose(train_frac + val_frac + test_frac, 1.0, atol=1e-6):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    df = df.sort_values([file_col, time_col]).copy()
    split_parts = []

    for file, g in df.groupby([file_col], sort=False):
        n = len(g)

        n_train = int(np.floor(n * train_frac))
        n_val = int(np.floor(n * val_frac))
        n_test = n - n_train - n_val
        n_total = n_train + n_val + n_test

        train_part = g.iloc[:n_train].copy()
        val_part = g.iloc[n_train:n_train + n_val].copy()
        test_part = g.iloc[n_train + n_val:].copy()

        assert len(train_part) + len(val_part) + len(test_part) == n_total

        if len(train_part) > 0:
            train_part["split"] = "train"
            split_parts.append(train_part)

        if len(val_part) > 0:
            val_part["split"] = "val"
            split_parts.append(val_part)

        if len(test_part) > 0:
            test_part["split"] = "test"
            split_parts.append(test_part)

    return pd.concat(split_parts, ignore_index=True)


def split_train_val_test(
        df: pd.DataFrame,
        file_col: str = "File",
        time_col: str = "Time",
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
):
    # Step 1: split by file within each label
    df_split = split_by_file_packet_percentages(
        df=df,
        time_col=time_col,
        file_col=file_col,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
    )

    # Step 2: separate raw splits
    train_df = df_split[df_split["split"] == "train"].copy()
    val_df = df_split[df_split["split"] == "val"].copy()
    test_df = df_split[df_split["split"] == "test"].copy()

    train_df.drop(columns=["split"], inplace=True)
    val_df.drop(columns=["split"], inplace=True)
    test_df.drop(columns=["split"], inplace=True)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    return train_df, val_df, test_df





def create_stream_idx_df(df: pd.DataFrame,
                         label_targets: dict[str, int],
                         label_column: str,
                         file_column: str,
                         source_column: str,
                         time_column: str,
                         stride: int,
                         sequence_length: int,
                         random_state: int = 0
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values([label_column, file_column, source_column, time_column]).reset_index(drop=True).copy()

    df['Sequence_ID'] =  df.groupby([label_column, file_column, source_column],sort=False).ngroup()
    df["Packet_in_Sequence"] = df.groupby("Sequence_ID").cumcount()

    sequence_rows = []
    for seq_id, g in df.groupby("Sequence_ID", sort=False):
        num_packets = len(g)
        num_streams = (
            ((num_packets - sequence_length) // stride) + 1
            if num_packets >= sequence_length
            else 0
        )

        sequence_rows.append({
            "Sequence_ID": int(seq_id),
            "Label": g[label_column].iloc[0],
            "File": g[file_column].iloc[0],
            "Source": g[source_column].iloc[0],
            "Start_Row": int(g.index[0]),
            "Num_of_Packets": int(num_packets),
            "Num_of_Streams": int(num_streams),
        })

    sequence_table = pd.DataFrame(sequence_rows)

    rng = np.random.default_rng(random_state)
    rows = []
    stream_id = 0

    available_labels = set(sequence_table['Label'].unique())
    requested_labels = set(label_targets.keys())

    missing = requested_labels - available_labels
    assert len(missing) == 0, f"Requested labels not found in data: {sorted(missing)}"

    for label, target in label_targets.items():
        g = sequence_table[
            (sequence_table["Label"] == label) & (sequence_table["Num_of_Streams"] > 0)].copy().reset_index(drop=True)


        assert int(g["Num_of_Streams"].sum()) >= target, "not sufficient streams for target value"

        k = len(g)
        alloc = np.full(k, target // k, dtype=int)
        alloc[: target % k] += 1

        capacities = g["Num_of_Streams"].to_numpy()
        alloc = np.minimum(alloc, capacities)

        leftover = target - int(alloc.sum())

        while leftover > 0:
            progressed = False
            for i in range(k):
                if alloc[i] < capacities[i]:
                    alloc[i] += 1
                    leftover -= 1
                    progressed = True
                    if leftover == 0:
                        break
            if not progressed:
                raise ValueError(f"Could not finish allocation for label {label!r}")

        for i, row in g.iterrows():
            take = int(alloc[i])
            if take == 0:
                continue

            max_windows = int(row["Num_of_Streams"])
            possible_starts = np.arange(0, max_windows * stride, stride)

            chosen = np.sort(rng.choice(possible_starts, size=take, replace=False))

            for start_packet_idx in chosen:
                rows.append({
                    "Stream_ID": stream_id,
                    "Sequence_ID": int(row["Sequence_ID"]),
                    label_column: row['Label'],
                    "Start_Packet_IDX": int(start_packet_idx),
                    "Length": int(sequence_length),
                })

                stream_id += 1

    stream_index = pd.DataFrame(rows)

    return df, sequence_table, stream_index

def prepare_synthetic_dataset(df: pd.DataFrame, n_sample: int, n_repeat: int, file_id: int) -> pd.DataFrame:
    df = df[['Time', 'Source', 'Channel', 'RSSI', 'Hex Data', 'Label']]

    df['File'] = f'File {file_id}'
    df.reset_index(drop=True, inplace=True)
    df['Packet ID'] = df.copy(deep=True).index + 1
    df.reset_index(drop=True, inplace=True)

    df = df.sample(n=n_sample, random_state=0, replace=False).reset_index(drop=True)

    df = df.loc[df.index.repeat(n_repeat)].reset_index(drop=True)

    df.sort_values(['Packet ID'], inplace=True, ascending=True)
    df.reset_index(inplace=True, drop=True)

    df['Time Delta'] = np.random.lognormal(mean=1, sigma=1, size=df.shape[0])
    df['Time Delta'] = (df['Time Delta'] * 1_000_000).astype(int)

    df['Time'] = df.groupby(['Packet ID'])['Time Delta'].cumsum()
    # create a pseudo realistic timestamp so that sorting works as expected
    df['Time'] = df['Time'] / 1_000_000 + 1776157445

    random.seed(0)
    channels = [int(i) for i in sorted(df['Channel'].unique())]

    def random_channel_selection(_):
        return random.choice(channels)

    df['Channel'] = df.apply(random_channel_selection, axis=1)

    df = df[['Time', 'Time Delta', 'Source', 'Channel', 'RSSI', 'Hex Data', 'Label', 'File']]

    return df

def prepare_real_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Label", "File", "Source", "Time"]).reset_index(drop=True)
    df['Time Delta'] = df.groupby(['Label', 'Source', 'File'])['Time'].diff().reset_index(drop=True)
    df.fillna(0, inplace=True)

    df['Time Delta'] = (df['Time Delta'] * 1_000_000).astype(int)
    df.reset_index(drop=True)

    df = df[['Time', 'Time Delta', 'Source', 'Channel', 'RSSI', 'Hex Data', 'Label', 'File']]

    return df



def create_pkt_gen_jobs(config: dict[str, int], job_size: int = 100) -> list[tuple[str, int]]:
    jobs = list()

    for config_path, size in config.items():
        remaining = size

        while remaining > 0:
            size = min(remaining, job_size)
            remaining -= size
            jobs.append((config_path, size))

    return jobs


def generate_packets(config_path: str, num_packets: int, shuffle_epoch: bool = True, random_state: int = 0) -> pd.DataFrame:
    stream = BleStream()

    gen_config = GenConfig()
    gen_config.from_yaml(config_path)

    if shuffle_epoch:
        pseudo_config = PseudomizerConfig()

        pseudo_config.seed = gen_config.global_pseudo_config.seed

        random.seed(random_state)
        rand_int = random.randint(0, 1_000_000)

        pseudo_config.epoch = rand_int

        gen_config.configure_pseudomizer(pseudo_config)

    stream.generate(gen_config, num_packets)

    return stream.to_pandas()

def execute_pkt_gen_jobs(gen_jobs: list[tuple[str, int]]):
    dfs = []

    random_states = [random.randint(0, 1_000_000) for _ in range(len(gen_jobs))]

    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = [executor.submit(generate_packets, job[0], job[1], True, random_state) for job, random_state in zip(gen_jobs, random_states)]

        for future in as_completed(futures):
            dfs.append(future.result())

    dataset = pd.concat(dfs)

    return dataset


def chunk_df(df: pd.DataFrame, n_chunks: int):
    chunk_size = len(df) // n_chunks + 1
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i+chunk_size]


def random_state_generator(n_states: int):
    random.seed(0)
    for i in range(n_states):
        yield random.randint(0, 1_000_000)


def mask(row, random_state: int, LUT: str, MaskPath: str) -> str:
    random.seed(random_state + int(row.name))

    label = row['Label']
    hex_data = row['Hex Data']

    config_name = LUT[label]

    config = MaskConfig()
    config.from_yaml(MaskPath + config_name)

    pseudo_config = PseudomizerConfig()
    pseudo_config.seed = config.global_seed
    pseudo_config.epoch = random.randint(0, 1_000_000)

    config.configure_pseudomizer(pseudo_config)

    pkt = Packet()
    pkt.from_string(hex_data, parse_mode="tolerant")

    pkt.mask(config)
    pkt.update()

    return pkt.to_string()


def apply_chunk_mask(df: pd.DataFrame, random_state: int, LUT: dict, MaskPath: str) -> pd.DataFrame:
    df['Hex Data'] = df.apply(mask,  axis=1, random_state=random_state, LUT=LUT, MaskPath=MaskPath)
    return df