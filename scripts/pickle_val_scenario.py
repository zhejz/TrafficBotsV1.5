# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
import tensorflow as tf
import pickle


def main():
    parser = ArgumentParser(allow_abbrev=True)
    parser.add_argument("--tfrecord-dir", default="/scratch/trace01/womd_scenario_v_1_2_0/validation")
    parser.add_argument("--out-dir", default="/scratch-second/trace01/h5_wosac/val_scenarios")
    args = parser.parse_args()

    dataset_size = 44097  # for validation
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    tfrecord_files = sorted([p.as_posix() for p in Path(args.tfrecord_dir).glob("*")])
    assert len(tfrecord_files) == 150, "Validation tfrecord not complete, please dowload womd_scenario_v_1_2_0!"
    tf.config.set_visible_devices([], "GPU")
    tf_dataset = tf.data.TFRecordDataset(tfrecord_files, compression_type="")
    dataset_iterator = tf_dataset.as_numpy_iterator()
    for i, scenario_bytes in tqdm(enumerate(dataset_iterator), total=dataset_size):
        with open(out_dir / f"{i}.pickle", "wb") as handle:
            pickle.dump(scenario_bytes, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
