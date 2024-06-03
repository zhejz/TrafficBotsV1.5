# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional, Dict, Any, Tuple
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
import h5py
import pickle


class DatasetBase(Dataset[Dict[str, np.ndarray]]):
    def __init__(self, h5_filepath: str, tensor_size: Dict[str, Tuple], scenario_dir: Optional[str] = None) -> None:
        super().__init__()
        self.tensor_size = tensor_size
        self.h5_filepath = h5_filepath
        with h5py.File(self.h5_filepath, "r", libver="latest", swmr=True) as hf:
            self.dataset_len = int(hf.attrs["data_len"])

        self.scenario_dir = scenario_dir
        if self.scenario_dir is not None:
            self.scenario_dir = Path(self.scenario_dir)
            assert len(list(self.scenario_dir.glob("*"))) == self.dataset_len

    def __len__(self) -> int:
        return self.dataset_len


class DatasetTrain(DatasetBase):
    """
    The waymo 9-sec trainging.h5 is repetitive, start at {0, 2, 4, 5, 6, 8, 10} seconds within the 20-sec episode.
    Always train with the whole training.h5 dataset.
    limit_train_batches just for controlling the validation frequency.
    """

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        idx_key = str(idx)
        out_dict = {"episode_idx": idx}
        with h5py.File(self.h5_filepath, "r", libver="latest", swmr=True) as hf:
            for k in self.tensor_size.keys():
                dtype = np.float16 if hf[idx_key][k].dtype == "<f4" else None  # convert fp32 to fp16
                out_dict[k] = np.ascontiguousarray(hf[idx_key][k], dtype=dtype)
        return out_dict


class DatasetVal(DatasetBase):
    # for validation.h5 and testing.h5
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        idx_key = str(idx)
        with h5py.File(self.h5_filepath, "r", libver="latest", swmr=True) as hf:
            out_dict = {
                "episode_idx": idx,
                "scenario_id": hf[idx_key].attrs["scenario_id"],
                "scenario_center": hf[idx_key].attrs["scenario_center"],
                "scenario_yaw": hf[idx_key].attrs["scenario_yaw"],
                "with_map": hf[idx_key].attrs["with_map"],  # some epidosdes in the testing dataset do not have map.
            }
            for k, _size in self.tensor_size.items():
                dtype = np.float16 if hf[idx_key][k].dtype == "<f4" else None  # convert fp32 to fp16
                out_dict[k] = np.ascontiguousarray(hf[idx_key][k], dtype=dtype)
                if out_dict[k].shape != _size:  # manually set number of agents to be different, for scalability test.
                    assert "agent" in k
                    out_dict[k] = np.ones(_size, dtype=out_dict[k].dtype)

        if self.scenario_dir is not None:  # for WOSAC validation
            with open(self.scenario_dir / f"{idx}.pickle", "rb") as handle:
                scenario_bytes = pickle.load(handle)
                # encode to hex, otherwise pytorch collate runs very slow.
                out_dict["scenario_bytes"] = scenario_bytes.hex()
        return out_dict


class DataH5womd(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        val_scenarios_dir: Optional[str] = None,  # if not None, load waymo scenarios for validation
        filename_train: str = "training",
        filename_val: str = "validation",
        filename_test: str = "testing",
        batch_size_train: int = 3,
        batch_size_test: int = 3,
        num_workers: int = 4,
        n_ag_sim: int = 64,  # if not the same as h5 dataset, use dummy agents, for scalability tests.
    ) -> None:
        super().__init__()

        self.val_scenarios_dir = val_scenarios_dir
        self.path_train_h5 = f"{data_dir}/{filename_train}.h5"
        self.path_val_h5 = f"{data_dir}/{filename_val}.h5"
        self.path_test_h5 = f"{data_dir}/{filename_test}.h5"
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.num_workers = num_workers

        space_dim = 3
        n_mp_type, n_tl_state = 11, 5
        n_ag_type, n_ag_role, ag_size_dim, n_ag_cmd = 3, 3, 3, 8
        n_step, n_step_history = 91, 11
        n_ag_no_sim = 256
        n_mp, n_mp_pl_node = 1024, 20
        n_tl_lane, n_tl_stop = 128, 50
        self.tensor_size_train = {
            # agent states
            "agent/valid": (n_ag_sim, n_step),  # bool,
            "agent/pos": (n_ag_sim, n_step, space_dim),  # float32, x, y, z
            # v[1] = p[1]-p[0]. if p[1] invalid, v[1] also invalid, v[2]=v[3]
            "agent/vel": (n_ag_sim, n_step, 2),  # float32, v_x, v_y
            "agent/spd": (n_ag_sim, n_step, 1),  # norm of vel, signed using yaw_bbox and vel_xy
            "agent/acc": (n_ag_sim, n_step, 1),  # m/s2, acc[t] = (spd[t]-spd[t-1])/dt
            "agent/yaw_bbox": (n_ag_sim, n_step, 1),  # float32, yaw of the bbox heading
            "agent/yaw_rate": (n_ag_sim, n_step, 1),  # rad/s, yaw_rate[t] = (yaw[t]-yaw[t-1])/dt
            # agent attributes
            "agent/type": (n_ag_sim, n_ag_type),  # bool one_hot [Vehicle=0, Pedestrian=1, Cyclist=2]
            "agent/cmd": (n_ag_sim, n_ag_cmd),  # bool one_hot
            "agent/role": (n_ag_sim, n_ag_role),  # bool [sdc=0, interest=1, predict=2]
            "agent/size": (n_ag_sim, ag_size_dim),  # float32: [length, width, height]
            "agent/goal": (n_ag_sim, 4),  # float32: [x, y, theta, spd]
            "agent/dest": (n_ag_sim,),  # int64: index to map [0, n_mp]
            # map polylines
            "map/valid": (n_mp, n_mp_pl_node),  # bool
            "map/type": (n_mp, n_mp_type),  # bool one_hot
            "map/pos": (n_mp, n_mp_pl_node, space_dim),  # float32
            "map/dir": (n_mp, n_mp_pl_node, space_dim),  # float32
            "map/boundary": (4,),  # xmin, xmax, ymin, ymax
            # traffic lights lane
            "tl_lane/valid": (n_tl_lane, n_step),  # bool
            "tl_lane/state": (n_tl_lane, n_step, n_tl_state),  # bool one_hot
            "tl_lane/idx": (n_tl_lane,),  # int, -1 means not valid
            # traffic lights stop point
            "tl_stop/valid": (n_tl_stop, n_step),  # bool
            "tl_stop/state": (n_tl_stop, n_step, n_tl_state),  # bool one_hot
            "tl_stop/pos": (n_tl_stop, space_dim),  # x,y,z
            "tl_stop/dir": (n_tl_stop, space_dim),  # x,y,z
        }

        self.tensor_size_test = {
            # object_id for waymo metrics
            "history/agent/object_id": (n_ag_sim,),
            "history/agent_no_sim/object_id": (n_ag_no_sim,),
            # agent_sim
            "history/agent/valid": (n_ag_sim, n_step_history),  # bool,
            "history/agent/pos": (n_ag_sim, n_step_history, space_dim),  # float32, x, y, z
            "history/agent/vel": (n_ag_sim, n_step_history, 2),  # float32, v_x, v_y
            "history/agent/spd": (n_ag_sim, n_step_history, 1),  # norm of vel, signed using yaw_bbox and vel_xy
            "history/agent/acc": (n_ag_sim, n_step_history, 1),  # m/s2, acc[t] = (spd[t]-spd[t-1])/dt
            "history/agent/yaw_bbox": (n_ag_sim, n_step_history, 1),  # float32, yaw of the bbox heading
            "history/agent/yaw_rate": (n_ag_sim, n_step_history, 1),  # rad/s, yaw_rate[t] = (yaw[t]-yaw[t-1])/dt
            "history/agent/type": (n_ag_sim, n_ag_type),  # bool one_hot [Vehicle=0, Pedestrian=1, Cyclist=2]
            "history/agent/role": (n_ag_sim, n_ag_role),  # bool [sdc=0, interest=1, predict=2]
            "history/agent/size": (n_ag_sim, ag_size_dim),  # float32: [length, width, height]
            "history/agent_no_sim/valid": (n_ag_no_sim, n_step_history),
            "history/agent_no_sim/pos": (n_ag_no_sim, n_step_history, space_dim),
            "history/agent_no_sim/vel": (n_ag_no_sim, n_step_history, 2),
            "history/agent_no_sim/spd": (n_ag_no_sim, n_step_history, 1),
            "history/agent_no_sim/yaw_bbox": (n_ag_no_sim, n_step_history, 1),
            "history/agent_no_sim/type": (n_ag_no_sim, n_ag_type),
            "history/agent_no_sim/size": (n_ag_no_sim, ag_size_dim),
            # map
            "map/valid": (n_mp, n_mp_pl_node),  # bool
            "map/type": (n_mp, n_mp_type),  # bool one_hot
            "map/pos": (n_mp, n_mp_pl_node, space_dim),  # float32
            "map/dir": (n_mp, n_mp_pl_node, space_dim),  # float32
            "map/boundary": (4,),  # xmin, xmax, ymin, ymax
            # traffic lights lane
            "history/tl_lane/valid": (n_tl_lane, n_step_history),  # bool
            "history/tl_lane/state": (n_tl_lane, n_step_history, n_tl_state),  # bool one_hot
            "history/tl_lane/idx": (n_tl_lane,),  # int, -1 means not valid
            # traffic lights stop point
            "history/tl_stop/valid": (n_tl_stop, n_step_history),  # bool
            "history/tl_stop/state": (n_tl_stop, n_step_history, n_tl_state),  # bool one_hot
            "history/tl_stop/pos": (n_tl_stop, space_dim),  # x,y
            "history/tl_stop/dir": (n_tl_stop, space_dim),  # dx,dy
        }

        # self.tensor_size_val = {
        #     "agent/object_id": (n_ag_sim,),
        #     "agent_no_sim/object_id": (n_ag_no_sim,),
        #     "agent_no_sim/valid": (n_ag_no_sim, n_step),  # bool,
        #     "agent_no_sim/pos": (n_ag_no_sim, n_step, space_dim),  # float32, x, y, z
        #     "agent_no_sim/vel": (n_ag_no_sim, n_step, 2),  # float32, v_x, v_y
        #     "agent_no_sim/spd": (n_ag_no_sim, n_step, 1),  # norm of vel, signed using yaw_bbox and vel_xy
        #     "agent_no_sim/yaw_bbox": (n_ag_no_sim, n_step, 1),  # float32, yaw of the bbox heading
        #     "agent_no_sim/type": (n_ag_no_sim, n_ag_type),  # bool one_hot [Vehicle=0, Pedestrian=1, Cyclist=2]
        #     "agent_no_sim/size": (n_ag_no_sim, ag_size_dim),  # float32: [length, width, height]
        # }
        self.tensor_size_val = self.tensor_size_train | self.tensor_size_test

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = DatasetTrain(self.path_train_h5, self.tensor_size_train)
            self.val_dataset = DatasetVal(self.path_val_h5, self.tensor_size_val, self.val_scenarios_dir)
        elif stage == "validate":
            self.val_dataset = DatasetVal(self.path_val_h5, self.tensor_size_val, self.val_scenarios_dir)
        elif stage == "test":
            self.test_dataset = DatasetVal(self.path_test_h5, self.tensor_size_test)

    def train_dataloader(self) -> DataLoader[Any]:
        return self._get_dataloader(self.train_dataset, self.batch_size_train, self.num_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader[Any]:
        return self._get_dataloader(self.val_dataset, self.batch_size_test, self.num_workers, shuffle=False)

    def test_dataloader(self) -> DataLoader[Any]:
        return self._get_dataloader(self.test_dataset, self.batch_size_test, self.num_workers, shuffle=False)

    @staticmethod
    def _get_dataloader(ds: Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader[Any]:
        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=shuffle,
            drop_last=False,
            persistent_workers=True,
        )
