import hashlib
import os

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class F16FlightDatasetCorrected:
    """
    F-16 flight maneuver dataset manager.

    Responsibilities:
    - resolve data paths safely on Windows/Linux
    - map filenames to semantic labels
    - split raw sequences before any window augmentation
    - optionally append first-order delta features
    - normalize with a scaler fit on the training split only
    """

    def __init__(
        self,
        data_folder="flight_data",
        time_steps=10,
        features_per_step=8,
        feature_names=None,
        windows=None,
        add_delta=True,
        normalize=True,
        train_ratio=0.8,
        window_strides=None,
        eval_perturbation=None,
    ):
        if not os.path.isabs(data_folder):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_folder = os.path.join(base_dir, data_folder)
        self.data_folder = os.path.abspath(data_folder)

        self.time_steps = time_steps
        self.base_features_per_step = features_per_step
        self.features_per_step = features_per_step
        self.windows = windows if windows is not None else [time_steps]
        self.add_delta = add_delta
        self.normalize = normalize
        self.train_ratio = train_ratio
        self.window_strides = window_strides or {}
        self.eval_perturbation = eval_perturbation or {}

        self.feature_names = feature_names or [
            "longitude",
            "latitude",
            "altitude",
            "roll",
            "pitch",
            "yaw",
            "roll_rate",
            "feature8",
        ]

        self.scaler = StandardScaler()
        self.raw_data = []
        self.processed_data = None

        self.flight_actions = {
            "01up.txt": "Up",
            "02Level Flight.txt": "Level Flight",
            "03Descent.txt": "Descent",
            "04Turn right.txt": "Turn Right",
            "05Turn left.txt": "Turn Left",
            "06Turn right up.txt": "Turn Right Up",
            "07Turn right descent.txt": "Turn Right Descent",
            "08Turn left up.txt": "Turn Left Up",
            "09Turn left descent.txt": "Turn Left Descent",
            "10Vertical turn up.txt": "Vertical Turn Up",
            "11Roll right.txt": "Roll Right",
            "12Roll left.txt": "Roll Left",
            "13Vertical turn descetn.txt": "Vertical Turn Descent",
        }

        self.class_names = sorted(self.flight_actions.values())
        self.label_mapping = {name: idx for idx, name in enumerate(self.class_names)}
        self.index_to_class = {idx: name for name, idx in self.label_mapping.items()}

        print(
            f"Initialize F16 dataset\n"
            f"  data_dir: {self.data_folder}\n"
            f"  time_steps={self.time_steps}, features={self.features_per_step}\n"
            f"  windows={self.windows}, add_delta={self.add_delta}\n"
            f"  train_ratio={self.train_ratio}, window_strides={self.window_strides}\n"
            f"  eval_perturbation={self.eval_perturbation}\n"
            f"  num_classes={len(self.class_names)}"
        )

    def _resample_time_series(self, seq, target_len):
        seq = np.asarray(seq, dtype=np.float32)
        length, feat_dim = seq.shape
        if length == target_len:
            return seq.copy()

        orig_t = np.linspace(0.0, 1.0, length)
        target_t = np.linspace(0.0, 1.0, target_len)
        out = np.zeros((target_len, feat_dim), dtype=np.float32)

        for feat_idx in range(feat_dim):
            out[:, feat_idx] = np.interp(target_t, orig_t, seq[:, feat_idx])
        return out

    def _make_delta(self, seq):
        diff = np.diff(seq, axis=0)
        zeros = np.zeros((1, seq.shape[1]), dtype=seq.dtype)
        return np.vstack([zeros, diff]).astype(np.float32, copy=False)

    def _stable_hash(self, seq):
        arr = np.ascontiguousarray(seq)
        return hashlib.sha256(arr.tobytes()).hexdigest()

    def _get_window_stride(self, win):
        stride = self.window_strides.get(win, 1)
        if stride <= 0:
            raise ValueError(f"Window stride must be positive, got {stride} for win={win}")
        return stride

    def _compute_split_indices(self, num_items, train_ratio, purge_gap):
        if not 0 < train_ratio < 1:
            raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
        if purge_gap < 0:
            raise ValueError(f"purge_gap must be >= 0, got {purge_gap}")
        if num_items <= 1:
            return num_items, num_items

        train_end = int(num_items * train_ratio)
        train_end = min(max(train_end, 1), num_items - 1)

        # Keep at least one sample on the test side when possible.
        max_gap = max(0, num_items - train_end - 1)
        effective_gap = min(purge_gap, max_gap)
        test_start = train_end + effective_gap
        return train_end, test_start

    def _shift_sequence(self, seq, offset):
        if offset == 0:
            return seq
        shifted = np.empty_like(seq)
        if offset > 0:
            shifted[:offset] = seq[:1]
            shifted[offset:] = seq[:-offset]
        else:
            offset = abs(offset)
            shifted[-offset:] = seq[-1:]
            shifted[:-offset] = seq[offset:]
        return shifted

    def _apply_eval_perturbation(self, X, rng):
        if len(X) == 0 or not self.eval_perturbation:
            return X

        X = X.copy()
        noise_sigma = float(self.eval_perturbation.get("noise_sigma", 0.0))
        missing_prob = float(self.eval_perturbation.get("missing_prob", 0.0))
        time_shift_range = int(self.eval_perturbation.get("time_shift_range", 0))

        if noise_sigma > 0:
            noise = rng.normal(0.0, noise_sigma, size=X.shape).astype(np.float32)
            X += noise

        if missing_prob > 0:
            missing_mask = rng.random(X.shape) < missing_prob
            X[missing_mask] = 0.0

        if time_shift_range > 0:
            offsets = rng.integers(-time_shift_range, time_shift_range + 1, size=len(X))
            for idx, offset in enumerate(offsets):
                X[idx] = self._shift_sequence(X[idx], int(offset))

        return X.astype(np.float32, copy=False)

    def _generate_windows(self, seq):
        """
        Expand one raw sequence into the configured window views.

        Raw sequences are always split before this function is called, which
        prevents derived windows from the same source sequence from landing in
        both train and test splits.
        """
        seq = np.asarray(seq, dtype=np.float32)
        time_len, _ = seq.shape

        if time_len != self.time_steps:
            seq = self._resample_time_series(seq, self.time_steps)
            time_len = self.time_steps

        windows = []
        for win in self.windows:
            if win <= 0:
                raise ValueError(f"Window size must be positive, got {win}")

            if win >= time_len:
                windows.append(seq.copy())
                continue

            stride = self._get_window_stride(win)
            for start in range(0, time_len - win + 1, stride):
                crop = seq[start:start + win]
                windows.append(self._resample_time_series(crop, self.time_steps))

        return windows

    def load_data(self):
        print("Loading raw data...")
        if not os.path.isdir(self.data_folder):
            raise FileNotFoundError(f"Data directory not found: {self.data_folder}")

        self.raw_data = []
        total_files = 0

        for filename, action_name in self.flight_actions.items():
            file_path = os.path.join(self.data_folder, filename)
            if not os.path.isfile(file_path):
                print(f"[WARN] Missing file, skip: {filename}")
                continue

            total_files += 1
            label = self.label_mapping[action_name]

            with open(file_path, "r", encoding="utf-8") as file_obj:
                lines = file_obj.readlines()

            sample_count = 0
            expected_len = 1 + self.time_steps * self.base_features_per_step
            for line in lines:
                values = line.strip().split(",")
                if len(values) < expected_len:
                    continue

                try:
                    feats = np.array(
                        list(map(float, values[1:expected_len])),
                        dtype=np.float32,
                    ).reshape(self.time_steps, self.base_features_per_step)
                except Exception:
                    continue

                self.raw_data.append((feats, label))
                sample_count += 1

            print(f"Loaded {filename}: {sample_count} raw sequences")

        print(
            f"Raw data loading complete | files: {total_files} | "
            f"sequences: {len(self.raw_data)}"
        )

        if not self.raw_data:
            raise RuntimeError("No samples were loaded")

        return self

    def preprocess_data(
        self,
        test_size=None,
        random_state=42,
        split_mode="grouped_random",
        purge_gap=0,
        train_ratio=None,
    ):
        if test_size is not None:
            inferred_train_ratio = 1.0 - test_size
            if train_ratio is not None and abs(train_ratio - inferred_train_ratio) > 1e-8:
                raise ValueError("Specify either test_size or train_ratio, not conflicting values")
            train_ratio = inferred_train_ratio

        train_ratio = self.train_ratio if train_ratio is None else train_ratio
        print(
            f"Preprocessing data... split_mode={split_mode}, "
            f"train_ratio={train_ratio}, purge_gap={purge_gap}"
        )
        import random

        rng = random.Random(random_state)
        np_rng = np.random.default_rng(random_state)
        self.features_per_step = self.base_features_per_step

        train_raw, y_train_raw = [], []
        test_raw, y_test_raw = [], []

        if split_mode == "grouped_random":
            # Group by label and raw-sequence hash so identical source sequences
            # stay on the same side of the split.
            label_to_groups = {}
            for seq, label in self.raw_data:
                seq_groups = label_to_groups.setdefault(label, {})
                seq_groups.setdefault(self._stable_hash(seq), []).append(seq)

            for label, seq_groups in label_to_groups.items():
                groups = list(seq_groups.values())
                rng.shuffle(groups)

                train_end, test_start = self._compute_split_indices(
                    len(groups),
                    train_ratio=train_ratio,
                    purge_gap=purge_gap,
                )

                for group in groups[:train_end]:
                    for seq in group:
                        train_raw.append(seq)
                        y_train_raw.append(label)

                for group in groups[test_start:]:
                    for seq in group:
                        test_raw.append(seq)
                        y_test_raw.append(label)

        elif split_mode == "contiguous":
            # Preserve within-file temporal order and split each label into a
            # front contiguous train block and a tail contiguous test block.
            label_to_seqs = {}
            for seq, label in self.raw_data:
                label_to_seqs.setdefault(label, []).append(seq)

            for label, seqs in label_to_seqs.items():
                train_end, test_start = self._compute_split_indices(
                    len(seqs),
                    train_ratio=train_ratio,
                    purge_gap=purge_gap,
                )

                for seq in seqs[:train_end]:
                    train_raw.append(seq)
                    y_train_raw.append(label)

                for seq in seqs[test_start:]:
                    test_raw.append(seq)
                    y_test_raw.append(label)

        else:
            raise ValueError(
                "split_mode must be one of {'grouped_random', 'contiguous'}"
            )

        def make_windows_with_labels(seqs, labels):
            out_x = []
            out_y = []
            for seq, label in zip(seqs, labels):
                for window_seq in self._generate_windows(seq):
                    out_x.append(window_seq)
                    out_y.append(label)

            return (
                np.asarray(out_x, dtype=np.float32),
                np.asarray(out_y, dtype=np.int64),
            )

        X_train, y_train = make_windows_with_labels(train_raw, y_train_raw)
        X_test, y_test = make_windows_with_labels(test_raw, y_test_raw)

        if self.add_delta:
            delta_train = np.asarray([self._make_delta(seq) for seq in X_train])
            X_train = np.concatenate([X_train, delta_train], axis=2)

            delta_test = np.asarray([self._make_delta(seq) for seq in X_test])
            X_test = np.concatenate([X_test, delta_test], axis=2)

            self.features_per_step = self.base_features_per_step * 2
            print(f"Delta features appended -> F={self.features_per_step}")

        print(f"Dedup guard | Train: {len(X_train)}, Test: {len(X_test)}")

        train_hashes = {self._stable_hash(sample) for sample in X_train}
        unique_indices = []
        duplicate_count = 0

        for idx, sample in enumerate(X_test):
            if self._stable_hash(sample) in train_hashes:
                duplicate_count += 1
            else:
                unique_indices.append(idx)

        if duplicate_count > 0:
            print(
                f"Removed {duplicate_count} duplicate test samples to prevent "
                f"data leakage"
            )
            X_test = X_test[unique_indices]
            y_test = y_test[unique_indices]
        else:
            print("No duplicate samples found across train/test splits")

        if self.normalize:
            train_count, train_time, feat_dim = X_train.shape
            X_train = self.scaler.fit_transform(
                X_train.reshape(-1, feat_dim)
            ).reshape(train_count, train_time, feat_dim)

            test_count = len(X_test)
            if test_count > 0:
                X_test = self.scaler.transform(
                    X_test.reshape(-1, feat_dim)
                ).reshape(test_count, train_time, feat_dim)

        X_test = self._apply_eval_perturbation(X_test, np_rng)

        self.processed_data = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

        print(f"Final dataset | Train: {len(X_train)} | Test: {len(X_test)}")
        return self

    def get_dataloader(
        self,
        batch_size=32,
        augment=False,
        noise_sigma=0.01,
        missing_prob=0.0,
        time_shift_range=0,
        scale_sigma=0.0,
        feature_dropout_prob=0.0,
        use_weighted_sampler=False,
        class_sampling_weights=None,
    ):
        train_ds = TorchFlightDataset(
            self.processed_data["X_train"],
            self.processed_data["y_train"],
            augment=augment,
            noise_sigma=noise_sigma,
            missing_prob=missing_prob,
            time_shift_range=time_shift_range,
            scale_sigma=scale_sigma,
            feature_dropout_prob=feature_dropout_prob,
        )
        test_ds = TorchFlightDataset(
            self.processed_data["X_test"],
            self.processed_data["y_test"],
        )

        sampler = None
        if use_weighted_sampler:
            train_labels = self.processed_data["y_train"]
            if class_sampling_weights is None:
                class_counts = np.bincount(train_labels)
                class_sampling_weights = 1.0 / np.clip(class_counts, 1, None)

            class_sampling_weights = np.asarray(class_sampling_weights, dtype=np.float64)
            if class_sampling_weights.ndim != 1:
                raise ValueError("class_sampling_weights must be a 1D array-like object")

            if len(class_sampling_weights) <= int(np.max(train_labels)):
                raise ValueError("class_sampling_weights length does not cover all labels")

            sample_weights = class_sampling_weights[train_labels]
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )

        return (
            DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=sampler is None,
                sampler=sampler,
            ),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        )


class TorchFlightDataset(Dataset):
    """Yield tensors shaped as (1, F, T) for the CNN."""

    def __init__(
        self,
        X,
        y,
        augment=False,
        noise_sigma=0.01,
        missing_prob=0.0,
        time_shift_range=0,
        scale_sigma=0.0,
        feature_dropout_prob=0.0,
    ):
        self.X = torch.FloatTensor(X).transpose(1, 2)
        self.y = torch.LongTensor(y)
        self.augment = augment
        self.noise_sigma = noise_sigma
        self.missing_prob = missing_prob
        self.time_shift_range = time_shift_range
        self.scale_sigma = scale_sigma
        self.feature_dropout_prob = feature_dropout_prob

    @staticmethod
    def _shift_along_time(x, offset):
        if offset == 0:
            return x

        shifted = torch.empty_like(x)
        if offset > 0:
            shifted[:, :offset] = x[:, :1]
            shifted[:, offset:] = x[:, :-offset]
        else:
            offset = abs(offset)
            shifted[:, -offset:] = x[:, -1:]
            shifted[:, :-offset] = x[:, offset:]
        return shifted

    def _augment_sample(self, x):
        if self.scale_sigma > 0:
            scale = 1.0 + torch.randn(x.size(0), 1) * self.scale_sigma
            x = x * scale

        if self.noise_sigma > 0:
            x = x + torch.randn_like(x) * self.noise_sigma

        if self.feature_dropout_prob > 0:
            feature_mask = torch.rand(x.size(0), 1) < self.feature_dropout_prob
            x = x.masked_fill(feature_mask.expand_as(x), 0.0)

        if self.missing_prob > 0:
            missing_mask = torch.rand_like(x) < self.missing_prob
            x = x.masked_fill(missing_mask, 0.0)

        if self.time_shift_range > 0:
            shift = int(torch.randint(
                -self.time_shift_range,
                self.time_shift_range + 1,
                (1,),
            ).item())
            x = self._shift_along_time(x, shift)

        return x

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            x = self._augment_sample(x)
        return x.unsqueeze(0), self.y[idx]
