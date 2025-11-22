import os
import json
import argparse
import numpy as np
from sklearn.metrics import pairwise_distances
from PIL import Image
import cv2


def rotation_matrix_from_vectors(vec1, vec2):
    """Find a rotation matrix that aligns vec1 to vec2.
    :param vec1: A 3D source vector
    :param vec2: A 3D target vector
    :return: 3x3 rotation matrix R such that R @ vec1 ≈ vec2
    """
    a = (vec1 / np.linalg.norm(vec1)).reshape(3)
    b = (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    kmat = np.array([
        [0,     -v[2],  v[1]],
        [v[2],   0,    -v[0]],
        [-v[1],  v[0],  0   ],
    ])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def generate_samples(n_samples, reference_translations,
                     distance_factor=0.8, deviation_factor=4.0):
    """
    Sample new camera positions based on the distribution of existing positions.

    Steps:
        - Compute the mean point of all reference camera positions
        - Compute the distances from the mean to each training position
        - Fit a Normal(mu, sigma) to these distances
        - Sample new distances = Normal(mu * distance_factor, sigma * deviation_factor)
        - Sample random direction vectors in 3D
        - Scale each direction to match the sampled distance and add to the mean point
    """
    middle = np.mean(reference_translations, axis=0)

    distances = np.linalg.norm(reference_translations - middle, axis=1)
    mu, sigma = np.mean(distances), np.std(distances)

    print(
        f"Sampling with mu: {mu:1.3f} * {distance_factor} = {(mu * distance_factor):1.3f} "
        f"and sigma: {sigma:1.3f} * {deviation_factor} = {(sigma * deviation_factor):1.3f}"
    )

    sample_dists = np.random.normal(
        mu * distance_factor,
        sigma * deviation_factor,
        size=n_samples
    )

    # random direction vectors in [-1, 1]^3
    direction = (np.random.rand(n_samples, 3) * 2) - 1
    dist_to_dir = np.linalg.norm(direction, axis=1)
    direction_multiplier = sample_dists / dist_to_dir

    new_points = np.broadcast_to(direction_multiplier[..., None], (n_samples, 3)) * direction + middle
    return new_points


def fix_dataset_filenames(dataset_root, classnames=None, zero_padding=3):
    """
    General version: unifies filenames for datasets following a NeRF-like structure.

    Expected dataset structure:
        dataset_root/
          <class_name>/
            train/good/*.png
            test/<various_defect_folders>/*.png
            ground_truth/<various_defect_folders>/*.png

    Behavior:
        - For train/test images: convert "3.png" → "003.png"
        - For mask images: convert "3_xxx.png" → "003_mask.png"
    """

    def regular_loop(cur_dir):
        """Rename standard images: <idx>.png → <idx:03d>.png"""
        if not os.path.isdir(cur_dir):
            return
        for entry in os.listdir(cur_dir):
            src = os.path.join(cur_dir, entry)
            if not os.path.isfile(src):
                continue

            name, ext = os.path.splitext(entry)
            if ext.lower() not in [".png", ".jpg", ".jpeg"]:
                continue

            try:
                idx = int(name)
            except ValueError:
                continue

            new_name = f"{idx:0{zero_padding}d}.png"
            dst = os.path.join(cur_dir, new_name)
            if src != dst:
                os.rename(src, dst)

    def masks_loop(cur_dir):
        """Rename masks: <idx>_<something>.png → <idx:03d>_mask.png"""
        if not os.path.isdir(cur_dir):
            return
        for entry in os.listdir(cur_dir):
            src = os.path.join(cur_dir, entry)
            if not os.path.isfile(src):
                continue

            name, ext = os.path.splitext(entry)
            if ext.lower() not in [".png", ".jpg", ".jpeg"]:
                continue

            parts = name.split("_")
            try:
                idx = int(parts[0])
            except ValueError:
                continue

            new_name = f"{idx:0{zero_padding}d}_mask.png"
            dst = os.path.join(cur_dir, new_name)
            if src != dst:
                os.rename(src, dst)

    # Auto-detect classes if not provided
    if classnames is None:
        classnames = [
            d for d in os.listdir(dataset_root)
            if os.path.isdir(os.path.join(dataset_root, d))
        ]

    for cl in classnames:
        class_path = os.path.join(dataset_root, cl)

        # train/good
        regular_loop(os.path.join(class_path, "train", "good"))

        # test subfolders
        test_root = os.path.join(class_path, "test")
        if os.path.isdir(test_root):
            for sub in os.listdir(test_root):
                regular_loop(os.path.join(test_root, sub))

        # ground_truth subfolders
        gt_root = os.path.join(class_path, "ground_truth")
        if os.path.isdir(gt_root):
            for sub in os.listdir(gt_root):
                masks_loop(os.path.join(gt_root, sub))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a dataset for 3DGS/NeRF by adding alpha masks, "
                    "fixing filenames, and generating additional test poses."
    )
    parser.add_argument("--data_root", type=str, required=True,
                        help="Input dataset root directory.")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Output dataset root directory.")
    parser.add_argument("--k_augments", type=int, default=5,
                        help="Number of augmented test poses (doubled internally).")
    parser.add_argument("--prepare_pose_dataset", action="store_true",
                        help="Only split train/test poses without generating new test poses.")
    parser.add_argument("--split_to_train", type=float, default=1.0,
                        help="Fraction of train frames kept when splitting poses.")
    parser.add_argument("--classnames", type=str, default=None,
                        help="Comma-separated list of class names to process.")
    parser.add_argument("--test_image_size", type=int, default=800,
                        help="Size (H=W) of generated placeholder test images.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    data_root = args.data_root
    output_root = args.output_root
    k_augments = args.k_augments
    split_to_train = args.split_to_train
    prepare_pose_dataset = args.prepare_pose_dataset
    test_image_size = args.test_image_size

    # Determine classes
    if args.classnames is not None:
        classnames = [c.strip() for c in args.classnames.split(",") if c.strip()]
    else:
        classnames = [
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
        ]

    os.makedirs(output_root, exist_ok=True)

    print("Fixing filenames...")
    fix_dataset_filenames(data_root, classnames)

    for cl in classnames:
        print(f"Processing class {cl}")

        class_in = os.path.join(data_root, cl)
        class_out = os.path.join(output_root, cl)
        os.makedirs(class_out, exist_ok=True)

        # --------------------------------------------
        # 1. Add alpha channel to training images
        # --------------------------------------------
        orig_train_dir = os.path.join(class_in, "train", "good")
        new_train_dir = os.path.join(class_out, "train")
        os.makedirs(new_train_dir, exist_ok=True)

        new_test_dir = os.path.join(class_out, "test")
        os.makedirs(new_test_dir, exist_ok=True)

        train_files = sorted(
            os.listdir(orig_train_dir),
            key=lambda x: int(x.split("_")[-1].split(".")[0])
        )

        n_train = len(train_files)
        chosen_train = np.random.choice(n_train, int(n_train * split_to_train), replace=False)
        chosen_test = np.array([i for i in range(n_train) if i not in chosen_train])

        for idx, fname in enumerate(train_files):
            img_path = os.path.join(orig_train_dir, fname)
            img = cv2.imread(img_path)

            # Create alpha mask using simple thresholding on green channel
            mask = np.abs(
                (cv2.threshold(img[:, :, 1], 254, 1, cv2.THRESH_BINARY)[1]) - 1
            ).astype(np.uint8)

            rgba = np.dstack((img, mask))

            stem = fname.split(".")[0]
            if prepare_pose_dataset:
                # Split train/test by random selection
                if idx in chosen_train:
                    out_path = os.path.join(new_train_dir, f"train_{stem}.png")
                else:
                    out_path = os.path.join(new_test_dir, f"test_{stem}.png")
            else:
                out_path = os.path.join(new_train_dir, f"train_{stem}.png")

            ok = cv2.imwrite(out_path, rgba)
            if not ok:
                raise RuntimeError(f"Failed to save RGBA: {out_path}")

        # --------------------------------------------
        # 2. Load original camera transforms
        # --------------------------------------------
        transforms_path = os.path.join(class_in, "transforms.json")
        with open(transforms_path, "r") as f:
            train_trans = json.load(f)

        camera_angle_x = train_trans.get("camera_angle_x", None)

        if prepare_pose_dataset:
            # --------------------------------------------
            # Only split train/test based on original poses
            # --------------------------------------------
            new_train_json = {"camera_angle_x": camera_angle_x, "frames": []}
            new_test_json = {"camera_angle_x": camera_angle_x, "frames": []}

            for idx in range(n_train):
                frame = train_trans["frames"][idx]
                cur_idx = int(frame["file_path"].split("/")[-1].split(".")[0])

                if idx in chosen_train:
                    file_path = f"./train/train_{cur_idx:03d}"
                    new_train_json["frames"].append(
                        {"file_path": file_path, "transform_matrix": frame["transform_matrix"]}
                    )
                else:
                    file_path = f"./test/test_{cur_idx:03d}"
                    new_test_json["frames"].append(
                        {"file_path": file_path, "transform_matrix": frame["transform_matrix"]}
                    )

            with open(os.path.join(class_out, "transforms_train.json"), "w") as f:
                json.dump(new_train_json, f, indent=2)

            with open(os.path.join(class_out, "transforms_test.json"), "w") as f:
                json.dump(new_test_json, f, indent=2)

            print(f"Done pose split for class {cl}.")
            continue

        # --------------------------------------------
        # 3. Normal 3DGS mode: rewrite train paths and generate test poses
        # --------------------------------------------
        train_mats = []

        for frame in train_trans["frames"]:
            cur_idx = int(frame["file_path"].split("/")[-1].split(".")[0])
            frame["file_path"] = f"./train/train_{cur_idx:03d}"
            train_mats.append(np.array(frame["transform_matrix"])[None, ...])

        with open(os.path.join(class_out, "transforms_train.json"), "w") as f:
            json.dump(train_trans, f, indent=2)

        train_mats = np.concatenate(train_mats, axis=0)
        translations = train_mats[:, :3, 3]
        rotations = train_mats[:, :3, :3]
        mean_point = np.mean(translations, axis=0)

        # Generate test positions
        test_t1 = generate_samples(k_augments, translations, distance_factor=0.8)
        test_t2 = generate_samples(k_augments, translations, distance_factor=1.2)
        test_translations = np.concatenate((test_t1, test_t2), axis=0)

        indices = pairwise_distances(X=test_translations, Y=translations).argmin(axis=1)
        closest_trans = translations[indices]
        closest_rots = rotations[indices]

        test_json = {"camera_angle_x": camera_angle_x, "frames": []}
        test_poses = np.zeros((test_translations.shape[0], 4, 4))

        # Placeholder white test image
        empty_img = Image.fromarray(
            np.ones((test_image_size, test_image_size), dtype=np.uint8) * 255
        )

        for i in range(test_translations.shape[0]):
            cur_vec = test_translations[i]
            ref_vec = closest_trans[i] - mean_point

            R_delta = rotation_matrix_from_vectors(ref_vec, cur_vec)
            R_final = R_delta @ closest_rots[i]

            test_poses[i, 3, 3] = 1.0
            test_poses[i, :3, 3] = cur_vec
            test_poses[i, :3, :3] = R_final

            test_json["frames"].append(
                {
                    "file_path": f"./test/test_{i:03d}",
                    "transform_matrix": test_poses[i].tolist(),
                }
            )

            empty_img.save(os.path.join(new_test_dir, f"test_{i:03d}.png"))

        with open(os.path.join(class_out, "transforms_test.json"), "w") as f:
            json.dump(test_json, f, indent=2)

        print(f"Done with class {cl}.\n")


if __name__ == "__main__":
    main()
