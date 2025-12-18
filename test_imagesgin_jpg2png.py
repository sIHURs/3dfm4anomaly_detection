import struct
import argparse


def read_next_bytes(fid, num_bytes, fmt, endian="<"):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian + fmt, data)


def write_next_bytes(fid, data, fmt, endian="<"):
    """Pack and write the given data to a binary file."""
    fid.write(struct.pack(endian + fmt, *data))


def read_images_bin(path):
    """
    Read COLMAP images.bin into a Python dict.

    Returns:
        images: dict keyed by image_id, each entry contains:
            Q      : (qw, qx, qy, qz)
            T      : (tx, ty, tz)
            cam_id : camera_id
            name   : image filename
            points : list of (x, y, point3D_id)
    """
    images = {}
    with open(path, "rb") as f:
        num_images = read_next_bytes(f, 8, "Q")[0]

        for _ in range(num_images):
            img_id = read_next_bytes(f, 4, "i")[0]
            qw, qx, qy, qz = read_next_bytes(f, 32, "dddd")
            tx, ty, tz = read_next_bytes(f, 24, "ddd")
            cam_id = read_next_bytes(f, 4, "i")[0]

            # Read image name (null-terminated string)
            name_bytes = b""
            c = f.read(1)
            while c != b"\x00":
                name_bytes += c
                c = f.read(1)
            name = name_bytes.decode("utf-8")

            # Read number of 2D points
            n_points = read_next_bytes(f, 8, "Q")[0]

            pts = []
            for _ in range(n_points):
                x, y, pid = read_next_bytes(f, 24, "ddq")
                pts.append((x, y, pid))

            images[img_id] = {
                "Q": (qw, qx, qy, qz),
                "T": (tx, ty, tz),
                "cam_id": cam_id,
                "name": name,
                "points": pts,
            }

    return images


def write_images_bin(images, out_path):
    """Write a Python dict back to COLMAP images.bin format."""
    with open(out_path, "wb") as f:
        write_next_bytes(f, [len(images)], "Q")

        for img_id, data in images.items():
            write_next_bytes(f, [img_id], "i")
            write_next_bytes(f, data["Q"], "dddd")
            write_next_bytes(f, data["T"], "ddd")
            write_next_bytes(f, [data["cam_id"]], "i")

            # Write image name as null-terminated string
            f.write(data["name"].encode("utf-8") + b"\x00")

            # Write 2D points
            write_next_bytes(f, [len(data["points"])], "Q")
            for x, y, pid in data["points"]:
                write_next_bytes(f, [x, y, pid], "ddq")


def overwrite_jpg_to_png(images_bin_path):
    """
    Overwrite COLMAP images.bin in-place by changing image filenames:
      *.jpg / *.jpeg  ->  *.png
    """
    print(f"Loading: {images_bin_path}")
    images = read_images_bin(images_bin_path)

    renamed = 0
    for _, data in images.items():
        name_lower = data["name"].lower()

        if name_lower.endswith(".jpg"):
            new_name = data["name"][:-4] + ".png"
            print(f"Rename: {data['name']} -> {new_name}")
            data["name"] = new_name
            renamed += 1

        elif name_lower.endswith(".jpeg"):
            new_name = data["name"][:-5] + ".png"
            print(f"Rename: {data['name']} -> {new_name}")
            data["name"] = new_name
            renamed += 1

    print(f"Renamed {renamed} file(s).")
    print("Writing back to images.bin (in-place overwrite) ...")
    write_images_bin(images, images_bin_path)
    print("Done.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overwrite COLMAP images.bin filenames: .jpg/.jpeg -> .png"
    )
    parser.add_argument(
        "--images_bin",
        type=str,
        required=True,
        help="Path to COLMAP images.bin (will be overwritten in-place)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    overwrite_jpg_to_png(args.images_bin)

'''
python rename_images_bin_ext.py \
  --images_bin scripts/demo_PIAD_Sim_vggt_3dgs/motor/sparse/0/images.bin
  '''