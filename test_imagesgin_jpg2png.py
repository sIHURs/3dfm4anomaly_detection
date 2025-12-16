import struct
import os

def read_next_bytes(fid, num_bytes, fmt, endian="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + fmt, data)

def write_next_bytes(fid, data, fmt, endian="<"):
    fid.write(struct.pack(endian + fmt, *data))

def read_images_bin(path):
    images = {}
    with open(path, "rb") as f:
        num_images = read_next_bytes(f, 8, "Q")[0]

        for _ in range(num_images):
            img_id = read_next_bytes(f, 4, "i")[0]
            qw, qx, qy, qz = read_next_bytes(f, 32, "dddd")
            tx, ty, tz     = read_next_bytes(f, 24, "ddd")
            cam_id         = read_next_bytes(f, 4, "i")[0]

            # read name (null terminated)
            name_bytes = b""
            c = f.read(1)
            while c != b"\x00":
                name_bytes += c
                c = f.read(1)
            name = name_bytes.decode("utf-8")

            # read number of 2D points
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
    with open(out_path, "wb") as f:
        write_next_bytes(f, [len(images)], "Q")

        for img_id, data in images.items():
            write_next_bytes(f, [img_id], "i")
            write_next_bytes(f, data["Q"], "dddd")
            write_next_bytes(f, data["T"], "ddd")
            write_next_bytes(f, [data["cam_id"]], "i")

            # write filename
            f.write(data["name"].encode("utf-8") + b"\x00")

            write_next_bytes(f, [len(data["points"])], "Q")
            for x, y, pid in data["points"]:
                write_next_bytes(f, [x, y, pid], "ddq")


def overwrite_jpg_to_png(images_bin_path):
    print(f"Loading {images_bin_path} ...")
    images = read_images_bin(images_bin_path)

    for img_id, data in images.items():
        name = data["name"].lower()

        if name.endswith(".jpg"):
            new_name = data["name"][:-4] + ".png"
            print(f"Rename: {data['name']} → {new_name}")
            data["name"] = new_name

        elif name.endswith(".jpeg"):
            new_name = data["name"][:-5] + ".png"
            print(f"Rename: {data['name']} → {new_name}")
            data["name"] = new_name

    print("Writing back to images.bin (overwrite!) ...")
    write_images_bin(images, images_bin_path)
    print("Done.")


# ---------------------
# Use:
# ---------------------
overwrite_jpg_to_png("scripts/demo_PIAD_Sim_vggt_3dgs/motor/sparse/0/images.bin")
