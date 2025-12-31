# generate_lego_like.py
import bpy
import json
import math
import random
import argparse
import sys
from mathutils import Vector, Matrix, Euler

FOV_DEG = 28.0

# ----------------------------
# Utils: CLI args after "--"
# ----------------------------
def parse_args():
    argv = sys.argv
    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1 :]
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--num_objs", type=int, default=10)
    p.add_argument("--views", type=int, default=32)
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)

# ----------------------------
# Scene reset / basic setup
# ----------------------------
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"  # or "BLENDER_EEVEE"
    scene.cycles.samples = 32       # keep small for speed
    scene.render.film_transparent = False

def ensure_dir(path: str):
    import os
    os.makedirs(path, exist_ok=True)

def delete_object(obj):
    bpy.data.objects.remove(obj, do_unlink=True)

# ----------------------------
# Simple material (plastic-ish)
# ----------------------------
def make_plastic_material(name="mat_plastic"):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        raise RuntimeError('Principled BSDF node not found in material node tree.')

    # Roughness is stable across versions
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.35

    # Specular naming differs across Blender versions
    if "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.25
    elif "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.25
    else:
        # As a fallback, you can tweak IOR (often exists) to influence specular response
        if "IOR" in bsdf.inputs:
            bsdf.inputs["IOR"].default_value = 1.45  # plastic-ish IOR

    return mat

# ----------------------------
# Geometry builders
# ----------------------------
def add_cube(name, size_xyz, location=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (size_xyz[0] / 2, size_xyz[1] / 2, size_xyz[2] / 2)  # cube is size=1
    bpy.ops.object.transform_apply(scale=True)
    return obj

def add_cylinder(name, radius, depth, location=(0, 0, 0)):
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth, location=location)
    obj = bpy.context.active_object
    obj.name = name
    return obj

def boolean_diff(target, cutter):
    mod = target.modifiers.new(name="bool_diff", type="BOOLEAN")
    mod.operation = "DIFFERENCE"
    mod.object = cutter
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.modifier_apply(modifier=mod.name)

def boolean_union(target, other):
    mod = target.modifiers.new(name="bool_union", type="BOOLEAN")
    mod.operation = "UNION"
    mod.object = other
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.modifier_apply(modifier=mod.name)

def add_bevel(obj, width=0.002, segments=2):
    mod = obj.modifiers.new(name="bevel", type="BEVEL")
    mod.width = width
    mod.segments = segments
    mod.limit_method = "NONE"
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)

# ----------------------------
# LEGO-like block generator
# ----------------------------
def make_block(
    n_studs_x: int,
    n_studs_y: int,
    stud_radius: float,
    stud_height: float,
    base_height: float,
    hole: bool,
    hole_radius: float,
    hole_depth: float,
):
    # Units: meters (Blender default)
    stud_pitch = 0.008   # 8mm LEGO pitch (approx)
    wall = 0.0015        # 1.5mm thickness-ish
    base_w = n_studs_x * stud_pitch
    base_d = n_studs_y * stud_pitch
    base_h = base_height

    # Base block
    base = add_cube(
        "base",
        (base_w, base_d, base_h),
        location=(0, 0, base_h / 2),
    )

    # Studs on top
    studs = []
    x0 = -base_w / 2 + stud_pitch / 2
    y0 = -base_d / 2 + stud_pitch / 2
    top_z = base_h + stud_height / 2

    for ix in range(n_studs_x):
        for iy in range(n_studs_y):
            cx = x0 + ix * stud_pitch
            cy = y0 + iy * stud_pitch
            c = add_cylinder(
                f"stud_{ix}_{iy}",
                radius=stud_radius,
                depth=stud_height,
                location=(cx, cy, top_z),
            )
            studs.append(c)

    # Union all studs into base
    for s in studs:
        boolean_union(base, s)
        delete_object(s)

    # Optional central hole (difference)
    if hole:
        hole_cutter = add_cylinder(
            "hole_cutter",
            radius=hole_radius,
            depth=hole_depth,
            location=(0, 0, base_h / 2),
        )
        boolean_diff(base, hole_cutter)
        delete_object(hole_cutter)

    # Small bevel to avoid razor edges
    add_bevel(base, width=0.0008, segments=2)

    return base

# ----------------------------
# Camera + light + render
# ----------------------------
def setup_camera(res, fov_deg=50.0):
    scene = bpy.context.scene
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.resolution_percentage = 100

    cam_data.lens_unit = "FOV"
    cam_data.angle = math.radians(fov_deg)
    return cam

def setup_light():
    scene = bpy.context.scene

    # Key light
    light_data = bpy.data.lights.new(name="KeyLight", type="AREA")
    light_data.energy = 1500
    light = bpy.data.objects.new(name="KeyLight", object_data=light_data)
    light.location = (0.3, -0.3, 0.5)
    light.rotation_euler = Euler((math.radians(60), 0, math.radians(45)), "XYZ")
    scene.collection.objects.link(light)

    # Fill light
    light2_data = bpy.data.lights.new(name="FillLight", type="AREA")
    light2_data.energy = 800
    light2 = bpy.data.objects.new(name="FillLight", object_data=light2_data)
    light2.location = (-0.3, 0.3, 0.4)
    light2.rotation_euler = Euler((math.radians(60), 0, math.radians(-135)), "XYZ")
    scene.collection.objects.link(light2)

def setup_background():
    scene = bpy.context.scene

    # If no world exists, create one
    if scene.world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    else:
        world = scene.world

    world.use_nodes = True
    nt = world.node_tree
    nodes = nt.nodes

    # Get or create Background node
    bg = nodes.get("Background")
    if bg is None:
        bg = nodes.new(type="ShaderNodeBackground")

    # Set grey background
    bg.inputs[0].default_value = (0.8, 0.8, 0.8, 1.0)

    # Ensure Background is connected to World Output
    out = nodes.get("World Output")
    if out is None:
        out = nodes.new(type="ShaderNodeOutputWorld")

    nt.links.new(bg.outputs[0], out.inputs[0])

def look_at(obj_camera, target: Vector):
    direction = target - obj_camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    obj_camera.rotation_euler = rot_quat.to_euler()

def get_camera_intrinsics(cam, scene):
    # Approx pinhole intrinsics in pixel units
    # Reference: Blender camera model
    render = scene.render
    w = render.resolution_x * render.resolution_percentage / 100.0
    h = render.resolution_y * render.resolution_percentage / 100.0
    cam_data = cam.data

    # FOV horizontal depends on sensor fit; we keep it simple and use angle
    f = 0.5 * w / math.tan(0.5 * cam_data.angle)
    cx = w / 2.0
    cy = h / 2.0
    return {"fx": f, "fy": f, "cx": cx, "cy": cy, "w": w, "h": h}

def get_camera_extrinsic(cam):
    # World-to-camera (OpenCV-style) often used:
    # Here we export Blender camera matrix_world inverse (camera_from_world)
    M = cam.matrix_world.inverted()
    return [list(row) for row in M]

def render_views(out_dir, cam, obj, views):
    import os
    scene = bpy.context.scene
    images_dir = os.path.join(out_dir, "images")
    ensure_dir(images_dir)

    # Normalize object position
    obj.location = (0, 0, 0)

    # Estimate radius (simple bounding box)
    bbox = [Vector(v) for v in obj.bound_box]
    max_r = max(v.length for v in bbox)
    dist = max(0.12, 3.0 * max_r)  # keep object in view

    target = Vector((0, 0, max_r * 0.4))

    K = get_camera_intrinsics(cam, scene)
    frames = []

    for i in range(views):
        theta = 2.0 * math.pi * (i / views)
        cam.location = (dist * math.cos(theta), dist * math.sin(theta), dist * 0.6)
        look_at(cam, target)

        scene.render.filepath = os.path.join(images_dir, f"{i:03d}.png")
        bpy.ops.render.render(write_still=True)

        frames.append({
            "file": f"images/{i:03d}.png",
            "c2w_blender": [list(row) for row in cam.matrix_world],
            "w2c": get_camera_extrinsic(cam),
        })

    poses = {"intrinsics": K, "frames": frames}
    with open(os.path.join(out_dir, "poses.json"), "w") as f:
        json.dump(poses, f, indent=2)

# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    random.seed(args.seed)

    reset_scene()
    setup_background()
    cam = setup_camera(args.res, fov_deg=FOV_DEG)
    setup_light()

    mat = make_plastic_material()

    for idx in range(args.num_objs):
        # Random parameters (your pseudo code equivalent)
        n_studs_x = random.randint(2, 8)
        n_studs_y = random.randint(1, 6)
        stud_height = random.uniform(0.003, 0.006)   # 3-6mm
        stud_radius = random.uniform(0.0022, 0.0027) # 2.2-2.7mm
        base_height = random.uniform(0.006, 0.012)   # 6-12mm

        hole = random.choice([True, False])
        hole_radius = random.uniform(0.002, 0.004)
        hole_depth = base_height * 1.2

        obj = make_block(
            n_studs_x=n_studs_x,
            n_studs_y=n_studs_y,
            stud_radius=stud_radius,
            stud_height=stud_height,
            base_height=base_height,
            hole=hole,
            hole_radius=hole_radius,
            hole_depth=hole_depth,
        )

        # Random color per object (helps diversity)
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        # tweak base color
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = (
            random.uniform(0.05, 0.95),
            random.uniform(0.05, 0.95),
            random.uniform(0.05, 0.95),
            1.0,
        )

        out_obj_dir = f"{args.out}/obj_{idx:04d}"
        ensure_dir(out_obj_dir)

        render_views(out_obj_dir, cam, obj, args.views)

        # Clean up object for next iteration
        delete_object(obj)

    print(f"[DONE] Wrote dataset to: {args.out}")

if __name__ == "__main__":
    main()
