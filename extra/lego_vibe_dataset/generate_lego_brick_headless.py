import bpy
import math
import json
import argparse
import sys
import os
from mathutils import Vector, Euler

def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True, help="Output directory")
    p.add_argument("--studs_x", type=int, default=4)
    p.add_argument("--studs_y", type=int, default=2)
    p.add_argument("--plates_high", type=int, default=3)
    p.add_argument("--tile", action="store_true", help="No studs on top")
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--render", action="store_true", help="Render preview.png")
    return p.parse_args(argv)

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.use_compositing = False
    scene.use_nodes = False

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def setup_world_flat(color=(1,1,1,1), strength=0.0):
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    w = scene.world
    w.use_nodes = True
    nt = w.node_tree
    bg = nt.nodes.get("Background") or nt.nodes.new(type="ShaderNodeBackground")
    out = nt.nodes.get("World Output") or nt.nodes.new(type="ShaderNodeOutputWorld")
    if "Color" in bg.inputs:
        bg.inputs["Color"].default_value = color
    else:
        bg.inputs[0].default_value = color
    if "Strength" in bg.inputs:
        bg.inputs["Strength"].default_value = strength
    # connect
    for l in list(out.inputs[0].links):
        nt.links.remove(l)
    nt.links.new(bg.outputs[0], out.inputs[0])

def add_light():
    scene = bpy.context.scene
    key_data = bpy.data.lights.new(name="Key", type="AREA")
    key_data.energy = 800
    key = bpy.data.objects.new("Key", key_data)
    key.location = (0.25, -0.35, 0.35)
    key.rotation_euler = Euler((math.radians(65), 0, math.radians(25)), "XYZ")
    scene.collection.objects.link(key)

def setup_camera(res, fov_deg=28.0):
    scene = bpy.context.scene
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    cam_data.lens_unit = "FOV"
    cam_data.angle = math.radians(fov_deg)
    return cam

def look_at(cam, target: Vector):
    direction = target - cam.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam.rotation_euler = rot_quat.to_euler()

def generate_lego_brick(
    studs_x=3,
    studs_y=2,
    plates_high=3,
    width_of_1x1=0.008,   # meters (8mm)
    verts_per_circle=24,
    tile=False,
):
    # LEGO-ish proportions
    plate_height = width_of_1x1 * 1.2 / 3.0          # ~3.2mm if width=8mm
    stud_protrusion = 1.7 / 7.8 * width_of_1x1       # ~1.7mm
    stud_rad = 2.4 / 7.8 * width_of_1x1              # ~2.4mm radius-ish

    layers = plates_high + 1
    if tile:
        layers = 2
        plate_height *= plates_high

    offs_x = -(studs_x * width_of_1x1) / 2.0
    offs_y = -(studs_y * width_of_1x1) / 2.0

    verts, faces = [], []
    verts_per_layer = (studs_y + studs_x) * 2

    # perimeter vertices for each z layer
    for lyr in range(layers):
        z = lyr * plate_height

        for y in range(studs_y):
            verts.append(Vector((offs_x, offs_y + y * width_of_1x1, z)))
        for x in range(studs_x):
            verts.append(Vector((offs_x + x * width_of_1x1, offs_y + studs_y * width_of_1x1, z)))
        for y in reversed(range(studs_y)):
            verts.append(Vector((offs_x + studs_x * width_of_1x1, offs_y + (y + 1) * width_of_1x1, z)))
        for x in reversed(range(studs_x)):
            verts.append(Vector((offs_x + (x + 1) * width_of_1x1, offs_y, z)))

    # side faces between layers
    for lyr in range(1, layers):
        for i in range(verts_per_layer):
            tl = lyr * verts_per_layer + i
            tr = tl + 1 if (i + 1) % verts_per_layer else tl + 1 - verts_per_layer
            bl = (lyr - 1) * verts_per_layer + i
            br = bl + 1 if (i + 1) % verts_per_layer else bl + 1 - verts_per_layer
            faces.append([tl, tr, br, bl])

    # top face (ngon)
    faces.append(list(range(len(verts) - verts_per_layer, len(verts))))

    # studs
    if not tile:
        top_z_base = plate_height * (layers - 1)
        for sx in range(studs_x):
            for sy in range(studs_y):
                cx = offs_x + sx * width_of_1x1 + width_of_1x1 / 2
                cy = offs_y + sy * width_of_1x1 + width_of_1x1 / 2
                zb = top_z_base
                zt = top_z_base + stud_protrusion

                base_idx = len(verts)
                # bottom ring
                for seg in range(verts_per_circle):
                    a = seg / verts_per_circle * math.tau
                    verts.append(Vector((cx + math.sin(a) * stud_rad, cy + math.cos(a) * stud_rad, zb)))
                # top ring
                for seg in range(verts_per_circle):
                    a = seg / verts_per_circle * math.tau
                    verts.append(Vector((cx + math.sin(a) * stud_rad, cy + math.cos(a) * stud_rad, zt)))

                # side quads
                for seg in range(verts_per_circle):
                    i0 = base_idx + seg
                    i1 = base_idx + (seg + 1) % verts_per_circle
                    i2 = i1 + verts_per_circle
                    i3 = i0 + verts_per_circle
                    faces.append([i0, i1, i2, i3])

                # top cap
                faces.append(list(range(base_idx + verts_per_circle, base_idx + 2 * verts_per_circle)))

    mesh = bpy.data.meshes.new("lego_brick_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new("lego_brick", mesh)
    bpy.context.collection.objects.link(obj)
    return obj

def export_glb(glb_path: str):
    bpy.ops.export_scene.gltf(
        filepath=glb_path,
        export_format="GLB",
        use_selection=True,
    )

def main():
    args = parse_args()
    ensure_dir(args.out)

    reset_scene()
    setup_world_flat(color=(1, 1, 1, 1), strength=0.0)
    add_light()
    cam = setup_camera(args.res, fov_deg=28.0)

    # Generate geometry
    brick = generate_lego_brick(
        studs_x=args.studs_x,
        studs_y=args.studs_y,
        plates_high=args.plates_high,
        tile=args.tile,
    )

    # Frame camera
    bbox = [Vector(v) for v in brick.bound_box]
    max_r = max(v.length for v in bbox)
    dist = 2.2 * max_r
    cam.location = (dist, -dist, 0.15 * max_r)
    look_at(cam, Vector((0, 0, 0.35 * max_r)))

    # Select only brick for export
    bpy.ops.object.select_all(action="DESELECT")
    brick.select_set(True)
    bpy.context.view_layer.objects.active = brick

    glb_path = os.path.join(args.out, f"brick_{args.studs_x}x{args.studs_y}_h{args.plates_high}.glb")
    export_glb(glb_path)
    print(f"[DONE] Exported: {glb_path}")

    # Save a .blend too (optional but handy)
    blend_path = os.path.join(args.out, "scene.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    # Optional render preview
    if args.render:
        bpy.context.scene.render.filepath = os.path.join(args.out, "preview.png")
        bpy.ops.render.render(write_still=True)

    print(f"[DONE] Exported: {glb_path}")
    print(f"[DONE] Saved: {blend_path}")
    if args.render:
        print(f"[DONE] Rendered: {os.path.join(args.out, 'preview.png')}")

if __name__ == "__main__":
    main()
