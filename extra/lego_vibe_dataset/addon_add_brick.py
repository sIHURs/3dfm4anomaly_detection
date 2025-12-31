bl_info = {
    "name": "Lego Object",
    "author": "Martin Ellis",
    "version": (1, 0),
    "blender": (2, 75, 0),
    "location": "View3D > Add > Mesh > l brick",
    "description": "Adds a new Mesh Object",
    "warning": "",
    "wiki_url": "",
    "category": "Add Mesh",
    }


import bpy
from bpy.types import Operator
from bpy.props import FloatVectorProperty
from bpy_extras.object_utils import AddObjectHelper, object_data_add
from mathutils import Vector
import math

def add_object(self, context):
    
    width_per_stud = self.width_of_1x1 
    
    plate_height = width_per_stud*1.2/3
    layers_to_do = self.plates_high+1
    stud_protrusion = 1.7/7.8 * width_per_stud
    stud_rad = 2.4/7.8 * width_per_stud
    stud_segs = self.verts_per_circle
    
    if self.plate_verts == False:
        layers_to_do = 2
        plate_height = plate_height * self.plates_high
        
    
    #create off-sets to ensure brick is cenered around origin
    offs_x = ((self.studs_x)*width_per_stud)/-2
    offs_y = ((self.studs_y)*width_per_stud)/-2
    
    #declare empty lists to append to...
    verts = []
    faces = []
    #I have no edges whcih are not part of a face, keep empty list.
    edges = []
    
    verts_per_layer = (self.studs_y + self.studs_x) *2
    
    #add verts
    for lyr_n in range(0, layers_to_do):
        #walk the perimiter
        for stu_y in range(0, self.studs_y):
            verts.append(Vector((
            (offs_x),
            (offs_y+stu_y*width_per_stud), 
            (lyr_n*plate_height))))
            
        for stu_x in range(0, self.studs_x):
            verts.append(Vector((
            (offs_x+stu_x*width_per_stud), 
            (offs_y+width_per_stud*self.studs_y), 
            (lyr_n*plate_height))))
                    
        for stu_y in reversed(range(0, self.studs_y)):
            verts.append(Vector((
            (offs_x+width_per_stud*self.studs_x), 
            (offs_y+(stu_y+1)*width_per_stud), 
            (lyr_n*plate_height))))
            
        for stu_x in reversed(range(0, self.studs_x)):
            verts.append(Vector((
            (offs_x+(stu_x+1)*width_per_stud), 
            #(offs_y+width_per_stud*self.studs_y), 
            (offs_y), 
            (lyr_n*plate_height))))
                
               
    #add faces
    #if this is the 2nd or higher layer, add faces to layer below
    for lyr_n in range(1, layers_to_do):
        for v_n in range(0, verts_per_layer):
            tl = lyr_n*verts_per_layer+v_n
            tr = tl+1
            bl = (lyr_n-1)*verts_per_layer+v_n
            br = bl+1
            
            #if this is the end of the layer, need to adjust to wrap round
            if (v_n+1) % verts_per_layer == 0:
                tr = tr - verts_per_layer
                br = br - verts_per_layer 
            faces.append([tl, tr, br, bl])    
        

    #add top face
    faces.append(range(len(verts)-verts_per_layer, len(verts)))

    verts_so_far = len(verts)


    #draw studs
    if (self.tile == False):
        for stu_x in range(0, self.studs_x):
        #for stu_x in range(0, 1):
            for stu_y in range(0, self.studs_y):
            #for stu_y in range(0, 1):
                c_x = (offs_x+stu_x*width_per_stud)+(width_per_stud/2)
                c_y = (offs_y+stu_y*width_per_stud)+(width_per_stud/2)
                    
                #verts, bottom of cylinder
                for seg in range(0, stud_segs):
                    ang = float(seg)/float(stud_segs)*math.pi*2
                    verts.append(Vector((
                    c_x + math.sin(ang)*stud_rad,
                    c_y + math.cos(ang)*stud_rad,
                    plate_height*(layers_to_do-1)+self.stud_base_offset)))
                #verts, top of cylinder
                for seg in range(0, stud_segs):
                    ang = float(seg)/float(stud_segs)*math.pi*2
                    verts.append(Vector((
                    c_x + math.sin(ang)*stud_rad,
                    c_y + math.cos(ang)*stud_rad,
                    plate_height*(layers_to_do-1)+stud_protrusion)))
                
                #add faces
                verts_so_far = len(verts)
                for v_n in range(0, stud_segs):
                    tl = verts_so_far - stud_segs+v_n
                    tr = tl+1
                    bl = tl - stud_segs
                    br = bl+1
            
                    #if this is the end of the layer, need to adjust to wrap round
                    if (v_n+1) % stud_segs == 0:
                        tr = tr - stud_segs
                        br = br - stud_segs 
                    faces.append([tl, tr, br, bl])  
                
                #add stud top face
                faces.append(range(len(verts)-stud_segs, len(verts)))
                
    mesh = bpy.data.meshes.new(name="New Object Mesh")
    mesh.from_pydata(verts, edges, faces)
    object_data_add(context, mesh, operator=self)
        

class OBJECT_OT_add_object(Operator, AddObjectHelper):
    """Create a new Mesh Object"""
    bl_idname = "mesh.add_object"
    bl_label = "Add Mesh Object"
    bl_options = {'REGISTER', 'UNDO'}


    studs_x = bpy.props.IntProperty(default=3)
    studs_y = bpy.props.IntProperty(default=2)
    plates_high = bpy.props.IntProperty(default=3)
    plate_verts = bpy.props.BoolProperty(default=False)
    tile = bpy.props.BoolProperty(default=False)
    
    width_of_1x1 = bpy.props.FloatProperty(default=1.0)
    verts_per_circle = bpy.props.IntProperty(default=32)
    stud_base_offset= bpy.props.FloatProperty(default=0.01)

    def execute(self, context):

        add_object(self, context)

        return {'FINISHED'}


# Registration

def add_object_button(self, context):
    self.layout.operator(
        OBJECT_OT_add_object.bl_idname,
        text="Add Lego",
        icon='PLUGIN')


# This allows you to right click on a button and link to the manual
def add_object_manual_map():
    url_manual_prefix = "https://docs.blender.org/manual/en/dev/"
    url_manual_mapping = (
        ("bpy.ops.mesh.add_object", "editors/3dview/object"),
        )
    return url_manual_prefix, url_manual_mapping


def register():
    bpy.utils.register_class(OBJECT_OT_add_object)
    bpy.utils.register_manual_map(add_object_manual_map)
    bpy.types.INFO_MT_mesh_add.append(add_object_button)


def unregister():
    bpy.utils.unregister_class(OBJECT_OT_add_object)
    bpy.utils.unregister_manual_map(add_object_manual_map)
    bpy.types.INFO_MT_mesh_add.remove(add_object_button)


if __name__ == "__main__":
    register()