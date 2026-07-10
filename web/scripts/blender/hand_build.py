# hand_build.py — baut die ayaka-hand als geriggtes Subdiv-Mesh und exportiert .glb
# Aufruf: blender --background --python hand_build.py -- --out ayaka-hand.glb --renders ./
#
# Raum-Konvention: Blender Z-up, Finger entlang +Y, Greifseite +Z.
# glTF-Export (Y-up): Blender X→three X, Z→three Y, +Y→three −Z
# → three: Finger −Z, Greifseite +Y, Daumen −X — exakt wie das alte prozedurale Rig.
import bpy
import bmesh
import math
import sys
from mathutils import Euler, Vector

ARGS = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []


def arg(name, default):
    return ARGS[ARGS.index(name) + 1] if name in ARGS else default


OUT = arg('--out', 'ayaka-hand.glb')
RDIR = arg('--renders', None)

# Maße gespiegelt an FINGER_SPECS (threeHand.ts) — Weltmaßstab identisch zum alten Rig.
# root = (x, y_blender) an der Palm-Vorderkante; lens proximal→distal.
FINGERS = {
    'index':  dict(root=(-0.24, 0.42), lens=[0.34, 0.27, 0.20], r=0.088),
    'middle': dict(root=(-0.02, 0.46), lens=[0.39, 0.30, 0.22], r=0.092),
    'ring':   dict(root=(0.19, 0.43),  lens=[0.36, 0.28, 0.21], r=0.085),
    'pinky':  dict(root=(0.37, 0.35),  lens=[0.30, 0.24, 0.17], r=0.070),
}
# Daumen: Richtungsvektor statt Euler-Konvertierung — zeigt auswärts-vorn, leicht zur Greifseite.
THUMB = dict(root=Vector((-0.36, -0.04, -0.02)), dir=Vector((-0.62, 0.76, 0.18)).normalized(),
             lens=[0.34, 0.28], r=0.105)
TAPER = 0.86      # Radius-Abnahme pro Glied
KNUCKLE = 1.16    # Ring-Aufdickung am Gelenk (sichtbarer Gelenk-Rhythmus unterm Subdiv)
MIDSLIM = 0.90    # Ring-Verschlankung in Glied-Mitte
PALM = dict(w=0.86, d=0.92, h=0.24)
RADIAL = 8        # Cage-Auflösung radial (Subdiv glättet)


def tube(bm, rings, radial=RADIAL, mat_index=0):
    """rings: Liste (center: Vector, radius, up: Vector). Loop-Strang + Kappen; Ringebene ⊥ Strangrichtung."""
    loops = []
    for i, (c, r) in enumerate(rings):
        nxt = rings[min(i + 1, len(rings) - 1)][0]
        prv = rings[max(i - 1, 0)][0]
        axis = (nxt - prv)
        axis = axis.normalized() if axis.length > 1e-8 else Vector((0, 1, 0))
        # Orthonormalbasis um die Strangachse
        ref = Vector((0, 0, 1)) if abs(axis.z) < 0.9 else Vector((1, 0, 0))
        u = axis.cross(ref).normalized()
        v = axis.cross(u).normalized()
        vs = []
        for k in range(radial):
            a = 2 * math.pi * k / radial
            vs.append(bm.verts.new(c + u * (math.cos(a) * r) + v * (math.sin(a) * r)))
        loops.append(vs)
    faces = []
    for a, b in zip(loops, loops[1:]):
        for i in range(radial):
            faces.append(bm.faces.new((a[i], a[(i + 1) % radial], b[(i + 1) % radial], b[i])))
    for loop, flip in ((loops[0], True), (loops[-1], False)):
        centro = bm.verts.new(sum((vv.co for vv in loop), Vector()) / radial)
        for i in range(radial):
            tri = (loop[i], loop[(i + 1) % radial], centro) if flip else (loop[(i + 1) % radial], loop[i], centro)
            faces.append(bm.faces.new(tri))
    for f in faces:
        f.material_index = mat_index
    return loops


def finger_dir(spec):
    """Fingerrichtung: leicht zur Handmitte konvergierend (statt parallel-gespreizt) — wie bei echten Händen."""
    return Vector((-spec['root'][0] * 0.10, 1.0, 0.0)).normalized()


def finger_rings(lens, r0, origin, direction):
    """Ring-Profil entlang `direction`: Wurzel tief in der Palm verankert, pro Glied [Mitte schlank, Gelenk dick], runde Kuppe."""
    rings = [(origin - direction * 0.14, r0 * 1.08)]
    dist, r = 0.0, r0
    for li, ln in enumerate(lens):
        rings.append((origin + direction * (dist + 0.5 * ln), r * MIDSLIM))
        dist += ln
        r_next = r * TAPER
        big = KNUCKLE if li < len(lens) - 1 else 0.92   # letztes „Gelenk" = Kuppenansatz
        rings.append((origin + direction * dist, r_next * big))
        r = r_next
    rings.append((origin + direction * (dist + 0.055), r * 0.55))  # Kuppen-Abschluss (Subdiv rundet)
    return rings


def build_hand_mesh():
    bm = bmesh.new()
    # Palm: Cage-Kiste mit Bevel — Subdiv macht daraus eine gerundete, leicht gewölbte Handfläche.
    ret = bmesh.ops.create_cube(bm, size=1.0)
    palm_verts = ret['verts']
    bmesh.ops.scale(bm, verts=palm_verts, vec=(PALM['w'], PALM['d'], PALM['h']))
    bmesh.ops.translate(bm, verts=palm_verts, vec=(0, -0.02, 0))
    for v in palm_verts:
        if v.co.y < 0:
            v.co.x *= 0.80                       # zum Handgelenk hin schmaler (Trapez-Silhouette)
            if v.co.z > 0:
                v.co.z += 0.05                   # Handrücken zum Gelenk hin gewölbt
    bmesh.ops.bevel(bm, geom=list(bm.edges), offset=0.075, segments=3, profile=0.72, affect='EDGES')
    for name, spec in FINGERS.items():
        origin = Vector((spec['root'][0], spec['root'][1], 0.0))
        tube(bm, finger_rings(spec['lens'], spec['r'], origin, finger_dir(spec)))
    tube(bm, finger_rings(THUMB['lens'], THUMB['r'], THUMB['root'], THUMB['dir']))
    # Daumenballen: Wulst zwischen Palm-Kante und Daumenwurzel, verschmilzt unterm Subdiv zur Silhouette
    tube(bm, [(THUMB['root'] + Vector((0.16, -0.10, 0.0)), 0.15),
              (THUMB['root'] + THUMB['dir'] * 0.06, 0.135)])
    # Handgelenk-Sockel nach hinten (−Y): schlank-konisch, Ende eingezogen (rundet unterm Subdiv statt Flachdeckel)
    tube(bm, [(Vector((0, -0.40, 0.01)), 0.225), (Vector((0, -0.53, 0.01)), 0.25),
              (Vector((0, -0.59, 0.01)), 0.16)], radial=12, mat_index=1)
    mesh = bpy.data.meshes.new('ayakaHand')
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new('ayakaHand', mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def make_mat(name, rgba, rough, metal):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes['Principled BSDF']
    p.inputs['Base Color'].default_value = rgba
    p.inputs['Roughness'].default_value = rough
    p.inputs['Metallic'].default_value = metal
    return m


def add_materials(obj):
    obj.data.materials.append(make_mat('shell', (0.56, 0.71, 0.79, 1), 0.5, 0.05))   # slot 0
    obj.data.materials.append(make_mat('joint', (0.24, 0.28, 0.32, 1), 0.45, 0.3))   # slot 1
    obj.data.materials.append(make_mat('lens', (0.10, 0.35, 0.40, 1), 0.15, 0.6))    # slot 2


def add_camera_ring():
    """Wrist-Cam auf der Greifseite nahe Handgelenk — sie schaut dahin, wo gegriffen wird (Produkt-Story)."""
    z = PALM['h'] * 0.5 + 0.03
    bpy.ops.mesh.primitive_torus_add(major_radius=0.095, minor_radius=0.028,
                                     location=(0, -0.28, z), major_segments=24, minor_segments=10)
    ring = bpy.context.object
    bpy.ops.mesh.primitive_cylinder_add(radius=0.062, depth=0.05, vertices=20, location=(0, -0.28, z + 0.006))
    lens = bpy.context.object
    return ring, lens


def assign_extra_materials(obj, ring_name, lens_name):
    """Nach dem Join: Ring→joint, Linse→lens (Slots 1/2) über gespeicherte Facezahl-Bereiche geht nicht —
    stattdessen wurden Ring/Linse VOR dem Join mit eigenem Slot versehen; Blender mappt beim Join um."""
    del obj, ring_name, lens_name  # Zuordnung passiert beim Join automatisch über Objekt-Materialien


def build_armature():
    bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
    arm = bpy.context.object
    eb = arm.data.edit_bones
    rootb = eb[0]
    rootb.name = 'root'
    rootb.head = Vector((0, -0.30, 0))
    rootb.tail = Vector((0, 0.05, 0))

    def chain(name, origin, lens, direction):
        prev, head = rootb, origin.copy()
        for i, ln in enumerate(lens):
            b = eb.new(f'{name}_{i}')
            b.head = head
            b.tail = head + direction * ln
            b.roll = 0.0
            b.parent = prev
            b.use_connect = i > 0
            prev, head = b, b.tail

    for name, spec in FINGERS.items():
        chain(name, Vector((spec['root'][0], spec['root'][1], 0.0)), spec['lens'], finger_dir(spec))
    chain('thumb', THUMB['root'], THUMB['lens'], THUMB['dir'])
    bpy.ops.object.mode_set(mode='OBJECT')
    return arm


def render_poses(arm):
    """Workbench-Renders (Studio-Licht): offen / power / pinch — Formkontrolle ohne Browser.
    Winkel-Mathe identisch zu applyToJoints (threeHand.ts): mult 1.15 (Daumen 0.9), Root ×0.8."""
    scn = bpy.context.scene
    scn.render.engine = 'BLENDER_WORKBENCH'
    scn.display.shading.light = 'STUDIO'
    scn.display.shading.color_type = 'MATERIAL'
    scn.render.resolution_x = 900
    scn.render.resolution_y = 900
    bpy.ops.object.camera_add(location=(1.7, -1.6, 1.9))
    cam = bpy.context.object
    cam.rotation_euler = (Vector((0, 0.15, 0)) - cam.location).to_track_quat('-Z', 'Y').to_euler()
    scn.camera = cam
    presets = {
        'open':  dict(thumb=0.05, index=0.05, middle=0.05, ring=0.05, pinky=0.05),
        'power': dict(thumb=0.85, index=0.95, middle=0.95, ring=0.95, pinky=0.95),
        'pinch': dict(thumb=0.70, index=0.75, middle=0.80, ring=0.90, pinky=0.90),
    }
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='POSE')
    for pname, curls in presets.items():
        for pb in arm.pose.bones:
            if '_' not in pb.name:
                continue
            f, si = pb.name.rsplit('_', 1)
            si = int(si)
            ang = (0.9 if f == 'thumb' else 1.15) * curls.get(f, 0.0) * (0.8 if si == 0 else 1.0)
            pb.rotation_mode = 'XYZ'
            pb.rotation_euler = (ang, 0, 0)   # lokale +X-Flex: krümmt Richtung Greifseite (+Z) — in Renders verifizieren
        scn.render.filepath = f'{RDIR}/render_{pname}.png'
        bpy.ops.render.render(write_still=True)
    for pb in arm.pose.bones:
        pb.rotation_euler = (0, 0, 0)
    bpy.ops.object.mode_set(mode='OBJECT')


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    obj = build_hand_mesh()
    add_materials(obj)
    ring, lens = add_camera_ring()
    ring.data.materials.append(bpy.data.materials['joint'])
    lens.data.materials.append(bpy.data.materials['lens'])
    bpy.ops.object.select_all(action='DESELECT')
    ring.select_set(True)
    lens.select_set(True)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.join()
    sub = obj.modifiers.new('subsurf', 'SUBSURF')
    sub.levels = 2
    sub.render_levels = 2
    bpy.ops.object.shade_smooth()
    arm = build_armature()
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    if RDIR:
        render_poses(arm)
    bpy.ops.export_scene.gltf(filepath=OUT, export_format='GLB', export_apply=True,
                              export_animations=False, export_skins=True, export_yup=True)
    print('EXPORTED', OUT)


main()
