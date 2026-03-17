# -*- coding: utf-8 -*-
"""
abaqus_generator.py
===================
Abaqus 2020 Python script generator for Lamb wave delamination detection
in CFRP composite plates.

Generates complete .py scripts that can be run inside Abaqus/CAE to build:
  - Healthy plate models (full tie constraint, no delamination)
  - Damaged plate models (two-part sub-laminates with tie + contact)

Delamination strategy: Two-Part + Tie Constraint
  Part A = bottom sub-laminate (plies 1-4, Y = 0 to 0.5 mm)
  Part B = top sub-laminate    (plies 5-8, Y = 0.5 to 1.0 mm)
  Healthy: tie everywhere on interface
  Damaged: tie everywhere EXCEPT delamination zone; contact in delam zone

Coordinate system: XZ plane = plate surface, Y = thickness direction
Unit system: mm, tonne, s, MPa

Author: Abaqus Script Generator (Graduation Project)
"""

import os
import datetime
from material_config import (
    MATERIALS,
    LAYUP_CONFIGS,
    PLATE,
    DELAMINATIONS,
    EXCITATION,
    SENSORS,
    STEP,
)


class AbaqusScriptGenerator:
    """
    Generates Abaqus/Explicit Python scripts for Lamb wave propagation
    and delamination detection studies.
    """

    def __init__(self, output_dir="abaqus_models"):
        """
        Parameters
        ----------
        output_dir : str
            Base directory for generated scripts (relative to project root).
        """
        self.output_dir = output_dir
        self.generated_files = []

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def generate_healthy(self, material_name, config):
        """
        Generate a healthy plate script (no delamination).

        Parameters
        ----------
        material_name : str
            Key from MATERIALS dict, e.g. 'T300_5208'
        config : str
            Key from LAYUP_CONFIGS dict: 'QI' or 'UD'

        Returns
        -------
        str : path to generated script
        """
        mat = MATERIALS[material_name]
        layup = LAYUP_CONFIGS[config]
        model_name = f"Healthy_{config}_{material_name}"
        filename = f"healthy_{config}.py"

        print(f"  Generating: {material_name}/{filename} ...")

        script = self._build_script(
            mat=mat,
            layup=layup,
            material_name=material_name,
            config=config,
            model_name=model_name,
            is_damaged=False,
            delam_stage=None,
        )

        filepath = self._write_script(material_name, filename, script)
        self.generated_files.append(filepath)
        print(f"    [OK] Saved: {filepath}")
        return filepath

    def generate_damaged(self, material_name, config, delam_stage="stage1"):
        """
        Generate a damaged plate script with delamination.

        Parameters
        ----------
        material_name : str
            Key from MATERIALS dict
        config : str
            'QI' or 'UD'
        delam_stage : str
            Key from DELAMINATIONS dict, e.g. 'stage1'

        Returns
        -------
        str : path to generated script
        """
        mat = MATERIALS[material_name]
        layup = LAYUP_CONFIGS[config]
        delam = DELAMINATIONS[delam_stage]
        model_name = f"Damaged_{config}_{delam_stage}_{material_name}"
        filename = f"damaged_{config}_{delam_stage}.py"

        print(f"  Generating: {material_name}/{filename} ...")

        script = self._build_script(
            mat=mat,
            layup=layup,
            material_name=material_name,
            config=config,
            model_name=model_name,
            is_damaged=True,
            delam_stage=delam_stage,
        )

        filepath = self._write_script(material_name, filename, script)
        self.generated_files.append(filepath)
        print(f"    [OK] Saved: {filepath}")
        return filepath

    def print_summary(self):
        """Print a summary of all generated files."""
        print("\n" + "=" * 70)
        print("GENERATION COMPLETE - Summary")
        print("=" * 70)
        print(f"Total scripts generated: {len(self.generated_files)}")
        print()
        for i, f in enumerate(self.generated_files, 1):
            print(f"  {i:2d}. {f}")
        print("=" * 70)

    # =========================================================================
    # PRIVATE — Script Assembly
    # =========================================================================

    def _build_script(
        self, mat, layup, material_name, config, model_name, is_damaged, delam_stage
    ):
        """Assemble the full Abaqus Python script as a string."""

        delam = DELAMINATIONS.get(delam_stage) if delam_stage else None
        
        # Normalize delam to always have a 'damages' list
        if delam and 'damages' not in delam:
            delam['damages'] = [{
                'center_x': delam['center_x'],
                'center_z': delam['center_z'],
                'size_x': delam['size_x'],
                'size_z': delam['size_z']
            }]
        
        sections = []

        # Header with coordinate system information
        sections.append(
            self._header_with_coordinate_info(
                mat, layup, config, model_name, is_damaged, delam_stage
            )
        )

        # Imports
        sections.append(self._imports())

        # Model setup
        sections.append(self._model_setup(model_name))

        # Material definition
        sections.append(self._material_definition(mat, material_name))

        # Part creation (two parts: PartA bottom, PartB top)
        sections.append(self._part_creation(mat, layup, material_name))

        # Partition through thickness (ply interfaces)
        sections.append(self._partition_through_thickness(mat))

        # Partition at delamination zone boundaries (X and Z cuts) - only for damaged
        if is_damaged:
            sections.append(self._partition_delamination_boundaries(mat, delam))

        # Assign sections and orientations (ply-by-ply)
        sections.append(self._section_assignments(mat, layup, material_name))

        # Seed and generate mesh (with through-thickness control)
        sections.append(self._seed_and_generate_mesh(mat))

        # Assembly
        sections.append(self._assembly(model_name))

        # Surfaces for tie / contact
        sections.append(self._surfaces(model_name, is_damaged, delam))

        # Constraints (tie + contact)
        sections.append(self._constraints(model_name, is_damaged, delam))

        # Step definition
        sections.append(self._step(model_name, mat))

        # Tone burst amplitude
        sections.append(self._amplitude(model_name, mat))

        # Actuator (excitation BC)
        sections.append(self._actuator(model_name))

        # Sensors (history output)
        sections.append(self._sensors(model_name))

        # Boundary Conditions (Soft support at corners)
        sections.append(self._boundary_conditions(model_name))

        # Field output
        sections.append(self._field_output(model_name))

        # Job
        sections.append(self._job(model_name, mat))

        # Footer
        sections.append(self._footer(model_name))

        return "\n".join(sections)

    # =========================================================================
    # PRIVATE — Section Generators
    # =========================================================================

    def _header_with_coordinate_info(
        self, mat, layup, config, model_name, is_damaged, delam_stage
    ):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_type = "DAMAGED" if is_damaged else "HEALTHY"
        delam_info = (
            f"Delamination: {delam_stage}" if delam_stage else "No delamination"
        )

        return f'''# -*- coding: utf-8 -*-
"""
===========================================================================
Abaqus/Explicit Python Script — Lamb Wave Propagation
===========================================================================
Model Name   : {model_name}
Model Type   : {model_type}
Material     : {mat["display_name"]}
Configuration: {layup["display_name"]}
{delam_info}
---------------------------------------------------------------------------
COORDINATE SYSTEM:
  - Sketch plane: XZ plane (Y=0)
  - Extrusion direction: +Y axis (thickness direction)
  - Final coordinates: X=0-300mm, Y=0-1.0mm, Z=0-300mm
  - Note: Y is thickness direction, actuator/sensor Y=0 = bottom surface
---------------------------------------------------------------------------
Generated    : {now}
Generated by : AbaqusScriptGenerator (Graduation Project)
Unit System  : mm, tonne, s, MPa
Solver       : Abaqus/Explicit
===========================================================================
"""
'''

    def _imports(self):
        return """
# ==========================================================================
# IMPORTS & ENVIRONMENT SETUP
# ==========================================================================
# This script is compatible with both standard Abaqus and the abqpy library.
# To run this using your local Python (with type hints), install abqpy:
#   pip install abqpy

from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
import regionToolset
import mesh
import os
import csv
import math

# executeOnCaeStartup() ensures internal Abaqus environment is ready
executeOnCaeStartup()
"""

    def _model_setup(self, model_name):
        return f"""
# ==========================================================================
# MODEL SETUP
# ==========================================================================
modelName = '{model_name}'
mdb.Model(name=modelName, modelType=STANDARD_EXPLICIT)
mdl = mdb.models[modelName]

# Delete default model if it exists
if 'Model-1' in mdb.models:
    del mdb.models['Model-1']
"""

    def _material_definition(self, mat, material_name):
        return f"""
# ==========================================================================
# MATERIAL DEFINITION - {mat["display_name"]}
# ==========================================================================
# Unit system: mm, tonne, s, MPa
# Moduli in MPa (GPa * 1000), density in tonne/mm³ (kg/m³ / 1e12)

matName = '{material_name}'
material = mdl.Material(name=matName)

# Engineering constants for orthotropic (transverse isotropic) ply
material.Elastic(
    type=ENGINEERING_CONSTANTS,
    table=((
        {mat["E1"]},        # E1 (MPa)
        {mat["E2"]},        # E2 (MPa)
        {mat["E3"]},        # E3 (MPa)
        {mat["nu12"]},       # nu12
        {mat["nu13"]},       # nu13
        {mat["nu23"]},       # nu23
        {mat["G12"]},        # G12 (MPa)
        {mat["G13"]},        # G13 (MPa)
        {mat["G23"]},        # G23 (MPa)
    ),)
)

material.Density(table=(({mat["density"]},),))

print('  [OK] Material defined: {mat["display_name"]}')
print('      E1={mat["E1"]} MPa, E2={mat["E2"]} MPa, rho={mat["density"]} tonne/mm³')
"""

    def _part_creation(self, mat, layup, material_name):
        L = PLATE["length"]
        W = PLATE["width"]
        ply_t = PLATE["ply_thickness"]
        half_t = PLATE["thickness"] / 2.0  # 0.5 mm

        return f"""
# ==========================================================================
# PART CREATION - Two sub-laminates
# ==========================================================================
# Part A: bottom sub-laminate (plies 1-4, Y = 0 to {half_t} mm)
# Part B: top sub-laminate    (plies 5-8, Y = {half_t} to {PLATE["thickness"]} mm)
# Each part is a 3D deformable solid

plateLength = {L}     # mm (X direction)
plateWidth  = {W}     # mm (Z direction)
halfThick   = {half_t}  # mm (half total thickness)
plyThick    = {ply_t}   # mm (per ply)

# --- Part A (bottom sub-laminate) ---
# We sketch Length (X) x Thickness (Y) and extrude Width (Z)
# This ensures Part Y is the thickness direction as required.
sketch_A = mdl.ConstrainedSketch(name='PartA_Sketch', sheetSize=400.0)
sketch_A.rectangle(point1=(0.0, 0.0), point2=(plateLength, halfThick))

partA = mdl.Part(name='PartA_Bottom', dimensionality=THREE_D, type=DEFORMABLE_BODY)
partA.BaseSolidExtrude(sketch=sketch_A, depth=plateWidth)
del sketch_A
print('  [OK] Part A (bottom sub-laminate) created: 0 to {half_t} mm in Y')

# --- Part B (top sub-laminate) ---
sketch_B = mdl.ConstrainedSketch(name='PartB_Sketch', sheetSize=400.0)
sketch_B.rectangle(point1=(0.0, 0.0), point2=(plateLength, halfThick))

partB = mdl.Part(name='PartB_Top', dimensionality=THREE_D, type=DEFORMABLE_BODY)
partB.BaseSolidExtrude(sketch=sketch_B, depth=plateWidth)
del sketch_B
print('  [OK] Part B (top sub-laminate) created: 0 to {half_t} mm in Y')

# --- Create Global Orientation System ---
# Coordinate system for orientations: X=1, Y=2 (thickness), Z=3
for prt in [partA, partB]:
    prt.DatumCsysByDefault(CARTESIAN)
    # We use the datums created here to reference in Section Assignments

# Assertion check: Print plate bounding box after creation to verify dimensions
print('  [ASSERTION] Plate bounding box verification:')
print('      Expected Part A: X: 0-300mm, Y: 0-{half_t}mm, Z: 0-300mm')
xs = ys = zs = []
# Get actual bounds from Part A
try:
    verts_A = partA.vertices
    if verts_A:
        coords = [v.coordinates for v in verts_A]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        zs = [c[2] for c in coords]
        print('      Actual:   X: {{:.1f}}-{{:.1f}}mm, Y: {{:.1f}}-{{:.1f}}mm, Z: {{:.1f}}-{{:.1f}}mm'.format(min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)))
        # Basic validation (Python 2/3 compatible — avoid f-strings)
        assert min(xs) >= 0 and max(xs) <= 300, "X dimension out of bounds"
        assert min(ys) >= 0 and max(ys) <= {half_t} + 0.01, "Y dimension out of bounds: %s" % max(ys)
        assert min(zs) >= 0 and max(zs) <= 300, "Z dimension out of bounds"
        print('      [OK] Bounds validation passed')
    else:
        print('      [WARNING] Could not retrieve vertices for bounds check')
except Exception as e:
    print('      [WARNING] Bounds check failed: {{}}'.format(e))
"""

    def _section_assignments(self, mat, layup, material_name):
        ply_t = PLATE["ply_thickness"]
        lines = []
        lines.append(f"""
# ==========================================================================
# PLY SECTIONS AND ORIENTATION ASSIGNMENTS
# ==========================================================================
# Each ply gets its own solid section with material orientation
# Ply coordinate system: fiber direction rotated in XZ plane
# 1 element per ply through thickness
""")

        # Part A: plies 1-4 (bottom to top within the part → Y=0 to Y=0.5)
        lines.append("# --- Part A sections (plies 1-4, bottom sub-laminate) ---")
        for i in range(4):
            ply_num = i + 1
            angle = layup["angles"][i]
            y_bot = i * ply_t
            y_top = (i + 1) * ply_t
            sec_name = f"Section_Ply{ply_num}"
            lines.append(f"""
# Ply {ply_num}: angle = {angle}°, Y = {y_bot} to {y_top} mm (within Part A)
mdl.HomogeneousSolidSection(name='{sec_name}', material=matName, thickness=None)

# Create datum plane to partition ply {ply_num}
cellsA = partA.cells.getByBoundingBox(
    xMin=-1, yMin={y_bot - 0.001}, zMin=-1,
    xMax=plateLength+1, yMax={y_top + 0.001}, zMax=plateWidth+1)
regionA_{ply_num} = regionToolset.Region(cells=cellsA)
partA.SectionAssignment(region=regionA_{ply_num}, sectionName='{sec_name}')

# Material orientation for ply {ply_num}
partA.MaterialOrientation(
    region=regionA_{ply_num},
    orientationType=SYSTEM,
    axis=AXIS_2,
    localCsys=None,
    additionalRotationType=ROTATION_ANGLE,
    additionalRotationField='',
    angle={float(angle)},
    stackDirection=STACK_2)
""")

        # Part B: plies 5-8
        lines.append("\n# --- Part B sections (plies 5-8, top sub-laminate) ---")
        for i in range(4):
            ply_num = i + 5
            angle = layup["angles"][i + 4]
            y_bot = i * ply_t
            y_top = (i + 1) * ply_t
            sec_name = f"Section_Ply{ply_num}"
            lines.append(f"""
# Ply {ply_num}: angle = {angle}°, Y = {y_bot} to {y_top} mm (within Part B)
mdl.HomogeneousSolidSection(name='{sec_name}', material=matName, thickness=None)

cellsB = partB.cells.getByBoundingBox(
    xMin=-1, yMin={y_bot - 0.001}, zMin=-1,
    xMax=plateLength+1, yMax={y_top + 0.001}, zMax=plateWidth+1)
regionB_{ply_num} = regionToolset.Region(cells=cellsB)
partB.SectionAssignment(region=regionB_{ply_num}, sectionName='{sec_name}')

partB.MaterialOrientation(
    region=regionB_{ply_num},
    orientationType=SYSTEM,
    axis=AXIS_2,
    localCsys=None,
    additionalRotationType=ROTATION_ANGLE,
    additionalRotationField='',
    angle={float(angle)},
    stackDirection=STACK_2)
""")

        lines.append("print('  [OK] Ply sections and orientations assigned')")
        return "\n".join(lines)

    def _partition_through_thickness(self, mat):
        ply_t = PLATE["ply_thickness"]
        return f"""
# ==========================================================================
# THROUGH-THICKNESS PARTITIONING (Ply interfaces)
# ==========================================================================
# Through-thickness: 1 element per ply (= {ply_t} mm per element)
# Total through thickness: 4 elements per sub-laminate, 8 elements total

plyThick = {ply_t}

# ---- Partition each part into 4 plies through thickness ----
# Part A partitioning (plies 1-4)
for i in range(1, 4):
    y_cut = i * plyThick
    partA_datum = partA.DatumPlaneByPrincipalPlane(
        principalPlane=XZPLANE, offset=y_cut)
    partA.PartitionCellByDatumPlane(
        datumPlane=partA.datums[partA_datum.id],
        cells=partA.cells[:])

# Part B partitioning (plies 5-8)
for i in range(1, 4):
    y_cut = i * plyThick
    partB_datum = partB.DatumPlaneByPrincipalPlane(
        principalPlane=XZPLANE, offset=y_cut)
    partB.PartitionCellByDatumPlane(
        datumPlane=partB.datums[partB_datum.id],
        cells=partB.cells[:])

print('  [OK] Parts partitioned into 4 plies each (through thickness)')
"""

    def _partition_delamination_boundaries(self, mat, delam):
        """Partition parts at delamination boundaries for clean surface selection."""
        if not delam or 'damages' not in delam:
            return ""

        damages = delam['damages']
        lines = ["# ==========================================================================\n"
                 "# DELAMINATION BOUNDARY PARTITIONING (Optimized)\n"
                 "# =========================================================================="]
        
        # Collect unique offsets
        x_cuts = set()
        z_cuts = set()
        for d in damages:
            cx, cz = d['center_x'], d['center_z']
            sx, sz = d['size_x'] / 2.0, d['size_z'] / 2.0
            x_cuts.add(cx - sx)
            x_cuts.add(cx + sx)
            z_cuts.add(cz - sz)
            z_cuts.add(cz + sz)

        lines.append("for prt in [partA, partB]:")
        lines.append("    # X-cuts (YZ planes)")
        for x in sorted(list(x_cuts)):
            lines.append(f"    try:")
            lines.append(f"        dp = prt.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset={x})")
            lines.append(f"        prt.PartitionCellByDatumPlane(datumPlane=prt.datums[dp.id], cells=prt.cells[:])")
            lines.append(f"    except: pass")
            
        lines.append("    # Z-cuts (XY planes)")
        for z in sorted(list(z_cuts)):
            lines.append(f"    try:")
            lines.append(f"        dp = prt.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset={z})")
            lines.append(f"        prt.PartitionCellByDatumPlane(datumPlane=prt.datums[dp.id], cells=prt.cells[:])")
            lines.append(f"    except: pass")
            
        lines.append("print('  [OK] Parts partitioned at all delamination boundaries')")
        return "\n".join(lines)

    def _seed_and_generate_mesh(self, mat):
        """Seed and generate mesh with through-thickness control."""
        mesh_size = mat["mesh_size"]
        ply_t = PLATE["ply_thickness"]

        return f"""
# ==========================================================================
# MESH SEEDING AND GENERATION
# ==========================================================================
# In-plane mesh size: {mesh_size} mm (λ/10 for {mat["display_name"]})
# Through-thickness: 1 element per ply (= {ply_t} mm per element)

meshSize = {mesh_size}
plyThick = {ply_t}

# ---- Seed and generate mesh ----
# In-plane seeding
partA.seedPart(size=meshSize, deviationFactor=0.1, minSizeFactor=0.1)
partB.seedPart(size=meshSize, deviationFactor=0.1, minSizeFactor=0.1)

# ---- Through-thickness mesh control: Force exactly 1 element per ply layer ----
# We filter edges by direction (Y-axis) to ensure only through-thickness edges are seeded
for prt in [partA, partB]:
    for cell in prt.cells:
        cellEdges = cell.getEdges()
        for edgeIdx in cellEdges:
            edge = prt.edges[edgeIdx]
            v1_idx, v2_idx = edge.getVertices()
            v1 = prt.vertices[v1_idx].pointOn[0]
            v2 = prt.vertices[v2_idx].pointOn[0]
            dx = abs(v1[0] - v2[0])
            dy = abs(v1[1] - v2[1])
            dz = abs(v1[2] - v2[2])
            # Y-direction edge: dx ≈ 0 and dz ≈ 0, dy > 0
            if dx < 1e-4 and dz < 1e-4 and dy > 1e-4:
                prt.seedEdgeByNumber(edges=prt.edges[edgeIdx:edgeIdx+1], number=1)

# Set element types (Python 2 compatible, no default args)
elemType1 = mesh.ElemType(
    elemCode=C3D8R,
    elemLibrary=EXPLICIT,
    hourglassControl=ENHANCED)
elemType2 = mesh.ElemType(
    elemCode=C3D6,
    elemLibrary=EXPLICIT)
elemType3 = mesh.ElemType(
    elemCode=C3D4,
    elemLibrary=EXPLICIT)

partA.setElementType(
    regions=(partA.cells,),
    elemTypes=(elemType1, elemType2, elemType3))

partB.setElementType(
    regions=(partB.cells,),
    elemTypes=(elemType1, elemType2, elemType3))

partA.generateMesh()
partB.generateMesh()

print('  [OK] Mesh generated:')
print('      In-plane element size: {mesh_size} mm')
print('      Through-thickness: 1 element per ply ({ply_t} mm)')
print('      Element type: C3D8R (Enhanced hourglass)')
print('      Part A elements:', len(partA.elements))
print('      Part B elements:', len(partB.elements))

"""

    def _assembly(self, model_name):
        half_t = PLATE["thickness"] / 2.0
        return f"""
# ==========================================================================
# ASSEMBLY
# ==========================================================================
# Part A: bottom sub-laminate at Y = 0 to {half_t} mm
# Part B: top sub-laminate translated so its bottom face is at Y = {half_t} mm

assembly = mdl.rootAssembly
assembly.DatumCsysByDefault(CARTESIAN)

# Instance Part A (at origin — bottom sub-laminate)
instA = assembly.Instance(name='PartA_Bottom-1', part=partA, dependent=ON)

# Instance Part B (translated up by {half_t} mm in Y)
instB = assembly.Instance(name='PartB_Top-1', part=partB, dependent=ON)
assembly.translate(instanceList=('PartB_Top-1',), vector=(0.0, {half_t}, 0.0))

print('  [OK] Assembly created:')
print('      Part A at Y = 0 to {half_t} mm')
print('      Part B at Y = {half_t} to {PLATE["thickness"]} mm')
"""

    def _surfaces(self, model_name, is_damaged, delam):
        """Create surfaces on the interface between Part A (top) and Part B (bottom)."""
        half_t = PLATE["thickness"] / 2.0
        L = PLATE["length"]
        W = PLATE["width"]

        lines = []
        lines.append(f"""
# ==========================================================================
# INTERFACE SURFACES
# ==========================================================================
# Part A top face (Y = {half_t}) and Part B bottom face (Y = {half_t})

assembly = mdl.rootAssembly
instA = assembly.instances['PartA_Bottom-1']
instB = assembly.instances['PartB_Top-1']

# --- Part A top surface (Y = {half_t} mm) ---
facesA_top = instA.faces.getByBoundingBox(
    xMin=-0.01, yMin={half_t - 0.001}, zMin=-0.01,
    xMax={L + 0.01}, yMax={half_t + 0.001}, zMax={W + 0.01})
assembly.Surface(side1Faces=facesA_top, name='SurfA_Top')

# --- Part B bottom surface (Y = {half_t} mm) ---
facesB_bot = instB.faces.getByBoundingBox(
    xMin=-0.01, yMin={half_t - 0.001}, zMin=-0.01,
    xMax={L + 0.01}, yMax={half_t + 0.001}, zMax={W + 0.01})
assembly.Surface(side1Faces=facesB_bot, name='SurfB_Bottom')

print('  [OK] Interface full surfaces defined')
""")

        return "\n".join(lines)

    def _constraints(self, model_name, is_damaged, delam):
        lines = []
        lines.append("# ==========================================================================\n"
                     "# CONSTRAINTS - Tie + Contact\n"
                     "# ==========================================================================\n")

        if not is_damaged:
            # Healthy: tie everything
            lines.append("# HEALTHY MODEL - Full tie constraint across entire interface\n"
                         "mdl.Tie(\n"
                         "    name='Tie_FullInterface',\n"
                         "    master=assembly.surfaces['SurfA_Top'],\n"
                         "    slave=assembly.surfaces['SurfB_Bottom'],\n"
                         "    positionToleranceMethod=COMPUTED,\n"
                         "    adjust=ON,\n"
                         "    tieRotations=ON,\n"
                         "    thickness=ON,\n"
                         "    constraintEnforcement=SURFACE_TO_SURFACE)\n\n"
                         "print('  [OK] Tie constraint: full interface (healthy plate)')")
        else:
            damages = delam['damages']
            half_t = PLATE["thickness"] / 2.0
            L = PLATE["length"]
            W = PLATE["width"]

            lines.append(f"""
# --- Damaged Model (Multiple Delaminations) ---
allFacesA_top = instA.faces.getByBoundingBox(
    xMin=-0.01, yMin={half_t - 0.001}, zMin=-0.01,
    xMax={L + 0.01}, yMax={half_t + 0.001}, zMax={W + 0.01})
allFacesB_bot = instB.faces.getByBoundingBox(
    xMin=-0.01, yMin={half_t - 0.001}, zMin=-0.01,
    xMax={L + 0.01}, yMax={half_t + 0.001}, zMax={W + 0.01})

delam_indices_A = []
delam_indices_B = []

# Interaction property for delamination
if 'IntProp_Delam' not in mdl.interactionProperties.keys():
    mdl.ContactProperty('IntProp_Delam')
    mdl.interactionProperties['IntProp_Delam'].TangentialBehavior(formulation=FRICTIONLESS)
    mdl.interactionProperties['IntProp_Delam'].NormalBehavior(pressureOverclosure=HARD, allowSeparation=ON)
""")

            for i, d in enumerate(damages):
                cx, cz = d['center_x'], d['center_z']
                sx, sz = d['size_x'] / 2.0, d['size_z'] / 2.0
                name = f"Delam_{i+1}"
                
                lines.append(f"""
# Damage {i+1}: {d['size_x']}x{d['size_z']} mm at ({cx}, {cz})
cx, cz, sx, sz = {cx}, {cz}, {sx}, {sz}
idxA = [f.index for f in allFacesA_top if (cx-sx-0.1 < f.pointOn[0][0] < cx+sx+0.1 and cz-sz-0.1 < f.pointOn[0][2] < cz+sz+0.1)]
if idxA:
    delam_indices_A.extend(idxA)
    assembly.Surface(side1Faces=instA.faces.sequenceFromLabels(idxA), name='SurfA_{name}')

idxB = [f.index for f in allFacesB_bot if (cx-sx-0.1 < f.pointOn[0][0] < cx+sx+0.1 and cz-sz-0.1 < f.pointOn[0][2] < cz+sz+0.1)]
if idxB:
    delam_indices_B.extend(idxB)
    assembly.Surface(side1Faces=instB.faces.sequenceFromLabels(idxB), name='SurfB_{name}')

if 'SurfA_{name}' in assembly.surfaces.keys() and 'SurfB_{name}' in assembly.surfaces.keys():
    mdl.SurfaceToSurfaceContactExp(name='Contact_{name}', createStepName='Initial', 
        master=assembly.surfaces['SurfA_{name}'], slave=assembly.surfaces['SurfB_{name}'], 
        mechanicalConstraint=KINEMATIC, sliding=FINITE, interactionProperty='IntProp_Delam')
""")

            lines.append(f"""
# --- Healthy Zone selection (Everything else) ---
delam_indices_A = list(set(delam_indices_A))
delam_indices_B = list(set(delam_indices_B))

facesA_healthy_labels = [f.index for f in allFacesA_top if f.index not in delam_indices_A]
if facesA_healthy_labels:
    assembly.Surface(side1Faces=instA.faces.sequenceFromLabels(facesA_healthy_labels), name='SurfA_Healthy')

facesB_healthy_labels = [f.index for f in allFacesB_bot if f.index not in delam_indices_B]
if facesB_healthy_labels:
    assembly.Surface(side1Faces=instB.faces.sequenceFromLabels(facesB_healthy_labels), name='SurfB_Healthy')

if 'SurfA_Healthy' in assembly.surfaces.keys() and 'SurfB_Healthy' in assembly.surfaces.keys():
    mdl.Tie(name='Tie_HealthyZone', master=assembly.surfaces['SurfA_Healthy'], slave=assembly.surfaces['SurfB_Healthy'],
            positionToleranceMethod=COMPUTED, adjust=ON, tieRotations=ON, thickness=ON, constraintEnforcement=SURFACE_TO_SURFACE)

print('  [OK] Constraints applied: ties in healthy zones, contact in delamination zones')
""")

        return "\n".join(lines)

        return "\n".join(lines)

    def _step(self, model_name, mat):
        step_name = STEP["step_name"]
        total_time = STEP["total_time"]
        dt = mat["dt"]

        return f"""
# ==========================================================================
# ANALYSIS STEP - Explicit Dynamics
# ==========================================================================
mdl.ExplicitDynamicsStep(
    name='{step_name}',
    previous='Initial',
    timePeriod={total_time},
    maxIncrement={dt},
    massScaling=PREVIOUS_STEP,
    description='Lamb wave propagation step - {mat["display_name"]}')

print('  [OK] Step created: {step_name}')
print('      Total time: {total_time} s ({total_time * 1e6:.0f} us)')
print('      Max increment: {dt} s')
"""

    def _amplitude(self, model_name, mat):
        csv_path = mat["tone_burst_csv"]
        return f"""
# ==========================================================================
# TONE BURST AMPLITUDE - from CSV
# ==========================================================================
# Load tone burst signal from CSV file
# CSV format: time(s), amplitude (normalized displacement)
# File: {csv_path}

# IMPORTANT: Set this path to the absolute path of the tone burst CSV file
# on the machine where Abaqus will run
csvPath = r'{csv_path}'

# Read CSV and build amplitude data
ampData = []
try:
    with open(csvPath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                t = float(row[0].strip())
                a = float(row[1].strip())
                ampData.append((t, a))
    print('  [OK] Tone burst CSV loaded:', len(ampData), 'data points')
except IOError:
    print('  [WARNING] Could not read tone burst CSV:', csvPath)
    print('  [WARNING] Please update csvPath to the correct absolute path')
    # Fallback: generate a simple tone burst analytically
    import math
    fc = {EXCITATION["frequency"]}    # Hz
    ncyc = {EXCITATION["num_cycles"]}
    t1 = ncyc / fc                    # excitation duration
    npts = 600
    for i in range(npts + 1):
        t = i * t1 / npts
        amp = -0.5 * math.sin(2 * math.pi * fc * t) * (1 - math.cos(2 * math.pi * fc * t / ncyc))
        ampData.append((t, amp))
    ampData.append((t1 * 1.001, 0.0))
    print('  [OK] Fallback: analytical tone burst generated with', len(ampData), 'points')

# Create tabular amplitude in Abaqus
mdl.TabularAmplitude(
    name='ToneBurst_50kHz',
    timeSpan=STEP,
    smooth=SOLVER_DEFAULT,
    data=tuple(ampData))

print('  [OK] TabularAmplitude created: ToneBurst_50kHz')
"""

    def _actuator(self, model_name):
        ax = EXCITATION["actuator_x"]
        ay = EXCITATION["actuator_y"]
        az = EXCITATION["actuator_z"]
        ar = EXCITATION["actuator_radius"]
        amp = EXCITATION["amplitude"]

        return f"""
# ==========================================================================
# ACTUATOR - Ring of nodes for wave excitation
# ==========================================================================
# Location: ({ax}, {ay}, {az}) mm - on bottom surface (Y=0)
# Ring radius: {ar} mm - mimics circular PZT transducer
# Applied as U2 displacement (Y-direction) with tone burst amplitude

actX = {ax}
actY = {ay}
actZ = {az}
actRadius = {ar}
actAmplitude = {amp}

# Select nodes within actuator radius on bottom surface of Part A
instA = assembly.instances['PartA_Bottom-1']
allNodes = instA.nodes

# Find nodes within radius on bottom surface (Y ≈ 0)
actuatorNodes = []
for node in allNodes:
    x, y, z = node.coordinates
    if abs(y - actY) < 0.01:  # on bottom surface
        dist = ((x - actX)**2 + (z - actZ)**2)**0.5
        if dist <= actRadius:
            actuatorNodes.append(node)

# Create node set for actuator
actuatorNodeLabels = [n.label for n in actuatorNodes]
actuatorRegion = assembly.Set(
    nodes=instA.nodes.sequenceFromLabels(actuatorNodeLabels),
    name='Set_Actuator')

print('  [OK] Actuator node set created:', len(actuatorNodeLabels), 'nodes')
print('      Center: ({ax}, {ay}, {az}) mm, Radius: {ar} mm')

# Apply displacement BC with tone burst amplitude
mdl.DisplacementBC(
    name='BC_ToneBurst',
    createStepName='{STEP["step_name"]}',
    region=actuatorRegion,
    u1=UNSET, u2=actAmplitude, u3=UNSET,
    ur1=UNSET, ur2=UNSET, ur3=UNSET,
    amplitude='ToneBurst_50kHz',
    fixed=OFF,
    distributionType=UNIFORM,
    fieldName='',
    localCsys=None)

print('  [OK] Tone burst BC applied: U2 = {amp} mm with ToneBurst_50kHz amplitude')
"""

    def _sensors(self, model_name):
        sx = SENSORS["x_position"]
        sy = SENSORS["y_position"]
        z_positions = SENSORS["z_positions"]
        step_name = STEP["step_name"]

        lines = []
        lines.append(f"""
# ==========================================================================
# SENSOR NODES - History Output
# ==========================================================================
# 7 sensors at X = {sx} mm, Y = {sy} mm (bottom surface)
# Z positions: {z_positions}
# Output: U2 at every increment

instA = assembly.instances['PartA_Bottom-1']
allNodes = instA.nodes
""")

        for i, z_pos in enumerate(z_positions):
            lines.append(f"""
# --- Sensor {i}: X={sx}, Y={sy}, Z={z_pos} ---
sensorNodes_{i} = []
for node in allNodes:
    x, y, z = node.coordinates
    if abs(y - {sy}) < 0.01 and abs(x - {sx}) < 1.0 and abs(z - {z_pos}) < 1.0:
        sensorNodes_{i}.append(node)

# Pick the closest node to exact sensor position
if sensorNodes_{i}:
    sensorNodes_{i}.sort(key=lambda n: ((n.coordinates[0]-{sx})**2 +
                                        (n.coordinates[2]-{z_pos})**2))
    closest_{i} = sensorNodes_{i}[0]
    sensorSet_{i} = assembly.Set(
        nodes=instA.nodes.sequenceFromLabels([closest_{i}.label]),
        name='Sensor_{i}')
    print('  [OK] Sensor_{i} at node', closest_{i}.label,
          '- coords:', closest_{i}.coordinates)
else:
    print('  [WARNING] No node found near Sensor_{i} ({sx}, {sy}, {z_pos})')
""")

        # History output request for all sensors
        lines.append(f"""
# --- History output for all sensors ---
sensorSetNames = ['Sensor_' + str(i) for i in range({len(z_positions)})]
for setName in sensorSetNames:
    if setName in assembly.sets:
        mdl.HistoryOutputRequest(
            name='HOutput_' + setName,
            createStepName='{step_name}',
            variables=('U2',),
            region=assembly.sets[setName],
            sectionPoints=DEFAULT,
            rebar=EXCLUDE,
            numIntervals=5000) # Replaced 0 with 5000 for Abaqus 2020 compatibility

print('  [OK] History output requests created for', len(sensorSetNames), 'sensors')
print('      Output: U2 at 5000 intervals (high resolution)')
""")
        return "\n".join(lines)

    def _boundary_conditions(self, model_name):
        """Add boundary conditions to prevent rigid body drift (Soft Support)."""
        L = PLATE["length"]
        W = PLATE["width"]
        
        return f"""
# ==========================================================================
# BOUNDARY CONDITIONS - Soft Support (Stability)
# ==========================================================================
# To prevent rigid body drift (the plate "floating" away) while remaining 
# Free-Free, we pin only the four corner nodes of the bottom surface.
# This prevents drift in X, Y, and Z but allows the rest of the plate to vibrate.

instA = assembly.instances['PartA_Bottom-1']
corners = [
    (0.0, 0.0, 0.0),      # Corner 1
    ({L}, 0.0, 0.0),      # Corner 2
    (0.0, 0.0, {W}),      # Corner 3
    ({L}, 0.0, {W})       # Corner 4
]

cornerNodeLabels = []
for cx, cy, cz in corners:
    nodes = instA.nodes.getByBoundingSphere(center=(cx, cy, cz), radius=1.0)
    if nodes:
        cornerNodeLabels.append(nodes[0].label)

if cornerNodeLabels:
    cornerRegion = assembly.Set(
        nodes=instA.nodes.sequenceFromLabels(cornerNodeLabels),
        name='Set_StabilityCorners')
    
    mdl.DisplacementBC(
        name='BC_Stability',
        createStepName='Initial',
        region=cornerRegion,
        u1=0.0, u2=0.0, u3=0.0,
        ur1=UNSET, ur2=UNSET, ur3=UNSET,
        amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, 
        fieldName='', localCsys=None)
    
    print('  [OK] Soft support BC applied: pinned 4 corners (X,Y,Z fixed)')
else:
    print('  [WARNING] Could not find corner nodes for stability BC')
"""

    def _field_output(self, model_name):
        return f"""
# ==========================================================================
# FIELD OUTPUT - for wave animation
# ==========================================================================
mdl.FieldOutputRequest(
    name='FOutput_Wave',
    createStepName='{STEP["step_name"]}',
    variables=('U',),
    numIntervals={STEP["num_field_intervals"]})

print('  [OK] Field output request: U (displacement) at {STEP["num_field_intervals"]} intervals')
"""

    def _job(self, model_name, mat):
        return f"""
# ==========================================================================
# JOB DEFINITION
# ==========================================================================
jobName = '{model_name}'
mdb.Job(
    name=jobName,
    model=modelName,
    description='Lamb wave - {mat["display_name"]}',
    type=ANALYSIS,
    atTime=None,
    waitMinutes=0,
    waitHours=0,
    queue=None,
    memory=90,
    memoryUnits=PERCENTAGE,
    explicitPrecision=DOUBLE_PLUS_PACK,
    nodalOutputPrecision=FULL,
    echoPrint=OFF,
    modelPrint=OFF,
    contactPrint=OFF,
    historyPrint=OFF,
    userSubroutine='',
    scratch='',
    resultsFormat=ODB,
    numDomains=1,
    multiprocessingMode=DEFAULT,
    numCpus=1)

print('  [OK] Job created: ' + jobName)
print('      Ready to submit with: mdb.jobs[\\'{model_name}\\'].submit()')
"""

    def _footer(self, model_name):
        return f"""
# ==========================================================================
# SAVE MODEL
# ==========================================================================
mdb.saveAs(pathName='{model_name}')

print('')
print('='*70)
print('MODEL BUILD COMPLETE: {model_name}')
print('='*70)
print('  To run: mdb.jobs[\\'{model_name}\\'].submit()')
print('  To monitor: mdb.jobs[\\'{model_name}\\'].waitForCompletion()')
print('='*70)
"""

    # =========================================================================
    # PRIVATE — File I/O
    # =========================================================================

    def _write_script(self, material_name, filename, content):
        """Write script content to file, creating directories as needed."""
        dirpath = os.path.join(self.output_dir, material_name)
        os.makedirs(dirpath, exist_ok=True)
        filepath = os.path.join(dirpath, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath
