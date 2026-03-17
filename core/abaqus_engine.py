# -*- coding: utf-8 -*-
"""
core/abaqus_engine.py
=====================
Flexible Abaqus 2020 Python script generator for Lamb wave studies.
Refactored from scripts/abaqus_generator.py to be integrated with the GUI.

Generates complete .py scripts that can be run inside Abaqus/CAE.
"""

import os
import datetime

class AbaqusEngine:
    """
    Generates Abaqus/Explicit Python scripts based on parameters provided as a dictionary.
    """

    def __init__(self, params):
        """
        Parameters
        ----------
        params : dict
            A dictionary containing all necessary simulation parameters.
        """
        self.p = params
        # Unit conversion: GPa to MPa for moduli, kg/m3 to tonne/mm3 for density
        self.p['E1_MPa'] = self.p['E1'] * 1000.0
        self.p['E2_MPa'] = self.p['E2'] * 1000.0
        self.p['E3_MPa'] = self.p['E2'] * 1000.0 # Transverse isotropic assumption
        self.p['G12_MPa'] = self.p['G12'] * 1000.0
        self.p['G13_MPa'] = self.p['G12'] * 1000.0
        self.p['G23_MPa'] = self.p['G23'] * 1000.0
        self.p['density_t_mm3'] = self.p['rho'] / 1e12
        
        # Derived properties for Transverse Isotropy (Planes 1-2 and 1-3 equal)
        self.p['nu13'] = self.p['nu12']
        # nu23 = E2/(2*G23) - 1
        self.p['nu23'] = self.p['E2'] / (2.0 * self.p['G23']) - 1.0

        # Normalize damages for multiple delaminations
        if 'damages' not in self.p:
            # Backward compatibility for single damage
            if all(k in self.p for k in ['delam_cx', 'delam_cz', 'delam_sx', 'delam_sz']):
                self.p['damages'] = [{
                    'center_x': self.p['delam_cx'],
                    'center_z': self.p['delam_cz'],
                    'size_x': self.p['delam_sx'],
                    'size_z': self.p['delam_sz']
                }]
            else:
                self.p['damages'] = [] # Healthy model or no damage defined
                
        # Run validation
        self.validate_parameters()

    def validate_parameters(self):
        """
        Performs geometrical and logical validation of the simulation parameters.
        Raises ValueError if any critical issue is found.
        """
        L, W = self.p['plate_L'], self.p['plate_W']
        T = self.p['thickness']
        angles = self.p.get('angles', [])
        num_plies = len(angles)
        
        # 1. Interface Validation
        k = self.p.get('delam_interface', 4)
        if k < 1 or k >= num_plies:
            raise ValueError(f"Invalid delamination interface {k}. Must be between 1 and {num_plies-1} for a {num_plies}-ply laminate.")

        # 2. Damage Validation (Bounds + Overlap)
        damages = self.p.get('damages', [])
        for i, d1 in enumerate(damages):
            cx1, cz1 = d1['center_x'], d1['center_z']
            sx1, sz1 = d1['size_x'], d1['size_z']
            
            # Non-zero size
            if sx1 <= 0 or sz1 <= 0:
                raise ValueError(f"Damage #{i+1} has invalid size: {sx1}x{sz1}. Must be positive.")
            
            # Out of bounds
            if (cx1 - sx1/2.0 < 0 or cx1 + sx1/2.0 > L or 
                cz1 - sz1/2.0 < 0 or cz1 + sz1/2.0 > W):
                raise ValueError(f"Damage #{i+1} at ({cx1}, {cz1}) is outside the plate ({L}x{W} mm).")
            
            # Overlap with other damages
            for j, d2 in enumerate(damages):
                if i == j: continue
                cx2, cz2 = d2['center_x'], d2['center_z']
                sx2, sz2 = d2['size_x'], d2['size_z']
                
                # Check for AABB intersection
                # |cx1 - cx2| < (sx1 + sx2)/2
                if (abs(cx1 - cx2) < (sx1 + sx2)/2.0 and 
                    abs(cz1 - cz2) < (sz1 + sz2)/2.0):
                    raise ValueError(f"Overlapping delaminations: Damage #{i+1} overlaps with Damage #{j+1}.")

        # 3. Actuator/Sensor Presence
        ax, az = self.p.get('actuator_x'), self.p.get('actuator_z')
        if ax is not None and (ax < 0 or ax > L or az < 0 or az > W):
            raise ValueError(f"Actuator at ({ax}, {az}) is outside the plate.")

        return True

    def generate_script(self, is_damaged=False):
        """Assemble the full Abaqus Python script as a string."""
        
        sections = []
        
        # Header
        sections.append(self._header(is_damaged))
        
        # Imports
        sections.append(self._imports())
        
        # Model setup
        sections.append(self._model_setup())
        
        # Material definition
        sections.append(self._material_definition())
        
        # Part creation (Two sub-laminates split at the delamination interface)
        sections.append(self._part_creation())
        
        # Partition through thickness
        sections.append(self._partition_through_thickness())
        
        # Partition delamination zone (only if damaged)
        if is_damaged:
            sections.append(self._partition_delamination_boundaries())
            
        # Assign sections and orientations
        sections.append(self._section_assignments())
        
        # Seed and generate mesh
        sections.append(self._seed_and_generate_mesh())
        
        # Assembly
        sections.append(self._assembly())
        
        # Surfaces
        sections.append(self._surfaces(is_damaged))
        
        # Constraints
        sections.append(self._constraints(is_damaged))
        
        # Step
        sections.append(self._step())
        
        # Amplitude (Tone burst)
        sections.append(self._amplitude())
        
        # Actuator
        sections.append(self._actuator())
        
        # Sensors
        sections.append(self._sensors())
        
        # Boundary Conditions
        sections.append(self._boundary_conditions())
        
        # Field output
        sections.append(self._field_output())
        
        # Job
        sections.append(self._job())
        
        # Footer
        sections.append(self._footer())
        
        return "\n".join(sections)

    def _header(self, is_damaged):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_type = "DAMAGED" if is_damaged else "HEALTHY"
        mat_name = self.p.get('mat_name', 'Custom Material')
        
        return f'''# -*- coding: utf-8 -*-
"""
===========================================================================
Abaqus/Explicit Python Script — Lamb Wave Propagation
===========================================================================
Model Name   : {self.p['model_name']}
Model Type   : {model_type}
Material     : {mat_name}
Layup        : {self.p['layup_str']}
---------------------------------------------------------------------------
Generated    : {now}
Generated by : Lamb Wave Dispersion Toolkit
Unit System  : mm, tonne, s, MPa
Solver       : Abaqus/Explicit
===========================================================================
"""
'''

    def _imports(self):
        return """
from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
import regionToolset
import mesh
import math

executeOnCaeStartup()
"""

    def _model_setup(self):
        return f"""
# --- Model Setup ---
modelName = '{self.p['model_name']}'
if modelName in mdb.models:
    del mdb.models[modelName]
mdb.Model(name=modelName, modelType=STANDARD_EXPLICIT)
mdl = mdb.models[modelName]
"""

    def _material_definition(self):
        return f"""
# --- Material Definition ---
matName = '{self.p.get('mat_name', 'Material-1')}'
material = mdl.Material(name=matName)
material.Elastic(type=ENGINEERING_CONSTANTS, table=((
    {self.p['E1_MPa']}, {self.p['E2_MPa']}, {self.p['E3_MPa']},
    {self.p['nu12']}, {self.p['nu13']}, {self.p['nu23']},
    {self.p['G12_MPa']}, {self.p['G13_MPa']}, {self.p['G23_MPa']}
),))
material.Density(table=(({self.p['density_t_mm3']},),))
"""

    def _part_creation(self):
        L = self.p['plate_L']
        W = self.p['plate_W']
        T = self.p['thickness']
        
        k = self.p.get('delam_interface', len(self.p['angles']) // 2)
        total_plies = len(self.p['angles'])
        ply_t = T / total_plies
        y_split = k * ply_t
        
        self.p['_y_split'] = y_split # save for later
        self.p['_k_split'] = k
        self.p['_ply_t'] = ply_t
        
        return f"""
# --- Part Creation (Sub-laminates) ---
plateL = {L}
plateW = {W}
ySplit = {y_split}
totalT = {T}

# Part A (Bottom sub-laminate)
sketch_A = mdl.ConstrainedSketch(name='PartA_Sketch', sheetSize=max(plateL, plateW)*1.5)
sketch_A.rectangle(point1=(0.0, 0.0), point2=(plateL, ySplit))
partA = mdl.Part(name='PartA_Bottom', dimensionality=THREE_D, type=DEFORMABLE_BODY)
partA.BaseSolidExtrude(sketch=sketch_A, depth=plateW)

# Part B (Top sub-laminate)
sketch_B = mdl.ConstrainedSketch(name='PartB_Sketch', sheetSize=max(plateL, plateW)*1.5)
sketch_B.rectangle(point1=(0.0, 0.0), point2=(plateL, totalT - ySplit))
partB = mdl.Part(name='PartB_Top', dimensionality=THREE_D, type=DEFORMABLE_BODY)
partB.BaseSolidExtrude(sketch=sketch_B, depth=plateW)

# --- Global Coordinate System for Orientations ---
# Capture the CSYS objects to ensure they are used for Material Orientations
# (Avoids using partition planes by mistake)
c1 = partA.DatumCsysByThreePoints(name='Global_Orientation', coordSysType=CARTESIAN, 
    origin=(0.0, 0.0, 0.0), point1=(1.0, 0.0, 0.0), point2=(0.0, 1.0, 0.0))
csysA = partA.datums[c1.id]

c2 = partB.DatumCsysByThreePoints(name='Global_Orientation', coordSysType=CARTESIAN, 
    origin=(0.0, 0.0, 0.0), point1=(1.0, 0.0, 0.0), point2=(0.0, 1.0, 0.0))
csysB = partB.datums[c2.id]
"""

    def _partition_through_thickness(self):
        k = self.p['_k_split']
        total_plies = len(self.p['angles'])
        ply_t = self.p['_ply_t']
        
        return f"""
# --- Layer Partitioning ---
# Part A partitioning (plies 1 to {k})
for i in range(1, {k}):
    y_cut = i * {ply_t}
    dp = partA.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=y_cut)
    partA.PartitionCellByDatumPlane(datumPlane=partA.datums[dp.id], cells=partA.cells[:])

# Part B partitioning (plies {k+1} to {total_plies})
num_plies_B = {total_plies - k}
for i in range(1, num_plies_B):
    y_cut = i * {ply_t}
    dp = partB.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=y_cut)
    partB.PartitionCellByDatumPlane(datumPlane=partB.datums[dp.id], cells=partB.cells[:])
"""

    def _partition_delamination_boundaries(self):
        damages = self.p['damages']
        if not damages:
            return ""
            
        lines = ["# --- Delamination Boundaries (Optimized Partitioning) ---"]
        
        # Collect all unique cut offsets to avoid redundant datum planes
        x_cuts = set()
        z_cuts = set()
        for d in damages:
            cx, cz = d['center_x'], d['center_z']
            sx, sz = d['size_x'], d['size_z']
            x_cuts.add(cx - sx/2.0)
            x_cuts.add(cx + sx/2.0)
            z_cuts.add(cz - sz/2.0)
            z_cuts.add(cz + sz/2.0)
            
        lines.append("for prt in [partA, partB]:")
        lines.append("    # Apply X-axis partitions (YZ planes)")
        for x in sorted(list(x_cuts)):
            lines.append(f"    try:")
            lines.append(f"        dp = prt.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset={x})")
            lines.append(f"        prt.PartitionCellByDatumPlane(datumPlane=prt.datums[dp.id], cells=prt.cells[:])")
            lines.append(f"    except: pass")
            
        lines.append("    # Apply Z-axis partitions (XY planes)")
        for z in sorted(list(z_cuts)):
            lines.append(f"    try:")
            lines.append(f"        dp = prt.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset={z})")
            lines.append(f"        prt.PartitionCellByDatumPlane(datumPlane=prt.datums[dp.id], cells=prt.cells[:])")
            lines.append(f"    except: pass")
            
        return "\n".join(lines)

    def _section_assignments(self):
        angles = self.p['angles']
        k = self.p['_k_split']
        ply_t = self.p['_ply_t']

        lines = []
        lines.append(f"""
# --- PLY SECTIONS AND ORIENTATION ASSIGNMENTS ---
# Each ply gets its own solid section with material orientation.
# Fiber direction is rotated in the XZ plane by the ply angle.
# 1 element per ply through thickness.
""")

        # Part A plies
        lines.append("# --- Part A sections (bottom sub-laminate) ---")
        for i in range(k):
            ply_num = i + 1
            angle = angles[i]
            y_bot = i * ply_t
            y_top = (i + 1) * ply_t
            sec_name = f'Section_Ply{ply_num}'
            lines.append(f"""
# Ply {ply_num}: angle = {angle}°, Y = {y_bot:.4f} to {y_top:.4f} mm (within Part A)
mdl.HomogeneousSolidSection(name='{sec_name}', material=matName, thickness=None)

cellsA_{ply_num} = partA.cells.getByBoundingBox(
    xMin=-1, yMin={y_bot - 0.001:.4f}, zMin=-1,
    xMax=plateL+1, yMax={y_top + 0.001:.4f}, zMax=plateW+1)
regionA_{ply_num} = regionToolset.Region(cells=cellsA_{ply_num})
partA.SectionAssignment(region=regionA_{ply_num}, sectionName='{sec_name}')

partA.MaterialOrientation(
    region=regionA_{ply_num},
    orientationType=SYSTEM,
    axis=AXIS_2,
    localCsys=csysA,
    additionalRotationType=ROTATION_ANGLE,
    additionalRotationField='',
    angle={float(angle)})
print('  [OK] Ply {ply_num} (Part A): angle={angle} deg, section assigned and orientation set')
""")

        # Part B plies
        lines.append("# --- Part B sections (top sub-laminate) ---")
        for i in range(len(angles) - k):
            ply_num = i + k + 1
            angle = angles[i + k]
            y_bot = i * ply_t
            y_top = (i + 1) * ply_t
            sec_name = f'Section_Ply{ply_num}'
            lines.append(f"""
# Ply {ply_num}: angle = {angle}°, Y = {y_bot:.4f} to {y_top:.4f} mm (within Part B)
mdl.HomogeneousSolidSection(name='{sec_name}', material=matName, thickness=None)

cellsB_{ply_num} = partB.cells.getByBoundingBox(
    xMin=-1, yMin={y_bot - 0.001:.4f}, zMin=-1,
    xMax=plateL+1, yMax={y_top + 0.001:.4f}, zMax=plateW+1)
regionB_{ply_num} = regionToolset.Region(cells=cellsB_{ply_num})
partB.SectionAssignment(region=regionB_{ply_num}, sectionName='{sec_name}')

partB.MaterialOrientation(
    region=regionB_{ply_num},
    orientationType=SYSTEM,
    axis=AXIS_2,
    localCsys=csysB,
    additionalRotationType=ROTATION_ANGLE,
    additionalRotationField='',
    angle={float(angle)})
print('  [OK] Ply {ply_num} (Part B): angle={angle} deg, section assigned and orientation set')
""")

        lines.append("print('  [OK] All ply sections and material orientations assigned')")
        return "\n".join(lines)

    def _seed_and_generate_mesh(self):
        return f"""
# --- Meshing ---
meshSize = {self.p['mesh_size']}
partA.seedPart(size=meshSize, deviationFactor=0.1, minSizeFactor=0.1)
partB.seedPart(size=meshSize, deviationFactor=0.1, minSizeFactor=0.1)

# Ensure 1 element per ply
for prt in [partA, partB]:
    for edge in prt.edges:
        v1, v2 = edge.getVertices()
        p1, p2 = prt.vertices[v1].pointOn[0], prt.vertices[v2].pointOn[0]
        # Thickness is now Y-axis. Find edges with dX=0 and dZ=0
        if abs(p1[0]-p2[0]) < 1e-4 and abs(p1[2]-p2[2]) < 1e-4:
            prt.seedEdgeByNumber(edges=prt.edges[edge.index:edge.index+1], number=1)

elemType = mesh.ElemType(elemCode=C3D8R, elemLibrary=EXPLICIT, hourglassControl=ENHANCED)
partA.setElementType(regions=(partA.cells,), elemTypes=(elemType,))
partB.setElementType(regions=(partB.cells,), elemTypes=(elemType,))
partA.generateMesh()
partB.generateMesh()
"""

    def _assembly(self):
        y_split = self.p['_y_split']
        return f"""
# --- Assembly ---
assembly = mdl.rootAssembly
assembly.Instance(name='PartA', part=partA, dependent=ON)
assembly.Instance(name='PartB', part=partB, dependent=ON)
assembly.translate(instanceList=('PartB',), vector=(0.0, {y_split}, 0.0))
"""

    def _surfaces(self, is_damaged):
        y_split = self.p['_y_split']
        L, W = self.p['plate_L'], self.p['plate_W']
        return f"""
# --- Interface Surfaces ---
instA = assembly.instances['PartA']
instB = assembly.instances['PartB']
facesA = instA.faces.getByBoundingBox(xMin=-0.1, yMin={y_split-0.01}, zMin=-0.1, xMax={L+0.1}, yMax={y_split+0.01}, zMax={W+0.1})
assembly.Surface(side1Faces=facesA, name='SurfA_Top')
facesB = instB.faces.getByBoundingBox(xMin=-0.1, yMin={y_split-0.01}, zMin=-0.1, xMax={L+0.1}, yMax={y_split+0.01}, zMax={W+0.1})
assembly.Surface(side1Faces=facesB, name='SurfB_Bot')
"""

    def _constraints(self, is_damaged):
        if not is_damaged:
            return """
# --- Healthy Model (Full Tie) ---
mdl.Tie(name='Tie_Full', master=assembly.surfaces['SurfA_Top'], slave=assembly.surfaces['SurfB_Bot'])
"""
        else:
            damages = self.p['damages']
            y_split = self.p['_y_split']
            
            lines = [f"""
# --- Damaged Model (Multiple Delaminations) ---
allA = assembly.surfaces['SurfA_Top'].faces
allB = assembly.surfaces['SurfB_Bot'].faces

# We will build a list of all delamination face indices to exclude them from the healthy tie
delam_indices_A = []
delam_indices_B = []
"""]

            # Shared interaction property
            lines.append(f"""
if 'IntProp_Delam' not in mdl.interactionProperties.keys():
    mdl.ContactProperty('IntProp_Delam')
    mdl.interactionProperties['IntProp_Delam'].TangentialBehavior(formulation=FRICTIONLESS)
    mdl.interactionProperties['IntProp_Delam'].NormalBehavior(pressureOverclosure=HARD, allowSeparation=ON)
""")

            for i, d in enumerate(damages):
                cx, cz = d['center_x'], d['center_z']
                sx, sz = d['size_x'], d['size_z']
                name = f"Delam_{i+1}"
                
                lines.append(f"""
# --- Damage {i+1} ---
# Center: ({cx}, {cz}), Size: {sx}x{sz}
# Bounds: X=[{cx-sx/2.0}, {cx+sx/2.0}], Z=[{cz-sz/2.0}, {cz+sz/2.0}]
cx, cz, sx, sz = {cx}, {cz}, {sx}, {sz}
idxA = [f.index for f in allA if ({cx-sx/2.0-0.1}<f.pointOn[0][0]<{cx+sx/2.0+0.1} and {cz-sz/2.0-0.1}<f.pointOn[0][2]<{cz+sz/2.0+0.1})]
if idxA:
    delam_indices_A.extend(idxA)
    fcsA = instA.faces[idxA[0]:idxA[0]+1]
    for idx in idxA[1:]: fcsA += instA.faces[idx:idx+1]
    assembly.Surface(side1Faces=fcsA, name='SurfA_{name}')

idxB = [f.index for f in allB if ({cx-sx/2.0-0.1}<f.pointOn[0][0]<{cx+sx/2.0+0.1} and {cz-sz/2.0-0.1}<f.pointOn[0][2]<{cz+sz/2.0+0.1})]
if idxB:
    delam_indices_B.extend(idxB)
    fcsB = instB.faces[idxB[0]:idxB[0]+1]
    for idx in idxB[1:]: fcsB += instB.faces[idx:idx+1]
    assembly.Surface(side1Faces=fcsB, name='SurfB_{name}')

if 'SurfA_{name}' in assembly.surfaces.keys() and 'SurfB_{name}' in assembly.surfaces.keys():
    mdl.SurfaceToSurfaceContactExp(name='Contact_{name}', createStepName='Initial', 
        master=assembly.surfaces['SurfA_{name}'], slave=assembly.surfaces['SurfB_{name}'], 
        sliding=FINITE, interactionProperty='IntProp_Delam', mechanicalConstraint=PENALTY)
""")

            # Define healthy zone (all faces NOT in any delam)
            lines.append(f"""
# --- Healthy Zone Tie ---
# Ensure indices are unique and valid
delam_indices_A = list(set(delam_indices_A))
delam_indices_B = list(set(delam_indices_B))

hA_idx = [f.index for f in allA if f.index not in delam_indices_A]
if hA_idx:
    hA_fcs = instA.faces[hA_idx[0]:hA_idx[0]+1]
    for idx in hA_idx[1:]: hA_fcs += instA.faces[idx:idx+1]
    assembly.Surface(side1Faces=hA_fcs, name='SurfA_Healthy')

hB_idx = [f.index for f in allB if f.index not in delam_indices_B]
if hB_idx:
    hB_fcs = instB.faces[hB_idx[0]:hB_idx[0]+1]
    for idx in hB_idx[1:]: hB_fcs += instB.faces[idx:idx+1]
    assembly.Surface(side1Faces=hB_fcs, name='SurfB_Healthy')

if 'SurfA_Healthy' in assembly.surfaces.keys() and 'SurfB_Healthy' in assembly.surfaces.keys():
    mdl.Tie(name='Tie_Healthy', master=assembly.surfaces['SurfA_Healthy'], slave=assembly.surfaces['SurfB_Healthy'],
            positionToleranceMethod=COMPUTED, adjust=ON, tieRotations=ON, thickness=ON, constraintEnforcement=SURFACE_TO_SURFACE)
""")
            return "\n".join(lines)

    def _step(self):
        return f"""
# --- Analysis Step ---
mdl.ExplicitDynamicsStep(name='Step-1', previous='Initial', timePeriod={self.p['t_sim']*1e-6}, maxIncrement={self.p['dt']})
"""

    def _amplitude(self):
        # We assume the user wants the tone burst signal
        return f"""
# --- Tone Burst Amplitude ---
freq = {self.p['freq_khz']} * 1000.0
cycles = {self.p['cycles']}
t1 = cycles / freq
amp = {self.p['amp_mm']}
# Generate tabular data points for the Hanning-windowed signal
points = []
for i in range(201):
    t = i * (t1 / 200.0)
    # Displacement = -A/2 * (1 - cos(2*pi*f/N * t)) * sin(2*pi*f * t)
    # The negative sign applies a downward pulse initially (standard for U2 excitation)
    val = -amp * 0.5 * (1 - math.cos(2*math.pi*freq*t/cycles)) * math.sin(2*math.pi*freq*t)
    points.append((t, val))
points.append((t1*1.01, 0.0))
mdl.TabularAmplitude(name='ToneBurst', data=points, smooth=SOLVER_DEFAULT)
"""

    def _actuator(self):
        ax, az = self.p['actuator_x'], self.p['actuator_z']
        r = self.p['actuator_r']
        amp = self.p['amp_mm']
        return f"""
# --- Actuator (Excitation BC) ---
# Select nodes on bottom surface within radius
nodes = instA.nodes.getByBoundingSphere(center=({ax}, 0.0, {az}), radius={r})
node_labels = [n.label for n in nodes]
if node_labels:
    actSet = assembly.Set(nodes=instA.nodes.sequenceFromLabels(node_labels), name='Set_Actuator')
    mdl.DisplacementBC(name='Excitation', createStepName='Step-1', region=actSet, 
        u1=UNSET, u2=1.0, u3=UNSET, amplitude='ToneBurst')
    mdl.boundaryConditions['Excitation'].setValues(u2={amp})
else:
    print('  [WARNING] No nodes found for actuator at ({ax}, {az})')
"""

    def _sensors(self):
        nx = self.p['sensor_nx']
        sz = self.p['sensor_sz'] 
        sx = self.p['sensor_sx'] 
        spacing = self.p['sensor_spacing']
        
        lines = ["# --- Sensors (History Output) ---"]
        for i in range(nx):
            z_curr = sz + i * spacing
            s_name = f'Sensor_{i+1}'
            lines.append(f"""
node = instA.nodes.getClosest(coordinates=(({sx}, 0.0, {z_curr}),))[0]
setName = '{s_name}'
assembly.Set(nodes=instA.nodes.sequenceFromLabels([node.label]), name=setName)
mdl.HistoryOutputRequest(name='H-Output_' + setName, createStepName='Step-1', 
    variables=('U2',), region=assembly.sets[setName], frequency=1)
""")
        return "\n".join(lines)

    def _boundary_conditions(self):
        L, W = self.p['plate_L'], self.p['plate_W']
        return f"""
# --- Boundary Conditions (Soft Corners for Stability) ---
# Pin the 4 corners of the bottom surface to prevent rigid body motion
# while allowing free-free plate vibration.
corners = [(0.0, 0.0, 0.0), ({L}, 0.0, 0.0), (0.0, 0.0, {W}), ({L}, 0.0, {W})]
corner_labels = []
for cx, cy, cz in corners:
    node = instA.nodes.getClosest(coordinates=((cx, cy, cz),))[0]
    corner_labels.append(node.label)

if corner_labels:
    region = regionToolset.Region(nodes=instA.nodes.sequenceFromLabels(corner_labels))
    mdl.DisplacementBC(name='Stability_SoftSupport', createStepName='Initial', 
        region=region, u1=0.0, u2=0.0, u3=0.0)
"""

    def _field_output(self):
        intrvl = self.p.get('field_intval', 200)
        return f"""
# --- Field Output ---
mdl.fieldOutputRequests['F-Output-1'].setValues(numIntervals={intrvl}, variables=('U', 'S'))
"""

    def _job(self):
        return f"""
# --- Job Creation ---
# Precision=DOUBLE_PLUS_PACK is recommended for wave propagation to reduce numerical dispersion.
# ResultsFormat=ODB ensures compatibility with standard post-processing.
mdb.Job(name=modelName, model=modelName, type=ANALYSIS, resultsFormat=ODB,
    explicitPrecision=DOUBLE_PLUS_PACK, nodalOutputPrecision=FULL)
"""

    def _footer(self):
        return """
print('Done!')
"""

    def _write_script(self, filename, content):
        with open(filename, 'w') as f:
            f.write(content)
        return filename

