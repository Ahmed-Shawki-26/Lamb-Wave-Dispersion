# -*- coding: utf-8 -*-
"""
run_generator.py
================
Runner script that generates all 12 Abaqus Python scripts:
  3 materials × 2 configs × 2 models (healthy + damaged) = 12 scripts

Output structure:
  abaqus_models/
  ├── T300_5208/
  │   ├── healthy_QI.py
  │   ├── healthy_UD.py
  │   ├── damaged_QI_stage1.py
  │   └── damaged_UD_stage1.py
  ├── T300_914/
  │   ├── healthy_QI.py
  │   ├── ...
  └── AS4_3501_6/
      ├── ...

Usage:
  python run_generator.py
"""

import os
import sys
import time

# Ensure the scripts directory is in the path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from abaqus_generator import AbaqusScriptGenerator
from material_config import MATERIALS, LAYUP_CONFIGS


def main():
    print('='*70)
    print('ABAQUS SCRIPT GENERATOR - Lamb Wave Delamination Detection')
    print('Graduation Project')
    print('='*70)
    print()
    print(f'Materials:      {len(MATERIALS)} ({", ".join(MATERIALS.keys())})')
    print(f'Configurations: {len(LAYUP_CONFIGS)} ({", ".join(LAYUP_CONFIGS.keys())})')
    print(f'Models per combo: 2 (healthy + damaged)')
    print(f'Total scripts:  {len(MATERIALS) * len(LAYUP_CONFIGS) * 2}')
    print()

    # Output directory: abaqus_models/ at project root level
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, 'abaqus_models')

    generator = AbaqusScriptGenerator(output_dir=output_dir)

    start_time = time.time()

    # ---- Generate all scripts ----
    for mat_name in MATERIALS:
        mat = MATERIALS[mat_name]
        print(f'\n{"-"*70}')
        print(f'Material: {mat["display_name"]} ({mat_name})')
        print(f'{"-"*70}')

        for config in LAYUP_CONFIGS:
            layup = LAYUP_CONFIGS[config]
            print(f'\n  Configuration: {layup["display_name"]}')

            # Healthy model
            generator.generate_healthy(mat_name, config)

            # Damaged model (Stage 1)
            generator.generate_damaged(mat_name, config, delam_stage='stage1')

    elapsed = time.time() - start_time

    # ---- Print summary ----
    generator.print_summary()
    print(f'\nTime elapsed: {elapsed:.2f} seconds')
    print()

    # ---- Print tone burst CSV reminder ----
    print('='*70)
    print('IMPORTANT: Tone Burst CSV Paths')
    print('='*70)
    print()
    print('Each generated script loads tone burst data from a CSV file.')
    print('If running Abaqus on a different machine, update the csvPath')
    print('variable in each script to point to the correct absolute path.')
    print()
    print('Current CSV paths (relative to project root):')
    for mat_name, mat in MATERIALS.items():
        csv_file = mat['tone_burst_csv']
        full_path = os.path.join(project_root, csv_file)
        exists = os.path.exists(full_path)
        status = 'OK found' if exists else 'MISSING'
        print(f'  {mat_name}: {csv_file}  [{status}]')
    print()
    print('='*70)


if __name__ == '__main__':
    main()
