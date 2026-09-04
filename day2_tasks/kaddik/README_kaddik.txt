KADID-10k manifest outputs

Dataset root: /data/projectwork/swati_mam/kadid10k
Output directory: /home/projectwork/student_package/day2_tasks/kaddik

Created files:
- kadid10k_manifest.csv: exact requested columns
- kadid10k_manifest_extended.csv: requested columns plus filename/code/variance audit columns
- kadid10k_distortion_code_map.csv: distortion code to distortion type
- kadid10k_manifest_summary.json: count and validation summary

Requested manifest columns:
dataset, ref_id, distorted_path, ref_path, distortion_type, severity_or_level, mos_or_dmos

Filename parsing:
I01_03_05.png -> ref_id=I01, distortion_code=03, severity_or_level=5.
mos_or_dmos is DMOS from the local dmos.csv.

No image files were modified.
