from pathlib import Path
import sys
sys.path.insert(0, "src")
from antares.craft.model.study import read_study_local

study_path = Path(r"C:\Users\jeannecor\Documents\1-PROJECTS\OPEN SOURCE\BP25")
study = read_study_local(study_path)

binding_constraints = study.get_binding_constraints()
print('binding constraints count', len(binding_constraints))
match = [name for name in binding_constraints if 'p2g_fatalband' in name]
print('p2g names', sorted(match)[:50])
print('contains se1', any('p2g_fatalband_se1' in name for name in binding_constraints))
print('contains se1_gt', any('p2g_fatalband_se1_gt' in name for name in binding_constraints))
print('example names first 20', sorted(list(binding_constraints.keys()))[:20])
