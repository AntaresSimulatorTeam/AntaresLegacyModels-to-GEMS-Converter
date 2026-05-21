from pathlib import Path
import shutil
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, 'src')
from antares.craft.model.study import read_study_local

src_study = Path(r"C:\Users\jeannecor\Documents\1-PROJECTS\OPEN SOURCE\BP25")
dest_root = Path(r"C:\Users\jeannecor\Documents\1-PROJECTS\OPEN SOURCE\tmp_delete_test")
if dest_root.exists():
    shutil.rmtree(dest_root)
copy_dest = dest_root / 'BP25'
shutil.copytree(src_study, copy_dest)
study = read_study_local(copy_dest)
print('binding count before', len(study.get_binding_constraints()))
constraint = study.get_binding_constraints()['p2g_fatalband_se1']
study.delete_binding_constraints([constraint])
print('binding count after delete call', len(study.get_binding_constraints()))
file_path = copy_dest / 'input' / 'bindingconstraints' / 'p2g_fatalband_se1_gt.txt'
print('file exists after delete call', file_path.exists(), file_path)
