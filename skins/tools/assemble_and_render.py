"""Assemble a work/shell.stl (+ ams.3mf, visual-overrides.json) on the whole robot and render views.

usage: assemble_and_render.py <skin-dir> <version-label>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import DESIGN, PROFILE
sys.path.insert(0, str(DESIGN / 'src'))
from shellflow.simulation import assemble
sys.path.append(str(DESIGN / 'scripts'))   # appended: scripts/shellflow.py must not shadow the package
from export_skin_collection import render_views

skin = Path(sys.argv[1]).resolve()
label = sys.argv[2]
faces = int(sys.argv[3]) if len(sys.argv) > 3 else 100000   # raise to skip decimation of a painted, subdivided shell
work = skin / 'work'
out = skin / 'assembly' / label
overrides = json.loads((work / 'visual-overrides.json').read_text())
result = assemble(work / 'shell.stl', out, PROFILE, ams3mf=work / 'ams.3mf',
                  visual_overrides=overrides, faces=faces)
report = json.loads((out / 'simulation-report.json').read_text())
render_views(out, skin / 'renders' / label)
print(json.dumps({'output': result['output'], 'colors': report['colors']['mode'],
                  'seam': report['seam_registration'], 'whole_robot': report['whole_robot'].get('passed'),
                  'urdf': report['urdf_verification']['passed'],
                  'faces': report['simplification']['output_faces']}, indent=1))
