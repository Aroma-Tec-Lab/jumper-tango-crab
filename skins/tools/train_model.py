"""Build a training-repo model with a skin's visuals: physics, joints and limits stay the Train asset's.

usage: train_model.py <assembly-dir> <out.xml>
Only visual geoms change: the upper-shell visual is replaced by the skin's colored parts and
limb/lower-shell visuals take the skin's declared colors. Collision, inertia and joints are untouched.
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROBOT = Path(__file__).resolve().parents[2]
TRAIN_XML = ROBOT / 'jumper/assets/jumper/jumper.xml'

assembly, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
report = json.loads((assembly / 'simulation-report.json').read_text())
skin = ET.parse(assembly / 'robot.xml').getroot()
skin_meshdir = assembly / skin.find('compiler').get('meshdir')
train = ET.parse(TRAIN_XML).getroot()
train.find('compiler').set('meshdir', str(TRAIN_XML.parent) + '/')
assets = train.find('asset')
upper = train.find(".//body[@name='upper_shell_link']")
old = upper.find("geom[@name='upper_shell_link_visual']")
upper.remove(old)
skin_upper = skin.find(".//body[@name='upper_shell_link']")
for geom in skin_upper.findall('geom'):
    if not geom.get('name', '').startswith('shellflow_upper_'):
        continue
    mesh = skin.find("asset/mesh[@name='" + geom.get('mesh') + "']")
    ET.SubElement(assets, 'mesh', name=geom.get('mesh'), file=str(skin_meshdir / mesh.get('file')))
    ET.SubElement(upper, 'geom', name=geom.get('name'), type='mesh', mesh=geom.get('mesh'),
                  contype='0', conaffinity='0', density='0', group='2', rgba=geom.get('rgba'))
colored = 0
by_link = {}
for record in report['visual_overrides']:
    by_link[record['link']] = record['rgba']
for body in train.iter('body'):
    rgba = by_link.get(body.get('name'))
    if rgba is None:
        continue
    for geom in body.findall('geom'):
        if geom.get('class') == 'visual':
            geom.set('rgba', ' '.join(map(str, rgba)))
            colored += 1
out.parent.mkdir(parents=True, exist_ok=True)
ET.ElementTree(train).write(out)
import mujoco
m = mujoco.MjModel.from_xml_path(str(out))
print(json.dumps({'model': str(out), 'recolored_visuals': colored, 'njnt': m.njnt,
                  'total_mass': round(float(sum(m.body_mass)), 4)}))
