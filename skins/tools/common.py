"""Shared helpers for the Jumper skin builds (Ballerina, Matterhorn)."""
import json
from pathlib import Path

import numpy as np
import trimesh

DESIGN = Path(__file__).resolve().parents[2] / 'jumper-design'
PROFILE = DESIGN / 'robots/jumper/profile.json'


def profile():
    return json.loads(PROFILE.read_text(encoding='utf-8'))


def original_shell_cad_mm():
    """Return the stock upper-shell visual mesh in the CAD millimeter frame."""
    import xml.etree.ElementTree as ET
    prof = profile()
    xml = ET.parse(PROFILE.parent / prof['baseline_mjcf']).getroot()
    geom = xml.find(".//geom[@name='" + prof['visual_replacement']['geom_name'] + "']")
    asset = xml.find("asset/mesh[@name='" + geom.get('mesh') + "']")
    source = PROFILE.parent / xml.find('compiler').get('meshdir') / asset.get('file')
    mesh = trimesh.load_mesh(source, process=False)
    t = prof['cad_mm_to_parent_m']
    mesh.vertices = ((mesh.vertices - np.asarray(t['translation_m']))
                     @ np.asarray(t['rotation_matrix'])) / t['scale']
    return mesh, prof


def seam_points():
    prof = profile()
    seam = json.loads((PROFILE.parent / prof['seam_reference']).read_text(encoding='utf-8'))
    return np.asarray(seam['points'], dtype=float)
