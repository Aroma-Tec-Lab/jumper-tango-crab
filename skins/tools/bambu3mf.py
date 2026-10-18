"""Write a minimal Bambu-style painted 3MF that shellflow's AMS reader accepts."""
import json
import zipfile

import numpy as np

# shellflow PAINT_CODES: slot 0..3
PAINT = ['4', '8', '0C', '1C']
NS = 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'
NS_P = 'http://schemas.microsoft.com/3dmanufacturing/production/2015/06'


def write(path, mesh, face_slots, palette):
    if not 1 <= len(palette) <= 4 or len(face_slots) != len(mesh.faces):
        raise ValueError('palette must have 1-4 colors and one slot per face')
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             f'<model unit="millimeter" xmlns="{NS}"><resources><object id="1" type="model">'
             '<mesh><vertices>']
    lines += [f'<vertex x="{x!r}" y="{y!r}" z="{z!r}"/>' for x, y, z in vertices.tolist()]
    lines.append('</vertices><triangles>')
    lines += [f'<triangle v1="{a}" v2="{b}" v3="{c}" paint_color="{PAINT[s]}"/>'
              for (a, b, c), s in zip(np.asarray(mesh.faces).tolist(), face_slots.tolist())]
    lines.append('</triangles></mesh></object></resources></model>')
    main = (f'<?xml version="1.0" encoding="UTF-8"?><model unit="millimeter" xmlns="{NS}" '
            f'xmlns:p="{NS_P}"><resources><object id="2" type="model"><components>'
            '<component p:path="/3D/Objects/object_1.model" objectid="1" '
            'transform="1 0 0 0 1 0 0 0 1 0 0 0"/></components></object></resources>'
            '<build><item objectid="2"/></build></model>')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('3D/3dmodel.model', main)
        z.writestr('3D/Objects/object_1.model', '\n'.join(lines))
        z.writestr('Metadata/project_settings.config',
                   json.dumps({'filament_colour': [c.upper() for c in palette]}))
