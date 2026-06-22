import json
import trimesh
import numpy as np

with open('SyntheticPipeline/configs/pipeline_config.json') as f:
    config = json.load(f)
print('Config wind:', config.get('simulation', {}).get('wind_x'), config.get('simulation', {}).get('wind_y'))
