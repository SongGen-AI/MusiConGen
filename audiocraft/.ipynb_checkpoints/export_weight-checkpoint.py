from audiocraft.utils import export
from audiocraft import train
import os
from pathlib import Path

sig = "88239d54"
output_dir = "./ckpt/sample100"


folder = f"./training_weights/xps/{sig}"
export.export_lm(Path(folder) / 'checkpoint.th', os.path.join(output_dir, 'state_dict.bin'))
export.export_pretrained_compression_model('facebook/encodec_32khz', os.path.join(output_dir, 'compression_state_dict.bin'))