"""Render every preselected validation example, without choosing favorable cases."""
from pathlib import Path
import os, json, textwrap
os.environ.setdefault('MPLCONFIGDIR', str(Path('.cache/matplotlib').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

root = Path('artifacts/blstm-comparison')
rows = list(json.loads((root / 'preview.json').read_text()).items())
for page in range((len(rows) + 5) // 6):
    fig, axes = plt.subplots(3, 2, figsize=(15, 16))
    for ax, (key, row) in zip(axes.flat, rows[page*6:(page+1)*6]):
        ax.imshow(Image.open(Path('data/flickr8k/Flicker8k_Dataset') / key))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title('\n'.join(textwrap.wrap('LSTM: ' + row['lstm'], 66)) + '\n' +
                     '\n'.join(textwrap.wrap('BLSTM: ' + row['blstm'], 66)), fontsize=11, loc='left')
        ax.set_xlabel('\n'.join(textwrap.wrap('Reference: ' + row['references'][0], 72)), fontsize=10)
    fig.suptitle('LSTM vs BLSTM · validation images\n' +
                 ('Same six images shown previously' if page == 0 else 'Six additional images, sampled with seed 20261021'), fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(root / f'comparison-{page+1}.png', dpi=130)
    plt.close(fig)
print('Rendered', len(rows), 'validation examples')
