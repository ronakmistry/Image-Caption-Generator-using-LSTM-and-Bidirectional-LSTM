"""Download the Flickr8k mirror and official Keras MobileNet weights for a local run."""
from pathlib import Path
import hashlib
import json
import time
import urllib.request
import zipfile

root = Path('data/flickr8k')
root.mkdir(parents=True, exist_ok=True)
archives = root / 'downloads'
archives.mkdir(exist_ok=True)
sources = {
    'Flickr8k_Dataset.zip': 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip',
    'Flickr8k_text.zip': 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip',
    'mobilenet_1_0_224_tf_no_top.h5': 'https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_1_0_224_tf_no_top.h5',
}
provenance = {}
for filename, url in sources.items():
    target = archives / filename
    start = time.perf_counter()
    if not target.exists():
        temporary = target.with_suffix(target.suffix + '.part')
        print('Downloading', filename, flush=True)
        with urllib.request.urlopen(url, timeout=90) as response, temporary.open('wb') as stream:
            received, previous = 0, 0
            while chunk := response.read(1024 * 1024):
                stream.write(chunk); received += len(chunk)
                if received - previous >= 64 * 1024**2:
                    print(f'{filename}: {received / 1024**2:.0f} MiB', flush=True); previous = received
        temporary.rename(target)
    digest = hashlib.sha256()
    with target.open('rb') as stream:
        while chunk := stream.read(1024 * 1024): digest.update(chunk)
    provenance[filename] = {'source': url, 'bytes': target.stat().st_size,
                            'sha256': digest.hexdigest(), 'preparation_seconds': time.perf_counter() - start}
    if target.suffix == '.zip':
        print('Extracting', filename, flush=True)
        with zipfile.ZipFile(target) as archive:
            for member in archive.infolist():
                source = Path(member.filename)
                if member.is_dir() or '__MACOSX' in source.parts or source.name.startswith('.'):
                    continue
                if filename == 'Flickr8k_Dataset.zip':
                    if source.suffix.lower() != '.jpg': continue
                    destination = root / 'Flicker8k_Dataset' / source.name
                else:
                    destination = root / 'Flickr8k_text' / source.name
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(archive.read(member))
    print(filename, 'ready', flush=True)
(root / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
print('Images:', len(list((root / 'Flicker8k_Dataset').glob('*.jpg'))), flush=True)
