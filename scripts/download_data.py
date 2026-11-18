"""Download the audio from its public source and check it against data/manifest.csv.

    python scripts/download_data.py                  # into data/audio, about 1.5 GB download, 2.2 GB on disk
    python scripts/download_data.py --check-only     # only compare an existing folder with the manifest

The repository contains no audio. The archive holds 800 WAV files of the MLEnd Hums and Whistles II
data set (see "Data" in the README).
"""

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile

from _common import ROOT, log

from humrec import config, data


def download(url, target, attempts=3):
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, open(target + ".part", "wb") as handle:
                total = int(response.headers.get("Content-Length", 0))
                done, shown = 0, 0
                while True:
                    block = response.read(1 << 22)
                    if not block:
                        break
                    handle.write(block)
                    done += len(block)
                    if done - shown >= 200 * (1 << 20):
                        shown = done
                        log(f"downloaded {done / 1e6:.0f} of {total / 1e6:.0f} MB")
            if total and done != total:
                raise IOError(f"incomplete download: {done} of {total} bytes")
            os.replace(target + ".part", target)
            return
        except Exception as error:  # network errors are retried
            log(f"attempt {attempt} failed: {error}")
    raise SystemExit("download failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=os.path.join(ROOT, "data", "audio"))
    parser.add_argument("--url", default=config.DATA_URL)
    parser.add_argument("--manifest", default=os.path.join(ROOT, "data", "manifest.csv"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()

    if not args.check_only:
        os.makedirs(args.data_dir, exist_ok=True)
        archive = os.path.join(args.data_dir, "archive.zip")
        if not os.path.exists(archive):
            log(f"downloading {args.url}")
            download(args.url, archive)
        log("unpacking")
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                name = os.path.basename(member.filename)
                if member.is_dir() or not name.lower().endswith(".wav") or name.startswith("._"):
                    continue
                with bundle.open(member) as source, open(os.path.join(args.data_dir, name), "wb") as target:
                    shutil.copyfileobj(source, target)
        if not args.keep_zip:
            os.remove(archive)

    n_files = len(data.find_wav_files(args.data_dir))
    log(f"{n_files} WAV files in {args.data_dir}")
    problems = data.check_against_manifest(args.data_dir, args.manifest)
    if problems:
        for line in problems[:20]:
            print("  " + line)
        sys.exit(f"{len(problems)} differences from {os.path.relpath(args.manifest, ROOT)}")
    log("all files match the manifest (names, sizes and SHA-256)")


if __name__ == "__main__":
    main()
