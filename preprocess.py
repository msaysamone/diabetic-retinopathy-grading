from pathlib import Path

# ---- config ----
KAGGLE_COMPETITION = "diabetic-retinopathy-detection"
KAGGLE_ARCHIVES = {     # joined zip name -> its split parts, in order
    "train.zip": [f"train.zip.{i:03d}" for i in range(1, 6)],
}
KAGGLE_EXTRAS = ["trainLabels.csv.zip"]   # small files, unzipped into DATA
KAGGLE_DATASETS = [     # Kaggle datasets (owner/name); their CSVs are extracted to DATA/<name>/
    "mariaherrerot/aptos2019",   # APTOS 2019 with labelled train / valid / test splits
]

URLS = [
    # "https://example.com/dataset/part1.zip",
    # "https://example.com/dataset/part2.zip",
]

SIZE = 512              # output is SIZE x SIZE (fundus crop -> pad to square -> resize)
FORMAT = "PNG"          # "PNG" (lossless), "JPEG" or "WEBP"
QUALITY = 95            # only used for JPEG / WEBP
BLACK_THRESHOLD = 10    # grayscale value (0-255) at or below which a pixel counts as background
MIN_FILL = 0.01         # a row/column is fundus only if > this fraction of its pixels are above the threshold
DELETE_ZIP_AFTER = True  # delete the zip / its parts once processed with no read errors
MIN_FREE_GB = 5         # stop before downloading if D: has less free space than this
WORKERS = None          # None = one per CPU core

DATA = Path("data")     # symlink to /mnt/d/datasets/prototype
ZIPS = DATA / "zips"
OUT = DATA / "processed"
DONE_FILE = DATA / "done.txt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

import bisect, csv, io, os, re, shutil, subprocess, zipfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from urllib.parse import urlparse, unquote

import numpy as np
import requests
from PIL import Image, ImageOps
from tqdm.auto import tqdm

EXT = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}[FORMAT]
SAVE_KW = {} if FORMAT == "PNG" else {"quality": QUALITY}


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def zip_name(url):
    return Path(unquote(urlparse(url).path)).name


def download(url, dest):
    """Download with resume support. Local paths are copied."""
    if dest.exists():
        return dest
    if not re.match(r"https?://", url):
        shutil.copy(url, dest)
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if r.status_code == 416:          # already fully downloaded
            part.rename(dest)
            return dest
        r.raise_for_status()
        if have and r.status_code != 206:  # server ignored resume; start over
            have = 0
        total = int(r.headers.get("content-length", 0)) + have
        with open(part, "ab" if have else "wb") as f, tqdm(
            total=total, initial=have, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    part.rename(dest)
    return dest


KAGGLE = shutil.which("kaggle") or str(Path.home() / ".local/bin/kaggle")


def kaggle_sizes(competition):
    """File name -> size in bytes for every file in the competition."""
    out = subprocess.run([KAGGLE, "competitions", "files", "-c", competition, "-v"],
                         capture_output=True, text=True, check=True).stdout
    lines = out.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("name,size"))
    return {row["name"]: int(row["size"]) for row in csv.DictReader(lines[start:])}


def kaggle_download(competition, name, size):
    """Download one competition file into ZIPS unless it's already there at the right size."""
    dest = ZIPS / name
    if dest.exists() and dest.stat().st_size == size:
        return dest
    cmd = [KAGGLE, "competitions", "download", "-c", competition, "-f", name, "-p", str(ZIPS)]
    if dest.exists():                     # incomplete from an earlier run
        cmd.append("-o")
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        for chunk in iter(lambda: proc.stdout.read(256), ""):
            print(chunk, end="")
    if proc.returncode:
        raise RuntimeError(f"kaggle download of {name} failed (exit {proc.returncode}). "
                           "Have you accepted the competition rules on Kaggle?")
    wrapped = ZIPS / (name + ".zip")      # Kaggle sometimes wraps a file in a zip
    if not dest.exists() and wrapped.exists():
        with zipfile.ZipFile(wrapped) as zf:
            zf.extract(name, ZIPS)
        wrapped.unlink()
    if not dest.exists() or dest.stat().st_size != size:
        got = dest.stat().st_size if dest.exists() else 0
        raise RuntimeError(f"{name} is {got} bytes, expected {size}")
    return dest


class MultiPartFile(io.RawIOBase):
    """Read-only file that presents split parts (.001, .002, ...) as one continuous file."""

    def __init__(self, parts):
        self.files = [open(p, "rb") for p in parts]
        self.starts, pos = [], 0
        for p in parts:
            self.starts.append(pos)
            pos += Path(p).stat().st_size
        self.size, self.pos = pos, 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def readinto(self, buf):
        n = 0
        view = memoryview(buf)
        while n < len(view) and self.pos < self.size:
            i = bisect.bisect_right(self.starts, self.pos) - 1
            f = self.files[i]
            f.seek(self.pos - self.starts[i])
            got = f.readinto(view[n:])
            if not got:
                break
            n += got
            self.pos += got
        return n

    def close(self):
        for f in self.files:
            f.close()
        super().close()


def finish_zip(name, sources, done):
    """Process a zip (one file, or its split parts), record it as done, optionally delete it.

    If any image failed its checksum, the zip is kept and not marked done, so a re-run
    retries it (images that already processed are skipped).
    """
    errors, read_errors = process_zip(name, sources)
    if errors:
        print(f"{len(errors)} images failed in {name}, e.g.:", *errors[:5], sep="\n  ")
    if read_errors:
        print(f"{name}: {len(read_errors)} images failed their checksum, so the download is "
              "probably corrupt. Kept the files and didn't mark it done; delete the bad "
              "part(s) and re-run to download them again.")
        return
    with open(DONE_FILE, "a") as f:
        f.write(name + "\n")
    done.add(name)
    if DELETE_ZIP_AFTER:
        for p in sources:
            p.unlink()
    print(f"done {name} — {free_gb(DATA):.0f} GB free on D:")


def fundus_crop(im):
    """Crop to the bounding box of the fundus, dropping the black border.

    Rows/columns need more than MIN_FILL of their pixels above BLACK_THRESHOLD, so
    text labels or noise in the border don't widen the crop.
    """
    mask = np.asarray(im.convert("L")) > BLACK_THRESHOLD
    rows = np.where(mask.mean(axis=1) > MIN_FILL)[0]
    cols = np.where(mask.mean(axis=0) > MIN_FILL)[0]
    if len(rows) == 0 or len(cols) == 0:   # (nearly) all black: leave as is
        return im
    return im.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))


def pad_to_square(im):
    """Center the image on a black square canvas."""
    side = max(im.size)
    canvas = Image.new("RGB", (side, side))
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    return canvas


def process_image(args):
    """Fundus crop, pad to square, resize, save. Returns None or an error string."""
    data, out_path = args
    try:
        with Image.open(io.BytesIO(data)) as im:
            # Decode big JPEGs at a reduced scale (1/2, 1/4, 1/8), staying at least 2x SIZE on each side.
            im.draft("RGB", (2 * SIZE, 2 * SIZE))
            im = ImageOps.exif_transpose(im).convert("RGB")
            im = pad_to_square(fundus_crop(im))
            im = im.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            im.save(out_path, FORMAT, **SAVE_KW)
        return None
    except Exception as e:
        return f"{out_path.name}: {e}"


def iter_jobs(zf, out_dir, read_errors):
    for info in zf.infolist():
        p = Path(info.filename)
        if info.is_dir() or p.suffix.lower() not in IMAGE_EXTS or p.name.startswith("._"):
            continue
        out_path = out_dir / p.with_suffix(EXT)
        if out_path.exists():             # already done on a previous run
            continue
        try:
            data = zf.read(info)          # also checks the CRC
        except (zipfile.BadZipFile, EOFError, OSError) as e:
            read_errors.append(f"{info.filename}: {e}")
            continue
        yield data, out_path


def process_zip(name, sources):
    """Returns (image errors, read/checksum errors)."""
    out_dir = OUT / Path(name).stem
    src = sources[0] if len(sources) == 1 else io.BufferedReader(MultiPartFile(sources), 1 << 20)
    try:
        zf = zipfile.ZipFile(src)
    except zipfile.BadZipFile:
        raise RuntimeError(
            f"{name} isn't a valid zip. If it's one part of a split archive, "
            "list all its parts in KAGGLE_ARCHIVES."
        )
    with zf:
        n = sum(1 for i in zf.infolist() if Path(i.filename).suffix.lower() in IMAGE_EXTS)
        errors, read_errors = [], []
        bar = tqdm(total=n, desc=f"process {name}")
        with ProcessPoolExecutor(WORKERS) as pool:
            # Keep a limited number of images in flight so a big zip isn't read into RAM all at once.
            max_pending = 4 * (WORKERS or os.cpu_count())
            pending = set()
            for job in iter_jobs(zf, out_dir, read_errors):
                pending.add(pool.submit(process_image, job))
                if len(pending) >= max_pending:
                    finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                    errors += [f.result() for f in finished if f.result()]
                    bar.update(len(finished))
            for f in pending:
                if f.result():
                    errors.append(f.result())
            bar.update(len(pending))
        bar.close()
    return errors, read_errors

for d in (ZIPS, OUT):
    d.mkdir(parents=True, exist_ok=True)
done = set(DONE_FILE.read_text().split()) if DONE_FILE.exists() else set()

# ---- Kaggle split archives ----
if KAGGLE_ARCHIVES or KAGGLE_EXTRAS:
    sizes = kaggle_sizes(KAGGLE_COMPETITION)

for name in KAGGLE_EXTRAS:
    target = DATA / name.removesuffix(".zip")
    if not target.exists():
        with zipfile.ZipFile(kaggle_download(KAGGLE_COMPETITION, name, sizes[name])) as zf:
            zf.extractall(DATA)
        print(f"extracted {name} into {DATA}")

for joined_name, part_names in KAGGLE_ARCHIVES.items():
    if joined_name in done:
        print(f"skip {joined_name} (already done)")
        continue
    missing = sum(sizes[n] for n in part_names
                  if not ((ZIPS / n).exists() and (ZIPS / n).stat().st_size == sizes[n]))
    need_gb = missing / 1e9 + MIN_FREE_GB
    if free_gb(DATA) < need_gb:
        raise RuntimeError(f"{joined_name} needs ~{need_gb:.0f} GB free, D: has {free_gb(DATA):.0f} GB.")
    parts = [kaggle_download(KAGGLE_COMPETITION, n, sizes[n]) for n in part_names]
    finish_zip(joined_name, parts, done)

# ---- Kaggle datasets ----
for ds in KAGGLE_DATASETS:
    name = ds.split("/")[1] + ".zip"
    if name in done:
        print(f"skip {name} (already done)")
        continue
    zip_path = ZIPS / name
    if not zip_path.exists():
        subprocess.run([KAGGLE, "datasets", "download", "-d", ds, "-p", str(ZIPS)], check=True)
    with zipfile.ZipFile(zip_path) as zf:       # keep the label / split CSVs
        for member in zf.namelist():
            if member.lower().endswith(".csv"):
                zf.extract(member, DATA / zip_path.stem)
    finish_zip(name, [zip_path], done)

# ---- standalone zips ----
for url in URLS:
    name = zip_name(url)
    if name in done:
        print(f"skip {name} (already done)")
        continue
    if free_gb(DATA) < MIN_FREE_GB:
        raise RuntimeError(f"Only {free_gb(DATA):.1f} GB free on D:, stopping.")

    finish_zip(name, [download(url, ZIPS / name)], done)
