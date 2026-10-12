# ===== CELL 1/5 · Setup + check helpers (~10 s) =====
import os, re, sys, time, shutil, tarfile, hashlib, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

RAW_ROOT = Path(os.environ.get("GAIT_RAW", "/content/gait_raw"))   # raw data on Colab disk
IMU_ROOT = Path(os.environ.get("GAIT_IMU", "/content/gait_imu"))   # extracted IMU folders
OUT_ROOT = Path(os.environ.get("GAIT_OUT", "/content/drive/MyDrive/gait_eeg_paper/n0"))  # small outputs on Drive
BASE_URL = "https://physionet.org/files/multimodal-gait-dataset/1.0.0/"
SPEEDS = (0.5, 0.75, 1.0)
N_SUBJECTS = 59


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


FAILED: list = []


def check(ok: bool, msg: str, critical: bool = True) -> bool:
    """✅ pass | ❌ critical fail (the cell stops at the end) | ⚠️ warning (look at it, you can continue)."""
    print(f"{'✅' if ok else ('❌' if critical else '⚠️')} CHECK: {msg}", flush=True)
    if not ok and critical:
        FAILED.append(msg)
    return ok


def end_cell(name: str) -> None:
    """Raise if any critical check failed, so the next cell never runs on bad input."""
    failed = FAILED.copy()
    FAILED.clear()
    if failed:
        raise RuntimeError(f"{name} STOPPED - fix the ❌ items first: {failed}")
    log(f"{name} OK -> run the next cell")


try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    log("not running in Colab -> Drive mount skipped")

RAW_ROOT.mkdir(parents=True, exist_ok=True)
OUT_ROOT.mkdir(parents=True, exist_ok=True)
log(f"python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__}")
free_gb = shutil.disk_usage(RAW_ROOT).free / 1e9
check(free_gb > 15, f"free disk {free_gb:.0f} GB (need > 15 GB)")
probe = OUT_ROOT / "_write_test.txt"
probe.write_text("ok")
check(probe.read_text() == "ok", f"outputs folder is writable: {OUT_ROOT}")
probe.unlink()
end_cell("CELL 1")

# ===== CELL 2/5 · Download from the AWS S3 mirror (~5-15 min) =====
# physionet.org is very slow from Colab (~0.02 MB/s). The same files are on Amazon S3, which is much faster.
# Progress every 15 s. Safe to stop (■) and re-run: complete files are skipped, partial files resume.
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import wait
import requests

S3 = "https://physionet-open.s3.amazonaws.com/"
PREFIX = "multimodal-gait-dataset/1.0.0/"
WORKERS = 8


def list_s3() -> list:
    """All (key, size) pairs under PREFIX, from the public bucket listing."""
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    objects, token = [], None
    while True:
        params = {"list-type": "2", "prefix": PREFIX, **({"continuation-token": token} if token else {})}
        resp = requests.get(S3, params=params, timeout=60)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        objects += [(c.find("s:Key", ns).text, int(c.find("s:Size", ns).text)) for c in root.findall("s:Contents", ns)]
        nxt = root.find("s:NextContinuationToken", ns)
        if nxt is None:
            return objects
        token = nxt.text


def local_path(key: str) -> Path:
    return RAW_ROOT / key[len(PREFIX):]


new_bytes = 0
lock = threading.Lock()
stop = threading.Event()


def fetch(key: str, size: int) -> str:
    """Download one file; a partial file is resumed with an HTTP Range request; 3 tries."""
    global new_bytes
    dest = local_path(key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if have == size:
        return "skip (already complete)"
    headers = {}
    for attempt in range(1, 4):
        if have > size:
            dest.unlink()
            have = 0
        try:
            headers = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(S3 + key, headers=headers, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                if have and resp.status_code != 206:  # range ignored -> write the file from the start
                    have = 0
                with open(dest, "ab" if have else "wb") as fh:
                    for chunk in resp.iter_content(1 << 20):
                        if stop.is_set():
                            return "stopped"
                        fh.write(chunk)
                        with lock:
                            new_bytes += len(chunk)
        except requests.RequestException as exc:
            log(f"retry {attempt}/3 for {dest.name}: {exc}")
            time.sleep(5 * attempt)
        have = dest.stat().st_size if dest.exists() else 0
        if have == size:
            return "resumed" if headers else "downloaded"
    raise IOError(f"{key}: {have} of {size} bytes after 3 tries")


try:
    subprocess.run(["pkill", "-f", BASE_URL], check=False)  # stop the slow wget from the old CELL 2 if still running
except FileNotFoundError:
    pass
objects = list_s3()
if not objects:
    raise RuntimeError("S3 listing is empty -> check the internet connection and re-run")
total = sum(size for _, size in objects)
present = sum(min(local_path(k).stat().st_size, s) for k, s in objects if local_path(k).exists())
log(f"S3 listing: {len(objects)} files, {total / 1e9:.2f} GB | already on disk: {present / 1e9:.2f} GB")

pool = ThreadPoolExecutor(WORKERS)
futures = {pool.submit(fetch, k, s): k for k, s in objects}
t0 = prev_t = last_change = time.time()
prev_new = 0
try:
    while True:
        finished, pending = wait(futures, timeout=15)
        now = time.time()
        with lock:
            got = new_bytes
        rate = (got - prev_new) / 1e6 / max(now - prev_t, 1e-6)
        if got != prev_new:
            last_change = now
        avg = got / max(now - t0, 1e-6)
        eta = f"{max(total - present - got, 0) / avg / 60:4.0f} min" if avg > 0 else "   ? min"
        stall = f"   ⚠️ no growth for {now - last_change:.0f}s" if now - last_change > 120 else ""
        log(f"{len(finished):3d}/{len(objects)} files | {min(present + got, total) / 1e9:5.2f}/{total / 1e9:.2f} GB | "
            f"{rate:6.1f} MB/s | ETA {eta}{stall}")
        prev_new, prev_t = got, now
        if not pending:
            break
except KeyboardInterrupt:
    stop.set()
    pool.shutdown(wait=True, cancel_futures=True)
    log("stopped by you -> re-run this cell to resume")
    raise
pool.shutdown()

status = {}
for fut, key in futures.items():
    exc = fut.exception()
    status[key] = f"FAILED ({exc})" if exc else fut.result()
print(pd.Series([s.split(" (")[0] for s in status.values()]).value_counts().to_string())
failed = [k for k, s in status.items() if s.startswith("FAILED")]
wrong_size = [k for k, s in objects if not local_path(k).exists() or local_path(k).stat().st_size != s]
check(not failed, f"no download failed ({len(failed)} failed: {[Path(k).name for k in failed[:5]]} -> re-run this cell)")
check(not wrong_size, f"every file has the exact size listed on S3 ({len(wrong_size)} wrong: {[Path(k).name for k in wrong_size[:5]]})")
check((RAW_ROOT / "SHA256SUMS.txt").exists(), "SHA256SUMS.txt present")
for sub in ["EEG_Data", "EMG_Data", "FP_Data", "IMU_Data"]:
    check((RAW_ROOT / "Dataset" / sub).is_dir(), f"folder Dataset/{sub} exists")
for label, pattern, documented in [("EEG .csv", "S*_raw.csv", 177), ("EEG .edf", "S*_raw.edf", 177),
                                   ("EMG", "EMG_S*.csv", 169), ("force plate", "FP_S*.csv", 173),
                                   ("IMU archive", "*.tar.gz", 1)]:
    found = len(list((RAW_ROOT / "Dataset").rglob(pattern)))
    check(found == documented, f"{label}: {found} files (dataset page says {documented})", critical=False)
end_cell("CELL 2")

# ===== CELL 3/5 · Verify checksums (1-3 min) =====
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


entries = [m.groups() for line in (RAW_ROOT / "SHA256SUMS.txt").read_text(errors="ignore").splitlines()
           if (m := re.match(r"^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$", line))]
log(f"{len(entries)} files listed in SHA256SUMS.txt")


def verify(entry: tuple) -> tuple:
    digest, rel = entry
    path = RAW_ROOT / rel
    if not path.exists():
        return rel, "missing"
    return rel, "ok" if sha256(path) == digest.lower() else "MISMATCH"


results = []
with ThreadPoolExecutor(4) as pool:
    for res in pool.map(verify, entries):
        results.append(res)
        if len(results) % 100 == 0:
            log(f"hashed {len(results)}/{len(entries)}")
sums = pd.DataFrame(results, columns=["path", "status"])
print(sums.status.value_counts().to_string())
bad = sums[sums.status != "ok"]
if len(bad):
    print(bad.head(20).to_string())
for rel in bad[bad.status == "MISMATCH"].path:
    (RAW_ROOT / rel).unlink()       # right size but wrong content: CELL 2 would skip it, so delete it to force a fresh download
    log(f"deleted corrupted {rel}")
check(len(entries) > 0, "checksum list is not empty")
check(len(bad) == 0, f"all files match their checksums ({len(bad)} problems -> corrupted files deleted; re-run CELL 2, then this cell)")
end_cell("CELL 3")

# ===== CELL 4/5 · Extract IMU + subject manifest + gate G0 (1-3 min) =====
DATA = RAW_ROOT / "Dataset"
tars = sorted((DATA / "IMU_Data").glob("*.tar.gz"))
if not tars:
    raise FileNotFoundError("IMU archive missing -> re-run CELL 2")
done_marker = IMU_ROOT / "_EXTRACTED_OK"
if done_marker.exists():
    log(f"IMU already extracted in {IMU_ROOT} -> skipping")
else:
    IMU_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"extracting {tars[0].name} ({tars[0].stat().st_size / 1e9:.2f} GB) ...")
    with tarfile.open(tars[0], "r:gz") as archive:
        for i, member in enumerate(archive, 1):
            if member.name.startswith("/") or ".." in Path(member.name).parts:
                raise ValueError(f"unsafe path in archive: {member.name}")
            try:
                archive.extract(member, IMU_ROOT, filter="data")
            except TypeError:  # older Python without the filter argument
                archive.extract(member, IMU_ROOT)
            if i % 300 == 0:
                log(f"extracted {i} entries")
    done_marker.touch()
    log("extraction done")

PATTERNS = {"EEG": r"^S(\d+)_([0-9.]+)_raw\.edf$", "EMG": r"^EMG_S(\d+)_([0-9.]+)\.csv$",
            "FP": r"^FP_S(\d+)_([0-9.]+)\.csv$"}
rows = []
for path in DATA.rglob("*"):
    for kind, pattern in PATTERNS.items():
        m = re.match(pattern, path.name)
        if m:
            rows.append((kind, int(m[1]), float(m[2]), str(path)))
for folder in IMU_ROOT.rglob("IMU_S*"):
    m = re.match(r"^IMU_S(\d+)_([0-9.]+)$", folder.name)
    if m and folder.is_dir():
        rows.append(("IMU", int(m[1]), float(m[2]), str(folder)))
inv = pd.DataFrame(rows, columns=["kind", "subject", "speed", "path"])
print(inv.kind.value_counts().to_string())
check(len(inv) > 0, "found data files")

grid = pd.MultiIndex.from_product([range(1, N_SUBJECTS + 1), SPEEDS], names=["subject", "speed"])
manifest = pd.DataFrame(index=grid)
for kind in ["EEG", "EMG", "IMU", "FP"]:
    have = set(zip(inv[inv.kind == kind].subject, inv[inv.kind == kind].speed))
    manifest[kind] = [key in have for key in grid]
manifest["all4"] = manifest[["EEG", "EMG", "IMU", "FP"]].all(axis=1)
per_subject = manifest.groupby(level="subject").all4.all()
complete_subjects = per_subject[per_subject].index.tolist()
missing = {f"S{s}_{sp:g}": [k for k in ["EEG", "EMG", "IMU", "FP"] if not manifest.loc[(s, sp), k]]
           for s, sp in manifest.index[~manifest.all4.to_numpy()]}
print(f"missing trials ({len(missing)}):", dict(list(missing.items())[:15]), "..." if len(missing) > 15 else "")

DOCUMENTED_MISSING = {"EEG": set(),
                      "EMG": {"S3_0.75", "S3_1", "S8_0.5", "S11_1", "S43_1", "S50_0.5", "S50_0.75", "S50_1"},
                      "IMU": {"S2_0.75", "S2_1", "S3_0.75", "S3_1", "S24_0.5", "S24_0.75", "S24_1"},
                      "FP": {"S3_0.75", "S3_1", "S8_0.75", "S45_1"}}
for kind, documented in DOCUMENTED_MISSING.items():
    observed = {key for key, kinds in missing.items() if kind in kinds}
    extra, back = sorted(observed - documented), sorted(documented - observed)
    check(not extra and not back, f"{kind} missing trials match the dataset page "
          f"(extra {len(extra)}: {extra[:8]} | not missing after all {len(back)}: {back[:8]})", critical=False)
n_csv = [len(list(Path(p).rglob("*.csv"))) for p in inv[inv.kind == "IMU"].path]
check(len(n_csv) > 0 and set(n_csv) == {18}, f"every IMU folder has 18 CSV files (counts seen: {sorted(set(n_csv))})", critical=False)
check(len(complete_subjects) >= 45,
      f"GATE G0: {len(complete_subjects)} subjects complete in all 4 modalities x 3 speeds (need >= 45)", critical=False)
manifest.reset_index().to_csv(OUT_ROOT / "manifest.csv", index=False)
check((OUT_ROOT / "manifest.csv").exists(), f"manifest saved: {OUT_ROOT / 'manifest.csv'}")
end_cell("CELL 4")

# ===== CELL 5/5 · Peek at one file per modality (seconds) -> paste this output back =====
def first_path(kind: str) -> Path:
    return Path(inv[inv.kind == kind].sort_values(["subject", "speed"]).iloc[0].path)


emg_path = first_path("EMG")
emg = pd.read_csv(emg_path)
t = emg.iloc[:, 0].to_numpy(float)
print(f"--- EMG {emg_path.name}: {emg.shape[0]} rows x {emg.shape[1]} cols | "
      f"fs = {1 / np.median(np.diff(t)):.2f} Hz | last t = {t[-1]:.1f} s")
print("columns:", list(emg.columns))
check(emg.shape[1] == 13 and all("MAV" in str(c) for c in emg.columns[1:]), "EMG = time + 12 MAV channels", critical=False)

fp_path = first_path("FP")
fp_head = pd.read_csv(fp_path, nrows=3)
with open(fp_path, "rb") as fh:
    fp_rows = sum(block.count(b"\n") for block in iter(lambda: fh.read(1 << 22), b"")) - 1
print(f"\n--- FP {fp_path.name}: {fp_rows} rows x {fp_head.shape[1]} cols (= {fp_rows / 1000:.1f} s if 1000 Hz)")
print("columns:", list(fp_head.columns))
print(fp_head.to_string())
check(fp_head.shape[1] == 22, "force plate has 22 columns (11 per leg)", critical=False)

eeg_csv = sorted(DATA.rglob("S*_raw.csv"))[0]
print(f"\n--- EEG CSV header {eeg_csv.name} (first 18 lines):")
with open(eeg_csv, errors="ignore") as fh:
    for _ in range(18):
        print("   ", fh.readline().rstrip()[:180])

imu_dir = first_path("IMU")
print(f"\n--- IMU folder {imu_dir.name}:")
for f in sorted(imu_dir.rglob("*.csv")):
    head = pd.read_csv(f, nrows=2)
    print(f"   {f.name}: {head.shape[1]} cols | first columns: {list(head.columns[:4])}")
end_cell("CELL 5")
