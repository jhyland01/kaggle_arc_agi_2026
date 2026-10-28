# --- system census, one line every 10 minutes -------------------------------
# Everything reads as used/total. RAM is total - MemAvailable, which runs higher
# than the "used" column in `free -h`: that one excludes reclaimable page cache,
# and MemAvailable is the figure that actually predicts an OOM. Disk likewise
# counts the root reserve as unavailable, matching df.
#
# A daemon thread rather than a hook in the run loop: the point of watching a
# long run is to see what happens when it stops behaving, and a print driven by
# the harness stops with it. This keeps reporting through a deadlock.
import os, subprocess, threading, sys

def _sysline(paths=("/", "/kaggle/working")):
    GB = 1024 ** 3
    out = []
    mem = {}
    try:
        for line in open("/proc/meminfo"):
            k, _, rest = line.partition(":")
            p = rest.split()
            if p:
                mem[k] = int(p[0]) * 1024
    except OSError:
        pass
    if mem.get("MemTotal") and mem.get("MemAvailable") is not None:
        out.append(f"RAM {(mem['MemTotal']-mem['MemAvailable'])/GB:.1f}/{mem['MemTotal']/GB:.1f}")
    if mem.get("SwapTotal"):
        out.append(f"swap {(mem['SwapTotal']-mem['SwapFree'])/GB:.1f}/{mem['SwapTotal']/GB:.1f}")
    else:
        out.append("swap none")
    st = {}
    try:
        for line in open("/proc/self/status"):
            k, _, rest = line.partition(":")
            p = rest.split()
            if k in ("VmRSS", "Threads") and p:
                st[k] = int(p[0])
    except OSError:
        pass
    proc = []
    if "VmRSS" in st:
        proc.append(f"{st['VmRSS']*1024/GB:.1f} GB")
    proc.append(f"{st.get('Threads', threading.active_count())} threads")
    try:
        proc.append(f"{len(os.listdir('/proc/self/fd'))} fds")
    except OSError:
        pass
    out.append("proc " + ", ".join(proc))
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5, check=False)
        gpus = []
        for i, line in enumerate(r.stdout.splitlines()):
            p = [x.strip() for x in line.split(",")]
            if len(p) == 2:
                gpus.append(f"GPU{i} {float(p[0])/1024:.1f}/{float(p[1])/1024:.1f}")
        out.append(" ".join(gpus) if gpus else "GPU unavailable")
    except Exception:
        out.append("GPU unavailable")          # nvidia-smi hangs on a wedged GPU
    disks = []
    for path in paths:
        try:
            s = os.statvfs(path)
        except OSError:
            continue
        total = s.f_blocks * s.f_frsize
        disks.append(f"{path} {(total - s.f_bavail*s.f_frsize)/GB:.1f}/{total/GB:.1f}")
    if disks:
        out.append("disk " + ", ".join(disks))
    return "[sys] " + " | ".join(out)


def start_monitor(interval_minutes=10.0, paths=("/", "/kaggle/working")):
    stop = threading.Event()
    def loop():
        while not stop.is_set():
            try:
                print(_sysline(paths), flush=True)
            except Exception as exc:
                print(f"[sys] census failed: {exc!r}", flush=True)
            stop.wait(interval_minutes * 60.0)
    threading.Thread(target=loop, name="system-stats", daemon=True).start()
    return stop

if not TRUE_SUBMISSION:
    _stop_monitor = start_monitor(2)
    print(_sysline())