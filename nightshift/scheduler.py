"""Scheduler: run the batch once a day. Deliberately boring.

    python3 -m nightshift.scheduler --at 02:00
    python3 -m nightshift.scheduler --every 3600
"""
import argparse
import time
import traceback
from datetime import datetime, timedelta

from .batch import run_batch


def _seconds_until(hhmm):
    now = datetime.now()
    h, m = (int(x) for x in hhmm.split(":"))
    target = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", help="daily wall-clock time, e.g. 02:00")
    ap.add_argument("--every", type=int, help="fixed interval in seconds")
    ap.add_argument("--fleet", type=int, default=None)
    ap.add_argument("--once", action="store_true", help="run immediately then exit")
    a = ap.parse_args()

    if a.once:
        run_batch(fleet_size=a.fleet)
        return

    while True:
        wait = a.every if a.every else _seconds_until(a.at or "02:00")
        nxt = datetime.now() + timedelta(seconds=wait)
        print(f"[sched] next batch at {nxt:%Y-%m-%d %H:%M:%S} ({wait/60:.0f} min)", flush=True)
        time.sleep(wait)
        try:
            print(f"[sched] starting batch {datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)
            run_batch(fleet_size=a.fleet)
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    main()
