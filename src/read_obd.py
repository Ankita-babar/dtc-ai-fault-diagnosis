"""
Read trouble codes from a REAL vehicle and diagnose them.

Hardware: ELM327 adapter (USB, Bluetooth or Wi-Fi) plugged into the car's OBD-II port, car ignition ON.
Library : python-OBD  (pip install obd)

  python src/read_obd.py                                  # auto-detect adapter
  python src/read_obd.py --port /dev/tty.usbserial-1410   # Mac USB      (Windows: COM4, Linux: /dev/ttyUSB0)
  python src/read_obd.py --port /dev/rfcomm0              # Bluetooth after pairing (Linux)
  python src/read_obd.py --send                           # also push codes to the dashboard (button: Load codes from car)
  python src/read_obd.py --watch 5                        # re-read every 5 s
  python src/read_obd.py --simulate P0301 P1A2B           # NO car: pretend these were read (for demo / testing)

Reads Mode 03 (stored) and Mode 07 (pending) codes. It NEVER clears codes (clearing erases freeze-frame data).
"""
import argparse, json, time, urllib.request

from diagnose import DiagnosisEngine, print_report


def read_codes(conn, commands):
    """commands: [(label, python-OBD command)] -> sorted unique list of codes, plus where each came from."""
    found = {}
    for label, cmd in commands:
        resp = conn.query(cmd)
        if resp is None or resp.is_null() if hasattr(resp, "is_null") else resp is None:
            continue
        for code, _desc in (resp.value or []):
            found.setdefault(str(code).upper(), label)
    return found


def send(codes, url):
    body = json.dumps({"codes": codes}).encode()
    req = urllib.request.Request(url + "/api/ingest", data=body, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=5); print(f"-> sent {len(codes)} code(s) to dashboard at {url}")
    except Exception as e:
        print(f"(dashboard not reachable at {url}: {e})")


def run_once(engine, codes, where, args):
    print(f"\nCodes read ({len(codes)}): " + (", ".join(f"{c} [{where.get(c, 'sim')}]" for c in codes) or "none - no faults stored"))
    print_report(engine.diagnose(codes))
    if args.send: send(codes, args.url)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port"); ap.add_argument("--baud", type=int); ap.add_argument("--send", action="store_true")
    ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--watch", type=float, default=0)
    ap.add_argument("--simulate", nargs="*")
    args = ap.parse_args()
    engine = DiagnosisEngine()
    if args.simulate is not None:
        return run_once(engine, [c.upper() for c in args.simulate], {}, args)
    import obd                                           # only needed with real hardware
    conn = obd.OBD(portstr=args.port, baudrate=args.baud, fast=False, timeout=30)
    if not conn.is_connected():
        raise SystemExit("Could not connect. Check: adapter plugged in/paired, ignition ON, correct --port "
                         "(list ports with: python -m serial.tools.list_ports)")
    print("Connected:", conn.port_name(), "| protocol:", conn.protocol_name())
    cmds = [("stored", obd.commands.GET_DTC), ("pending", obd.commands.GET_CURRENT_DTC)]
    try:
        while True:
            found = read_codes(conn, cmds)
            run_once(engine, sorted(found), found, args)
            if not args.watch: break
            time.sleep(args.watch)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
