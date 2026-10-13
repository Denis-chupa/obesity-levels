"""Run notebook cells by id in a clean process (cwd holds research.ipynb).
Usage: run_cells.py <ids,comma> [--sub pairs.json]   pairs: [[old, new], ...] applied to cell sources."""
import json, sys, time, resource
import matplotlib
matplotlib.use("Agg")
args = sys.argv[1:]
subs = json.load(open(args[args.index("--sub") + 1], encoding="utf-8")) if "--sub" in args else []
nb = json.load(open("research.ipynb", encoding="utf-8"))
by_id = {c["id"]: "".join(c["source"]) for c in nb["cells"]}
hits = [0] * len(subs)
ns = {"display": lambda o: print(o.to_string() if hasattr(o, "to_string") else o, flush=True), "__name__": "__main__"}
for cid in args[0].split(","):
    code = by_id[cid] if cid in by_id else open(cid, encoding="utf-8").read()
    for i, (old, new) in enumerate(subs):
        hits[i] += code.count(old); code = code.replace(old, new)
    t = time.perf_counter()
    print(f"[start] {cid} {time.strftime('%H:%M:%S')}", flush=True)
    exec(compile(code, cid, "exec"), ns)
    print(f"[ok] {cid} {(time.perf_counter() - t) / 60:.1f} min", flush=True)
assert all(hits), ("substitution never matched", hits)
print("peak RSS MB:", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20), flush=True)
print("DONE", flush=True)
