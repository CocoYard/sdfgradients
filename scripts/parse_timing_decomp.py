"""
Parse the timing-decomposition SLURM logs into a per-step runtime breakdown for
OURS, and compare total wall-time against the earlier timing_4algo run.

Buckets (sum ~= ours wall):
  bvh_build, rt (radical tessellation / find_intersections),
  visible_arc (compute_exposed_batch), gradient_opt, rbf_fit (init+loop),
  visibility (loop+final are_points_visible), surface_extract.

Usage:
  python scripts/parse_timing_decomp.py \
    --logs '/scratch/ycheng27/timing_decomp/logs/slurm_*_*.out' \
    --old-timing-dir /scratch/ycheng27/timing_4algo/logs
"""
import argparse, glob, re, csv
from collections import defaultdict
import numpy as np

def f(pat, s):
    m = re.search(pat, s)
    return float(m.group(1)) if m else None

TAGS = [
    ('rt',     r'\[get_visible_arcs\] find_intersections:\s+([\d.eE+-]+) s'),
    ('arc',    r'compute_exposed_batch:\s+([\d.eE+-]+) s'),
    ('init_fit', r'\[DECOMP-init\] rbf_fit:\s+([\d.eE+-]+) s'),
    ('iploop', r'iterative_projection_3d:\s+([\d.eE+-]+) s'),
    ('final_vis', r'final projection \+ visibility:\s+([\d.eE+-]+) s'),
    ('main_total', r'\[main_algorithm\] total:\s+([\d.eE+-]+) s'),
    ('coarse_pred', r'coarse predict \(\d+ pts\):\s+([\d.eE+-]+) s'),
    ('fine_pred',   r'fine predict \(\d+ pts\):\s+([\d.eE+-]+) s'),
    ('dc',          r'dual_contouring:\s+([\d.eE+-]+) s'),
]

def parse_log(path):
    """C++ stdout (block-buffered) and Python stdout interleave, so we CANNOT
    segment by the Python 'ours gl=' line. Instead: (1) collect ordered C++
    blocks bounded by 'build SphereBVH' .. '[extract_surface] total' (the C++
    stream is internally ordered), (2) collect ordered (gl, wall) from the
    Python 'ours gl=' lines, (3) zip positionally — both are in gl-ascending
    order with the same count."""
    lines = open(path, errors='ignore').read().splitlines()
    gls = []; fid = None
    for ln in lines:
        m = re.search(r'\[(\d+)\]\s+ours gl=(\d+)\s+(\w+)\s+([\d.]+)s', ln)
        if m:
            fid = m.group(1)
            if m.group(3) == 'ok':
                gls.append((int(m.group(2)), float(m.group(4))))
    blocks, cur = [], None
    for ln in lines:
        if 'build SphereBVH' in ln:
            if cur is not None: blocks.append(cur)
            cur = {'bvh': f(r'build SphereBVH:\s+([\d.eE+-]+) s', ln)}
        if cur is None:
            continue
        for key, pat in TAGS:
            v = f(pat, ln)
            if v is not None: cur[key] = v
        m = re.search(r'\[DECOMP-loop\] rbf_fit:\s+([\d.eE+-]+) s\s+visibility:\s+([\d.eE+-]+) s\s+rbf_eval:\s+([\d.eE+-]+) s', ln)
        if m:
            cur['loop_fit'] = float(m.group(1)); cur['loop_vis'] = float(m.group(2)); cur['loop_eval'] = float(m.group(3))
        if '[extract_surface] total' in ln:
            cur['extract'] = f(r'\[extract_surface\] total:\s+([\d.eE+-]+) s', ln)
            blocks.append(cur); cur = None
    if cur is not None: blocks.append(cur)
    if len(blocks) != len(gls):
        print(f"  WARN {path}: {len(blocks)} cpp blocks vs {len(gls)} ok gl-lines")
    for (gl, wall), blk in zip(gls, blocks):
        blk['gl'] = gl; blk['wall'] = wall; blk['status'] = 'ok'; blk['fid'] = fid
        yield blk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', default='/scratch/ycheng27/timing_decomp/logs/slurm_*_*.out')
    ap.add_argument('--old-timing-dir', default='/scratch/ycheng27/timing_4algo/logs')
    args = ap.parse_args()

    runs = []
    for p in glob.glob(args.logs):
        runs.extend(list(parse_log(p)))
    runs = [r for r in runs if r.get('status') == 'ok' and 'wall' in r]

    # derive buckets
    for r in runs:
        r['rbf_fit'] = (r.get('init_fit',0) or 0) + (r.get('loop_fit',0) or 0)
        r['visibility'] = (r.get('loop_vis',0) or 0) + (r.get('final_vis',0) or 0)
        r['grad_opt'] = (r.get('iploop',0) or 0) - (r.get('loop_fit',0) or 0) - (r.get('loop_vis',0) or 0)
        named = ['bvh','rt','arc','grad_opt','rbf_fit','visibility','extract']
        r['other'] = r['wall'] - sum(r.get(k,0) or 0 for k in named)

    BUCKETS = ['bvh','rt','arc','grad_opt','rbf_fit','visibility','extract','other']
    LABEL = {'bvh':'BVH build','rt':'RT','arc':'visible arc','grad_opt':'gradient opt',
             'rbf_fit':'RBF fit','visibility':'visibility chk','extract':'surface extract','other':'other'}

    gls = sorted(set(r['gl'] for r in runs))
    print(f"Parsed {len(runs)} ours runs over {len(set(r['fid'] for r in runs))} meshes, gls={gls}\n")

    # ---- decomposition table: mean seconds per gl ----
    print("=== Mean per-step wall-time (s) by grid_len ===")
    hdr = "step".ljust(16) + "".join(f"{gl:>9}" for gl in gls)
    print(hdr); print("-"*len(hdr))
    by = {b: {} for b in BUCKETS + ['wall']}
    for gl in gls:
        rr = [r for r in runs if r['gl']==gl]
        for b in BUCKETS + ['wall']:
            by[b][gl] = np.mean([r.get(b,0) or 0 for r in rr])
    for b in BUCKETS:
        print(LABEL[b].ljust(16) + "".join(f"{by[b][gl]:>9.3f}" for gl in gls))
    print("total".ljust(16) + "".join(f"{by['wall'][gl]:>9.3f}" for gl in gls))

    # ---- percentage table ----
    print("\n=== Mean per-step share (% of ours wall) by grid_len ===")
    print(hdr); print("-"*len(hdr))
    for b in BUCKETS:
        print(LABEL[b].ljust(16) + "".join(f"{100*by[b][gl]/by['wall'][gl]:>8.1f}%" for gl in gls))

    # ---- RBF interpolation total (fit + all evaluations) — the reviewer's question ----
    # fit           = init + in-loop refits
    # eval(opt)     = predict_gradients inside optimize_best_gradients (g_rbf_eval_s)
    # eval(extract) = coarse + fine predict on the extraction grid
    for r in runs:
        r['rbf_eval_opt'] = r.get('loop_eval',0) or 0
        r['rbf_eval_ext'] = (r.get('coarse_pred',0) or 0) + (r.get('fine_pred',0) or 0)
        r['rbf_total'] = r['rbf_fit'] + r['rbf_eval_opt'] + r['rbf_eval_ext']
    print("\n=== RBF interpolation share of ours wall (%) — fit vs evaluation ===")
    print(hdr); print("-"*len(hdr))
    for key, lab in [('rbf_fit','RBF fit'), ('rbf_eval_opt','RBF eval (opt)'),
                     ('rbf_eval_ext','RBF eval (extract)'), ('rbf_total','RBF TOTAL')]:
        row = lab.ljust(16)
        for gl in gls:
            rr = [r for r in runs if r['gl']==gl]
            m = np.mean([r.get(key,0) or 0 for r in rr])
            row += f"{100*m/by['wall'][gl]:>8.1f}%"
        print(row)

    # ---- total-time comparison vs old timing_4algo ----
    old = {}
    for r in runs:
        fid = r['fid']
        tf = f"{args.old_timing_dir}/{fid}.timing.csv"
        try:
            for row in csv.DictReader(open(tf)):
                if row['algo']=='ours' and row['status'] in ('ok','skipped'):
                    old[(fid,int(row['grid_len']))] = float(row['wall_s'])
        except FileNotFoundError:
            pass
    print("\n=== Total ours wall: NEW vs OLD (mean over meshes, s) ===")
    print("gl".ljust(6)+"new".rjust(10)+"old".rjust(10)+"speedup".rjust(10)+"n".rjust(5))
    for gl in gls:
        rr=[r for r in runs if r['gl']==gl]
        pairs=[(r['wall'], old.get((r['fid'],gl))) for r in rr]
        pairs=[(n,o) for n,o in pairs if o is not None]
        if not pairs:
            print(f"{gl:<6}{by['wall'][gl]:>10.2f}{'—':>10}{'—':>10}{0:>5}"); continue
        nm=np.mean([n for n,o in pairs]); om=np.mean([o for n,o in pairs])
        print(f"{gl:<6}{nm:>10.2f}{om:>10.2f}{om/nm:>9.2f}x{len(pairs):>5}")

if __name__ == '__main__':
    main()
