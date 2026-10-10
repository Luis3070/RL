"""Single-thread, batch-1 ONNX Runtime latency of the deployed networks (desktop proxy for onboard timing).
Usage: taskset -c 31 python tools/time_onnx_inference.py model1.onnx [model2.onnx ...]"""
import json, sys, time
import numpy as np, onnxruntime as ort

out = {}
for path in sys.argv[1:]:
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    inp = s.get_inputs()[0]; x = np.random.default_rng(0).normal(size=(1, inp.shape[1])).astype(np.float32)
    for _ in range(2000):
        s.run(None, {inp.name: x})
    t = []
    for _ in range(20000):
        a = time.perf_counter_ns(); s.run(None, {inp.name: x}); t.append(time.perf_counter_ns() - a)
    t = np.array(t) / 1e3
    out[path] = {"median_us": round(float(np.median(t)), 1), "p95_us": round(float(np.percentile(t, 95)), 1),
                 "p99_us": round(float(np.percentile(t, 99)), 1), "max_us": round(float(t.max()), 1), "n": len(t)}
    print(path, out[path], flush=True)
print(json.dumps(out))
