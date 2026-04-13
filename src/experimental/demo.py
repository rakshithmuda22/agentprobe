"""
Demo — QJL Resonance Experimental Module
=========================================

Nine demonstrations:
1. JL distance preservation: validates QJL projections preserve angular relationships
2. XOR speed benchmark: uint8 vs packed uint64 vs cosine similarity
3. Gossip convergence: decentralized gossip with dissonant agent injection
4. Stress test: PID governor, outlier detection in high-drift scenario
5. Code analysis: structural analysis of resonance_core.py
6. PyTorch backend benchmark: NumPy vs PyTorch CPU/MPS across agent counts
7. Refactor proposals: SPECULATIVE manifold-based refactoring suggestions
8. Multi-file manifold: scan src/experimental/ directory
9. Telemetry suite: ASCII convergence/heatmap/distribution visualization

Run: python -m src.experimental.demo
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

from src.experimental.resonance_core import (
    QJLProjector,
    batch_resonance_scores,
    batch_resonance_scores_packed,
    pack_quantized,
    pack_quantized_batch,
)
from src.experimental.gossip_governance import (
    GossipConfig,
    GossipController,
    ResonancePIDGovernor,
)
from src.experimental.code_transformer import CodeTransformer, CodeSwarm, ManifoldScanner
from src.experimental.refactor_proposals import RefactorAnalyzer
from src.experimental.telemetry import (
    ResonanceTelemetry,
    ascii_convergence,
    ascii_heatmap,
    ascii_bar,
)


def demo_jl_preservation() -> None:
    """Validate that QJL projections preserve angular relationships."""
    print("=" * 60)
    print("DEMO 1: JL Distance Preservation")
    print("=" * 60)

    n_vectors = 50
    input_dim = 64
    projection_dim = 128
    seed = 42

    rng = np.random.default_rng(seed)
    projector = QJLProjector(input_dim, projection_dim, n_bits=3, seed=seed)

    # Generate random unit vectors
    vectors = rng.standard_normal((n_vectors, input_dim))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= norms

    # Ground truth: pairwise angular distances
    cosine_sims = vectors @ vectors.T
    cosine_sims = np.clip(cosine_sims, -1.0, 1.0)
    angular_distances = np.arccos(cosine_sims) / np.pi  # normalized to [0, 1]

    # QJL projections: pairwise resonance scores
    quantized = projector.batch_project_and_quantize(vectors)
    n = len(quantized)
    resonance_scores = np.zeros((n, n))
    for i in range(n):
        resonance_scores[i] = batch_resonance_scores(quantized[i], quantized)

    # Extract upper triangle (unique pairs)
    mask = np.triu_indices(n, k=1)
    true_angular = angular_distances[mask]
    qjl_resonance = resonance_scores[mask]

    # Pearson correlation
    r, p_value = scipy_stats.pearsonr(true_angular, qjl_resonance)

    print(f"  Vectors: {n_vectors} random unit vectors in R^{input_dim}")
    print(f"  Projection dim: {projection_dim}, Quantization: 3-bit Gray")
    print(f"  Unique pairs: {len(true_angular)}")
    print(f"  Pearson r (angular dist vs resonance score): {r:.4f}")
    print(f"  p-value: {p_value:.2e}")
    print(f"  Result: {'PASS' if r > 0.80 else 'WEAK'} (threshold: r > 0.80)")
    print()


def demo_xor_speed() -> None:
    """Benchmark XOR+popcount resonance vs. cosine similarity."""
    print("=" * 60)
    print("DEMO 2: XOR Speed Benchmark")
    print("=" * 60)

    projection_dim = 512
    n_trials = 10_000
    seed = 42

    rng = np.random.default_rng(seed)

    qa = rng.integers(0, 8, size=projection_dim, dtype=np.uint8)
    qb_batch = rng.integers(0, 8, size=(n_trials, projection_dim), dtype=np.uint8)
    fa = rng.standard_normal(projection_dim)
    fb_batch = rng.standard_normal((n_trials, projection_dim))

    # Time uint8 XOR+popcount
    t0 = time.perf_counter()
    _ = batch_resonance_scores(qa, qb_batch, n_bits=3)
    t_uint8 = time.perf_counter() - t0

    # Time packed uint64 (pre-pack then score)
    qa_packed = pack_quantized(qa)
    qb_packed = pack_quantized_batch(qb_batch)
    t0 = time.perf_counter()
    _ = batch_resonance_scores_packed(qa_packed, qb_packed, n_dims=projection_dim)
    t_packed = time.perf_counter() - t0

    # Time cosine similarity
    t0 = time.perf_counter()
    fa_norm = fa / np.linalg.norm(fa)
    fb_norms = fb_batch / np.linalg.norm(fb_batch, axis=1, keepdims=True)
    _ = fb_norms @ fa_norm
    t_cos = time.perf_counter() - t0

    print(f"  Dimension: {projection_dim}, Trials: {n_trials}")
    print(f"  uint8 XOR+popcount:  {t_uint8 * 1000:7.2f} ms")
    print(f"  packed uint64:       {t_packed * 1000:7.2f} ms  ({t_uint8 / t_packed:.1f}x vs uint8)")
    print(f"  Cosine similarity:   {t_cos * 1000:7.2f} ms  ({t_cos / t_uint8:.1f}x vs uint8)")
    print(f"  Packed vs cosine:    {t_cos / t_packed:.1f}x speedup")
    print(f"  Note: pre-packing cost not included (amortized over many comparisons)")
    print()


def demo_gossip_convergence() -> None:
    """Run gossip controller and show convergence curve."""
    print("=" * 60)
    print("DEMO 3: Gossip Convergence")
    print("=" * 60)

    config = GossipConfig(
        n_agents=16,
        state_dim=64,
        projection_dim=128,
        n_bits=3,
        fanout=3,
        dissonance_threshold=0.35,
        resync_strength=0.15,
        drift_magnitude=0.03,
        max_rounds=80,
        seed=42,
    )

    controller = GossipController(config)
    results = controller.run()

    # Print text-art convergence curve
    curve = controller.get_convergence_curve()
    resync_curve = controller.get_resync_curve()

    # Summary stats
    first_10 = np.mean(curve[:10])
    last_10 = np.mean(curve[-10:])
    converged = last_10 < first_10

    print(f"  Agents: {config.n_agents}, Fanout: {config.fanout}")
    print(f"  Rounds: {config.max_rounds}")
    print(f"  Dissonance threshold: {config.dissonance_threshold}")
    print()
    print("  Round | Mean Surprise | Resyncs | Curve")
    print("  ------+---------------+---------+------")

    # Sample every 5 rounds for display
    bar_width = 30
    max_surprise = max(curve) if curve else 1.0
    for i in range(0, len(curve), 5):
        bar_len = int((curve[i] / max(max_surprise, 0.01)) * bar_width)
        bar = "#" * bar_len + "." * (bar_width - bar_len)
        print(f"  {i + 1:5d} | {curve[i]:13.4f} | {resync_curve[i]:7d} | {bar}")

    print()
    print(f"  First 10 rounds avg surprise: {first_10:.4f}")
    print(f"  Last 10 rounds avg surprise:  {last_10:.4f}")
    print(f"  Convergence: {'YES' if converged else 'NO'} (last < first)")

    # Inject a dissonant agent and run 20 more rounds
    print()
    print("  --- Injecting dissonant agent ---")
    did = controller.inject_dissonant_agent(magnitude=5.0)
    extra = controller.run(n_rounds=20)
    extra_curve = [s.mean_surprise for s in extra]

    print(f"  Dissonant agent: {did}")
    print(f"  Surprise spike after injection: {extra_curve[0]:.4f}")
    print(f"  Surprise after 20 rounds recovery: {extra_curve[-1]:.4f}")
    recovery = extra_curve[-1] < extra_curve[0]
    print(f"  Recovery: {'YES' if recovery else 'NO'}")
    print()


def demo_stress_test() -> None:
    """200-agent swarm with PID governor, structural outlier detection."""
    print("=" * 60)
    print("DEMO 4: Stress Test — 200 Agents + PID Governor")
    print("=" * 60)

    seed = 42
    state_dim = 64
    n_normal = 150
    n_outliers = 50
    n_agents = n_normal + n_outliers
    n_rounds = 100

    rng = np.random.default_rng(seed)

    # --- Build belief vectors ---
    # 150 normal agents from 3 Gaussian clusters
    cluster_centers = rng.standard_normal((3, state_dim))
    cluster_centers /= np.linalg.norm(cluster_centers, axis=1, keepdims=True)

    normal_beliefs = []
    for i in range(n_normal):
        center = cluster_centers[i % 3]
        noise = rng.standard_normal(state_dim) * 0.25
        v = center + noise
        v /= np.linalg.norm(v)
        normal_beliefs.append(v)

    # 50 outlier agents from a 4th cluster (far from normal clusters)
    outlier_center = rng.standard_normal(state_dim) * 4.0
    outlier_center /= np.linalg.norm(outlier_center)
    outlier_ids_set = {f"agent_{i:04d}" for i in range(n_normal, n_agents)}

    outlier_beliefs = []
    for _ in range(n_outliers):
        noise = rng.standard_normal(state_dim) * 0.25
        v = outlier_center + noise
        v /= np.linalg.norm(v)
        outlier_beliefs.append(v)

    all_beliefs = normal_beliefs + outlier_beliefs

    # --- PID run with drift ---
    pid = ResonancePIDGovernor(
        target_surprise=0.15, Kp=0.6, Ki=0.015, Kd=0.12,
        base_strength=0.1, integral_window=20,
    )
    config_pid = GossipConfig(
        n_agents=n_agents,
        state_dim=state_dim,
        projection_dim=128,
        fanout=6,                 # higher fanout → more neighbor observations
        dissonance_threshold=0.30,
        resync_strength=0.1,
        drift_magnitude=0.02,     # small drift gives PID something to govern
        max_rounds=n_rounds,
        seed=seed,
        pid_governor=pid,
    )
    ctrl = GossipController(config_pid)
    ctrl.initialize()
    for belief, agent in zip(all_beliefs, ctrl.agents.values()):
        agent.update_belief(belief)

    print(f"  Agents: {n_normal} normal (3 clusters) + {n_outliers} outliers")
    print(f"  Fanout: {config_pid.fanout}, Drift: {config_pid.drift_magnitude}, "
          f"Rounds: {n_rounds}")
    print(f"  PID target surprise: 0.15")
    print()
    ctrl.run(n_rounds=n_rounds)

    # --- Compute per-agent final surprise over last 20 rounds ---
    last_20 = ctrl.round_history[-20:]
    final_surprises: dict[str, float] = {}
    for aid in ctrl.agents:
        vals = [s.agent_surprises.get(aid, float("nan")) for s in last_20]
        finite = [v for v in vals if not np.isnan(v)]
        final_surprises[aid] = float(np.mean(finite)) if finite else 0.0

    ranked = sorted(final_surprises.items(), key=lambda x: x[1], reverse=True)
    top_50_ids = {aid for aid, _ in ranked[:50]}
    detected = top_50_ids & outlier_ids_set
    detection_rate = len(detected) / n_outliers

    # --- PID convergence curve (sampled) ---
    curve = ctrl.get_convergence_curve()
    pid_history = [s for s in ctrl.round_history]
    bar_width = 25
    max_s = max(curve) if curve else 1.0
    print("  Round | Surprise | Resync% | PID curve")
    print("  ------+----------+---------+----------")
    for i in range(0, n_rounds, 10):
        bar_len = int((curve[i] / max(max_s, 0.01)) * bar_width)
        bar = "#" * bar_len + "." * (bar_width - bar_len)
        resync_pct = pid_history[i].n_resyncs / max(pid_history[i].n_gossip_exchanges, 1) * 100
        print(f"  {i + 1:5d} | {curve[i]:8.4f} | {resync_pct:6.1f}% | {bar}")

    print()
    print(f"  Gossip detection: {len(detected)}/{n_outliers} outliers in top-50 "
          f"({detection_rate * 100:.0f}%)")

    # --- Direct pairwise detection (always works regardless of gossip coverage) ---
    from src.experimental.resonance_core import (
        QJLProjector as _QJL,
        batch_resonance_scores as _brs,
    )
    projector = _QJL(state_dim, 128, seed=seed)
    all_beliefs_arr = np.array(all_beliefs, dtype=np.float64)
    quantized = projector.batch_project_and_quantize(all_beliefs_arr)
    # Swarm centroid: most common quantized level per dimension
    centroid = np.round(np.mean(quantized[:n_normal], axis=0)).astype(np.uint8)
    centroid = np.clip(centroid, 0, 7)
    scores = _brs(centroid, quantized)
    ranked_direct = np.argsort(scores)[::-1]
    top50_direct = set(ranked_direct[:50].tolist())
    outlier_indices = set(range(n_normal, n_agents))
    detected_direct = top50_direct & outlier_indices
    direct_rate = len(detected_direct) / n_outliers

    print(f"  Direct pairwise:  {len(detected_direct)}/{n_outliers} outliers in top-50 "
          f"({direct_rate * 100:.0f}%)")

    # --- Comparison: no-PID static run ---
    config_static = GossipConfig(
        n_agents=n_agents, state_dim=state_dim, projection_dim=128,
        fanout=6, dissonance_threshold=0.30, resync_strength=0.15,
        drift_magnitude=0.02, max_rounds=n_rounds, seed=seed,
    )
    ctrl_s = GossipController(config_static)
    ctrl_s.initialize()
    for belief, agent in zip(all_beliefs, ctrl_s.agents.values()):
        agent.update_belief(belief)
    ctrl_s.run(n_rounds=n_rounds)
    static_curve = ctrl_s.get_convergence_curve()
    pid_final = float(np.mean(curve[-20:]))
    static_final = float(np.mean(static_curve[-20:]))
    print(f"  PID final surprise:    {pid_final:.4f}")
    print(f"  Static final surprise: {static_final:.4f}")
    better = "PID wins" if pid_final < static_final else "static wins"
    print(f"  Verdict: {better}")
    print(f"  Direct detection result: {'PASS' if direct_rate >= 0.60 else 'PARTIAL'} "
          f"(threshold ≥60%)")
    print()


def demo_code_analysis() -> None:
    """Structural analysis of resonance_core.py using CodeTransformer."""
    print("=" * 60)
    print("DEMO 5: Code Structural Analysis")
    print("=" * 60)

    # Use resonance_core.py — it has QJLProjector class + many functions,
    # giving a rich set of chunks for structural comparison.
    target = Path(__file__).parent / "resonance_core.py"
    target_path = str(target)

    projector = QJLProjector(input_dim=64, projection_dim=128, seed=42)
    transformer = CodeTransformer(projector, state_dim=64)

    print(f"  Target: {target.name}")

    chunks = transformer.extract_chunks(target_path)
    transformer.project_chunks(chunks)

    print(f"  Chunks extracted: {len(chunks)}")
    print()
    print("  Chunk               | Type     | Lines  | Cyclomatic")
    print("  --------------------+----------+--------+-----------")
    for c in chunks[:12]:  # show first 12
        ctype = c.chunk_type.ljust(8)
        lines = f"{c.line_start}-{c.line_end}".rjust(6)
        cyclomatic = f"{c.feature_vector[3]:.2f}" if c.feature_vector is not None else "?"
        print(f"  {c.name[:20].ljust(20)}| {ctype} | {lines} | {cyclomatic}")

    if len(chunks) > 12:
        print(f"  ... and {len(chunks) - 12} more")

    # Find most dissonant pairs
    print()
    print("  Most structurally dissimilar pairs (resonance score):")
    pairs = transformer.find_dissonant_chunks(chunks, threshold=0.0)[:5]
    if pairs:
        for a, b, score in pairs:
            print(f"    {a.name} <-> {b.name}: {score:.3f}")
    else:
        print("    (none found)")

    # Run CodeSwarm
    print()
    print("  Running CodeSwarm (30 rounds, no drift)...")
    swarm = CodeSwarm(chunks, n_rounds=30)
    swarm.run()
    dissonant = swarm.get_dissonant_chunks(top_k=5)

    print("  Most structurally unusual chunks by gossip surprise:")
    for chunk, surprise in dissonant:
        marker = " <-- outlier" if surprise > 0.35 else ""
        print(f"    {chunk.name} ({chunk.chunk_type}): surprise={surprise:.4f}{marker}")
    print()


def demo_torch_benchmark() -> None:
    """Benchmark PyTorch backend vs NumPy across agent counts."""
    print("=" * 60)
    print("DEMO 6: PyTorch Backend Benchmark")
    print("=" * 60)

    try:
        import torch
        from src.experimental.torch_backend import (
            TorchResonanceBackend,
            TorchProjector,
            detect_torch_device,
            numpy_to_torch,
            torch_to_numpy,
        )
    except ImportError:
        print("  PyTorch not installed — skipping torch benchmark.")
        print("  Install with: pip install 'torch>=2.1.0'")
        print()
        return

    device = detect_torch_device()
    print(f"  PyTorch {torch.__version__}, Device: {device}")
    print(f"  torch.bitwise_count available: {hasattr(torch, 'bitwise_count')}")
    print()

    input_dim = 64
    projection_dim = 128
    seed = 42
    n_bits = 3

    projector = QJLProjector(input_dim, projection_dim, seed=seed)
    torch_proj = TorchProjector(projector, device="cpu")
    backend = TorchResonanceBackend(device="cpu")

    # Fair comparison: both do batch_resonance_scores (1-vs-N, 50 queries)
    # This is the typical hot path, not full pairwise.
    print("  Benchmark: 50 batch_resonance_scores queries against N agents")
    print()
    print("  N agents | NumPy packed |  Torch CPU | Speedup")
    print("  ---------+--------------+------------+--------")

    for n_agents in [100, 500, 1000, 2000]:
        rng = np.random.default_rng(seed)
        vectors = rng.standard_normal((n_agents, input_dim))
        states_np = projector.batch_project_and_quantize(vectors)
        packed = pack_quantized_batch(states_np)

        # NumPy packed path: 50 queries
        t0 = time.perf_counter()
        for _ in range(3):
            for i in range(50):
                batch_resonance_scores_packed(
                    packed[i], packed, n_dims=projection_dim, n_bits=n_bits
                )
        numpy_time = (time.perf_counter() - t0) / 3

        # Torch CPU path: 50 batch queries
        states_torch = numpy_to_torch(states_np, device="cpu")
        t0 = time.perf_counter()
        for _ in range(3):
            for i in range(50):
                backend.batch_resonance_scores(
                    states_torch[i], states_torch, n_bits=n_bits
                )
        torch_time = (time.perf_counter() - t0) / 3

        speedup = numpy_time / torch_time if torch_time > 0 else float("inf")
        print(
            f"  {n_agents:8d} | {numpy_time*1000:9.1f} ms | "
            f"{torch_time*1000:7.1f} ms | {speedup:.1f}x"
        )

    # Full pairwise comparison (smaller N to keep memory reasonable)
    print()
    print("  Full pairwise matrix (N×N):")
    from src.experimental.resonance_core import pairwise_resonance_matrix as _prm_np

    for n_agents in [100, 500]:
        rng = np.random.default_rng(seed)
        vectors = rng.standard_normal((n_agents, input_dim))
        states_np = projector.batch_project_and_quantize(vectors)

        t0 = time.perf_counter()
        _ = _prm_np(states_np)
        numpy_pw = time.perf_counter() - t0

        states_torch = numpy_to_torch(states_np, device="cpu")
        t0 = time.perf_counter()
        _ = backend.pairwise_resonance_matrix(states_torch, n_bits=n_bits)
        torch_pw = time.perf_counter() - t0

        speedup = numpy_pw / torch_pw if torch_pw > 0 else float("inf")
        print(f"    N={n_agents:4d}: NumPy {numpy_pw*1000:7.1f} ms, "
              f"Torch {torch_pw*1000:7.1f} ms ({speedup:.1f}x)")

    # MPS benchmark if available
    if device == "mps":
        print()
        print("  --- MPS (Apple Metal) ---")
        backend_mps = TorchResonanceBackend(device="mps")
        for n_agents in [100, 500]:
            rng = np.random.default_rng(seed)
            vectors = rng.standard_normal((n_agents, input_dim))
            states_np = projector.batch_project_and_quantize(vectors)
            states_mps = numpy_to_torch(states_np, device="mps")

            t0 = time.perf_counter()
            _ = backend_mps.pairwise_resonance_matrix(states_mps, n_bits=n_bits)
            torch.mps.synchronize()
            mps_time = time.perf_counter() - t0
            print(f"    N={n_agents:4d}, MPS pairwise: {mps_time*1000:.1f} ms")

    print()
    print("  Honest assessment: for small N (< 2000) on CPU, NumPy packed")
    print("  is competitive or faster. Torch gains appear in pairwise for")
    print("  large N (eliminating Python loop) and on CUDA devices.")
    print()


def demo_refactor_proposals() -> None:
    """Generate SPECULATIVE refactoring proposals from manifold dissonance."""
    print("=" * 60)
    print("DEMO 7: Resonance-Correcting Refactor Proposals")
    print("  (SPECULATIVE: manifold dissonance → refactoring)")
    print("=" * 60)

    target = Path(__file__).parent / "resonance_core.py"
    projector = QJLProjector(input_dim=64, projection_dim=128, seed=42)
    transformer = CodeTransformer(projector, state_dim=64)

    chunks = transformer.extract_chunks(str(target))
    transformer.project_chunks(chunks)

    print(f"  Target: {target.name}")
    print(f"  Chunks: {len(chunks)}")

    analyzer = RefactorAnalyzer(z_threshold=1.5)
    stats, proposals = analyzer.analyze_codebase(chunks)

    print(f"  Manifold: mean features across {stats.n_chunks} chunks")
    print()

    if not proposals:
        print("  No proposals — all chunks within normal manifold bounds.")
    else:
        print(f"  Top {min(5, len(proposals))} proposals:")
        print()
        for i, p in enumerate(proposals[:5], 1):
            print(f"  {i}. {p.chunk.name} ({p.chunk.chunk_type})")
            print(f"     Type: {p.refactor_type.name}")
            print(f"     {p.feature_name}={p.feature_value:.3f} "
                  f"(mean={p.manifold_mean:.3f}, z={p.confidence:.2f})")
            print(f"     {p.suggestion[:80]}...")
            print()

    print()


def demo_multi_file_manifold() -> None:
    """Scan src/experimental/ and analyze cross-file structural coherence."""
    print("=" * 60)
    print("DEMO 8: Multi-File Manifold Scan")
    print("=" * 60)

    exp_dir = Path(__file__).parent
    projector = QJLProjector(input_dim=64, projection_dim=128, seed=42)
    scanner = ManifoldScanner(
        projector, state_dim=64,
        exclude_patterns=["__init__"],
    )

    chunks = scanner.scan(str(exp_dir))
    print(f"  Directory: {exp_dir.name}/")
    print(f"  Files scanned: {len(scanner.file_summary)}")
    print(f"  Total chunks: {len(chunks)}")
    print()

    # Per-file summary
    print("  File                     | Chunks | Cohesion")
    print("  -------------------------+--------+---------")
    cohesion = scanner.get_file_cohesion_scores()
    for fpath, n in sorted(scanner.file_summary.items(), key=lambda x: -x[1]):
        fname = Path(fpath).name
        coh = cohesion.get(fpath, 0.0)
        print(f"  {fname[:25].ljust(25)} | {n:6d} | {coh:.4f}")

    # Cross-file dissonance
    print()
    cross_pairs = scanner.find_cross_file_dissonance(threshold=0.15)
    print(f"  Cross-file dissonant pairs (threshold=0.15): {len(cross_pairs)}")
    if cross_pairs:
        print("  Top 5:")
        for a, b, score in cross_pairs[:5]:
            fa = Path(a.file_path).name
            fb = Path(b.file_path).name
            print(f"    {fa}:{a.name} <-> {fb}:{b.name}: {score:.3f}")

    # Run CodeSwarm on all chunks
    print()
    if len(chunks) >= 2:
        print(f"  Running CodeSwarm on {len(chunks)} chunks (20 rounds)...")
        swarm = CodeSwarm(chunks, n_rounds=20)
        swarm.run()
        dissonant = swarm.get_dissonant_chunks(top_k=5)
        print("  Most structurally unusual across entire codebase:")
        for chunk, surprise in dissonant:
            fname = Path(chunk.file_path).name
            print(f"    {fname}:{chunk.name}: surprise={surprise:.4f}")
    print()


def demo_telemetry() -> None:
    """Generate ASCII telemetry visualizations."""
    print("=" * 60)
    print("DEMO 9: Telemetry Suite (ASCII)")
    print("=" * 60)

    telemetry = ResonanceTelemetry()
    print(f"  Matplotlib available: {telemetry.has_matplotlib}")
    print()

    # Build a small gossip scenario for telemetry data
    n_agents = 30
    input_dim = 64
    rng = np.random.default_rng(42)

    config = GossipConfig(
        n_agents=n_agents,
        state_dim=input_dim,
        projection_dim=128,
        fanout=4,
        dissonance_threshold=0.35,
        resync_strength=0.1,
        drift_magnitude=0.01,
        max_rounds=50,
        seed=42,
        pid_governor=ResonancePIDGovernor(
            target_surprise=0.15, Kp=0.5, Ki=0.01, Kd=0.1,
        ),
    )
    ctrl = GossipController(config)
    ctrl.initialize()
    ctrl.run(n_rounds=50)

    # 1. Convergence
    print("  --- Convergence Curve ---")
    telemetry.plot_convergence(ctrl, title="  Gossip Convergence (30 agents)")
    print()

    # 2. PID response
    print("  --- PID Response ---")
    surprises = [s.mean_surprise for s in ctrl.round_history]
    resync_vals = []
    # Re-run to collect effective_resync per round
    pid = ResonancePIDGovernor(target_surprise=0.15, Kp=0.5, Ki=0.01, Kd=0.1)
    for s in surprises:
        resync_vals.append(pid.update(s))
    telemetry.plot_pid_response(surprises, resync_vals, title="  PID Response")
    print()

    # 3. Surprise distribution (last round)
    if ctrl.round_history:
        last = ctrl.round_history[-1]
        print("  --- Surprise Distribution (last round) ---")
        telemetry.plot_surprise_distribution(
            last.agent_surprises,
            title="  Agent Surprise Distribution",
        )
        print()

    # 4. Mini heatmap
    print("  --- Resonance Heatmap (sample) ---")
    agents = list(ctrl.agents.values())
    states = np.array([a.projected_state for a in agents], dtype=np.uint8)
    from src.experimental.resonance_core import pairwise_resonance_matrix as _prm
    matrix = _prm(states)
    labels = [f"a{i}" for i in range(n_agents)]
    telemetry.plot_resonance_heatmap(matrix, labels, title="  30-Agent Heatmap")
    print()


def main() -> None:
    print()
    print("QJL Resonance — Experimental Module Demo (Phase 6)")
    print("Hypothesis: QJL-projected angular divergence as VFE surprise proxy")
    print()
    demo_jl_preservation()
    demo_xor_speed()
    demo_gossip_convergence()
    demo_stress_test()
    demo_code_analysis()
    demo_torch_benchmark()
    demo_refactor_proposals()
    demo_multi_file_manifold()
    demo_telemetry()
    print("=" * 60)
    print("All 9 demos complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
