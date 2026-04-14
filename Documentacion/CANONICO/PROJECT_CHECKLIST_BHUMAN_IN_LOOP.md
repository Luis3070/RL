# Project Checklist: BHuman In Loop

Documento relacionado:

- [BHUMAN_IN_LOOP_COMPACT_GUIDE.md](file:///c:/ALEJO/RL/Documentacion/CANONICO/BHUMAN_IN_LOOP_COMPACT_GUIDE.md)
  Punto de entrada corto para smokes canonicos y documentacion minima.

- [DOCUMENTATION_AND_SMOKE_GOVERNANCE.md](file:///c:/ALEJO/RL/Documentacion/CANONICO/DOCUMENTATION_AND_SMOKE_GOVERNANCE.md)
  Define que documentos son canonicos, cuales son referencia y cuales son historicos.

## 1. Bridge and integration

- [x] Real `RLBridge` executable resolved from `SabanaHerons2026`.
- [x] `cfg_root` resolved to real BHuman `Config`.
- [x] Real backend selected inside `NaoSoccerEnv`.
- [x] Transport stable in `pipe_process`.
- [x] Bridge `location` aligned to `4v4_Scaled` instead of legacy `Default`.
- [x] Bridge baseline made explicit as `bridge_mode = persistent_2d_cognition`.
- [x] `contract_validation_rate = 0` in practical runs.
- [x] `non_pipe_rate = 0` in practical runs.
- [x] Bridge frame counter resets cleanly on `env.reset()`.
- [x] Bridge provenance fields exposed in `infos`.

## 2. Semantic honesty

- [x] `shoot` validated in `env.step()` with real positive takeover overlap.
- [x] `duel` validated in `env.step()` with real positive takeover overlap.
- [x] `dribble` validated in `env.step()` as real bridge-driven semantic output.
- [x] `pass` validated in `env.step()` as real bridge-driven semantic output.
- [x] At least one real takeover reaches `completed=True` inside `env.step()`.
- [x] `macro_type` in `infos` reflects real sidecar macro when takeover is inactive.
- [x] `takeover_macro_type` preserved separately to avoid semantic leakage.
- [x] `current_tactic` normalized to `*_4v4_scaled`.

## 3. Curriculum fidelity

- [x] Reduced-agent stages preserve 4v4 own-team context in the sidecar.
- [x] Stage 0 still trains positioning only, not ball-control actions.
- [x] Stage advancement still follows contract-defined metric.
- [x] Stage 0 gate remains contract-aligned and is not relaxed by convenience heuristics.

## 4. Reward and optimization

- [x] Dense `handoff_alignment` shaping added from real contract geometry.
- [x] Stage 0 shaping weights retuned toward handoff pose acquisition.
- [x] PPO reward scale aligned with intended fixed scaling.
- [x] Exploration reduced to stabilize early approach behavior.
- [ ] Re-run practical training after latest reward/config changes.

## 5. Training evidence

- [x] 50-episode practical run reaches stable `TkReady ~ 0.80`.
- [x] Stability remains near `1.0` under real backend.
- [x] `handoff_reached` and `TkReady` are tracked as separate signals to avoid semantic leakage.
- [ ] Achieve `TkReady >= 0.95` over the full stage-0 window.
- [ ] Observe non-zero `TkSucc` in practical training, not just targeted heuristic probes.

## 6. Open gaps

- [x] Add a real smoke that asserts successful takeover completion, not only takeover overlap/start.
- [ ] Audit `MotionInfo.last_kick_type / last_kick_timestamp` under real `shoot/pass` takeovers.
- [ ] Current persistent bridge still runs with `scenario=2D`, so `Motion` thread stays unavailable and `MotionInfo` evidence for `shoot/pass` is still not surfacing.
- [ ] Find stable positive env fixtures where `pass` overlaps `handoff_ready/takeover_active`.
- [ ] Find stable positive env fixtures where `dribble` overlaps `handoff_ready/takeover_active`.
- [ ] Profile long-run bridge latency under many episode resets.
- [ ] Decide whether large-scale parallel envs are worth the added bridge complexity.
- [ ] Keep heavy real-backend smokes serialized; parallel execution can force `contract_validation`.

## 7. Go / no-go for closing this phase

- [x] Real BHuman transport and provenance are operational.
- [x] Semantic contract is preserved end-to-end.
- [x] Physics-effect takeover path exists for real backend.
- [ ] Stage 0 curriculum still needs empirical confirmation of the final `0.95` gate.
- [ ] Practical training still needs repeated evidence that takeover success is learnable, not only reachable.
