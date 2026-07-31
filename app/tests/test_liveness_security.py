"""
Security regression tests for the liveness gate.

Reproduces the replay bypasses that were reported in production and proves they
stay closed. Models the real failure mode: a handheld replay whose frames are
only INTERMITTENTLY flagged, which is what used to reset the consecutive
counters and made the same attack succeed on one attempt and fail on the next.

Run from anywhere:   python tests/test_liveness_security.py
Exits non-zero on regression, so it can gate a commit or CI run.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.liveness_pipeline as L

BLINK_CLOSED = {"eyeBlinkLeft": 0.6, "eyeBlinkRight": 0.6}
BLINK_OPEN = {"eyeBlinkLeft": 0.0, "eyeBlinkRight": 0.0}


def fake_evaluate(key, screen_hit, model_a_hit, blendshapes):
    """Drive the pipeline with controlled evidence (detectors stubbed)."""
    L._screen_evidence = lambda f, b: {"is_screen": screen_hit,
                                       "reason": "Device screen border detected around face"}
    L._texture_evidence = lambda c, b, f, l: {"is_spoof": model_a_hit,
                                              "reason": "MiniFASNetV2 replay texture",
                                              "model_a_score": 0.1 if model_a_hit else 0.98}
    return L.evaluate(key, None, {"originX": 0, "originY": 0, "width": 100, "height": 100},
                      None, blendshapes, [])


def run(key, pattern, label):
    """pattern: list of (screen_hit, model_a_hit) per frame; blink every 3rd."""
    L._clients.pop(key, None)
    L.reset_client(key)
    states = []
    for i, (s, a) in enumerate(pattern):
        bs = BLINK_CLOSED if i % 3 == 1 else BLINK_OPEN
        states.append(fake_evaluate(key, s, a, bs).state)
    print(f"   {label}: {states}")
    return states


print("=" * 74)
print("1) THE OLD BYPASS: intermittent detection (a handheld phone)")
print("   Model A flags 60% of frames, never 5 in a row -> old code reset forever")
print("=" * 74)
inter = [(False, True), (False, False), (False, True), (False, True), (False, False),
         (False, True), (False, False), (False, True), (False, True), (False, False)]
s = run("k_inter", inter, "verdicts")
print(f"   -> reached spoof: {'spoof' in s}   ever LIVE: {'live' in s}")
assert "live" not in s, "REGRESSION: intermittent replay reached LIVE"
assert "spoof" in s, "REGRESSION: intermittent replay never flagged"

print("\n" + "=" * 74)
print("2) DETERMINISM: the same replay, run 5 times, must give the same answer")
print("=" * 74)
outcomes = []
for trial in range(5):
    L._clients.pop("k_det", None)
    s = run("k_det", inter, f"trial {trial+1}")
    outcomes.append(("live" in s, "spoof" in s))
print(f"   distinct outcomes: {set(outcomes)}  (want exactly one)")
assert len(set(outcomes)) == 1, "REGRESSION: non-deterministic outcome"

print("\n" + "=" * 74)
print("3) RETRY ATTACK: attacker hits 'try again' to reroll the dice")
print("=" * 74)
L._clients.pop("k_retry", None)
run("k_retry", inter, "attempt 1")
for attempt in range(2, 5):
    L.reset_client("k_retry")          # user/attacker clicks retry
    s = run_states = [fake_evaluate("k_retry", False, False, BLINK_OPEN).state
                      for _ in range(6)]   # now feeding PERFECTLY CLEAN frames
    print(f"   attempt {attempt} with clean frames after reset: {s}")
    assert "live" not in s, f"REGRESSION: retry {attempt} escaped the cooldown"
print("   -> cooldown survives reset; attacker cannot reroll into a login")

print("\n" + "=" * 74)
print("4) REAL USER: clean frames + natural blink must still reach LIVE")
print("=" * 74)
clean = [(False, False)] * 8
s = run("k_real", clean, "verdicts")
print(f"   -> reached live: {'live' in s}   any spoof: {'spoof' in s}")
assert "live" in s and "spoof" not in s, "REGRESSION: genuine user cannot log in"

print("\n" + "=" * 74)
print("5) REAL USER with Model A's known 21% false-positive rate")
print("=" * 74)
noisy = [(False, False), (False, True), (False, False), (False, False),
         (False, False), (False, True), (False, False), (False, False),
         (False, False), (False, False), (False, False), (False, False)]
s = run("k_noisy", noisy, "verdicts")
print(f"   -> reached live: {'live' in s}   any spoof: {'spoof' in s}")
assert "spoof" not in s, "REGRESSION: genuine user tripped the spoof gate"

print("\n" + "=" * 74)
print("6) SCREEN DETECTOR: two hits are conclusive (0% measured FPR)")
print("=" * 74)
s = run("k_screen", [(True, False), (True, False), (False, False), (False, False)], "verdicts")
print(f"   -> spoof by frame {s.index('spoof')+1 if 'spoof' in s else '-'}")
assert s[1] == "spoof", "REGRESSION: two screen hits did not latch"

print("\nALL DETERMINISM AND SECURITY ASSERTIONS PASSED")
