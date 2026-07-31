"""
Liveness Pipeline — the single authority on "is this a real, present person?"

One responsibility: turn a stream of frames into exactly one verdict. It performs
no face recognition and knows nothing about identity. It is the gatekeeper that
runs BEFORE recognition, so a spoof can never reach the matcher.

    LIVE     — positively confirmed real and present. Recognition may proceed.
    SPOOF    — a display or print is being shown. Stop. Never recognise.
    PENDING  — not yet confirmed. Also the default: absence of evidence is NOT
               liveness.

THREE DESIGN RULES, each fixing a way a replay previously got through:

  1. EVIDENCE ACCUMULATES, IT DOES NOT RESET.
     The previous version counted *consecutive* spoof frames and zeroed the
     counter on any clean-looking frame. A handheld phone video varies frame to
     frame (motion blur, autofocus, glare), so the counter kept resetting and
     whether it ever reached the threshold was luck — the same replay could pass
     one attempt and fail the next. Evidence is now a weighted score that only
     decays slowly, so intermittent detections still add up.

  2. THE VERDICT LATCHES, WITH A COOLDOWN.
     Once an attempt is judged a spoof the client stays refused for
     SPOOF_COOLDOWN_SECONDS even across "try again". Otherwise an attacker just
     retries until a lucky frame sequence lets them through, which is
     non-deterministic by construction.

  3. FAIL CLOSED.
     LIVE requires positive corroboration — a blink AND enough frames AND a
     suspicion score near zero. Previously the pipeline returned LIVE simply
     because nothing had fired, treating "no evidence" as proof of life. For an
     attendance system that is backwards.

Evidence weights reflect each signal's MEASURED reliability, not intuition:

    screen_detect   0.0% false positives on 145 genuine faces  -> weight 1.2
    MiniFASNetV2   20.7% false positives on the same set       -> weight 0.5

so the trustworthy signal trips the gate in two frames while the noisy one needs
sustained agreement and cannot fire on a single unlucky frame.

Blink is deliberately NOT an evidence source for liveness on its own. A replayed
video blinks. It is a necessary condition consulted last, and any blink observed
while spoof evidence is present is discarded outright.
"""

import time

from utils import screen_detect
from utils.anti_spoofing import get_anti_spoof_engine

# ── verdict constants ──────────────────────────────────────────
LIVE = "live"
SPOOF = "spoof"
PENDING = "pending"

# ── evidence weights (see module docstring for the measurements) ──
WEIGHT_SCREEN = 1.2      # 0% measured FPR: two hits are conclusive
WEIGHT_MODEL_A = 0.5     # 21% measured FPR: needs sustained agreement
CLEAN_FRAME_DECAY = 0.10 # slow, so intermittent detections still accumulate

# A partial identity match is strong evidence on its own: genuine live matches
# start at 0.638 and true strangers top out at 0.43, so the band between them
# is where an enrolled face lands when it arrives through a screen. One
# observation latches the gate.
WEIGHT_PARTIAL_MATCH = 2.0

SPOOF_THRESHOLD = 2.0    # suspicion at which the attempt is latched as a spoof
LIVE_MAX_SUSPICION = 0.5 # must be near zero to be confirmed live

# Refuse a client for this long after a spoof, across retries, so an attacker
# cannot simply try again until a lucky frame sequence gets through.
SPOOF_COOLDOWN_SECONDS = 8.0

# Fail-closed: this many frames must be assessed before LIVE is possible.
MIN_FRAMES_FOR_LIVE = 4

# Blink
BLINK_FRESHNESS_SECONDS = 4.0
BLINK_CLOSE_THRESHOLD = 0.22
BLINK_OPEN_THRESHOLD = 0.12

STALE_AFTER_SECONDS = 30.0


class Verdict:
    """The single result object the pipeline emits."""

    def __init__(self, state, reason="", detail=None):
        self.state = state
        self.reason = reason
        self.detail = detail or {}

    @property
    def is_spoof(self):
        return self.state == SPOOF

    @property
    def is_live(self):
        return self.state == LIVE

    @property
    def is_pending(self):
        return self.state == PENDING

    def __repr__(self):
        return f"<Verdict {self.state} {self.reason!r}>"


class _ClientState:
    """Per-client accumulated evidence. Liveness is inherently multi-frame."""

    def __init__(self):
        self.suspicion = 0.0
        self.frames = 0
        self.screen_hits = 0
        self.model_a_hits = 0
        self.eye_closed = False
        self.last_blink_at = 0.0
        self.spoof_latched_at = 0.0
        self.spoof_reason = ""
        self.last_seen = time.time()

    def in_cooldown(self):
        return (self.spoof_latched_at > 0.0 and
                (time.time() - self.spoof_latched_at) < SPOOF_COOLDOWN_SECONDS)


_clients = {}


def _get_client(key):
    now = time.time()
    for k, st in list(_clients.items()):
        # never expire a client still inside its spoof cooldown
        if now - st.last_seen > STALE_AFTER_SECONDS and not st.in_cooldown():
            _clients.pop(k, None)
    st = _clients.get(key)
    if st is None:
        st = _ClientState()
        _clients[key] = st
    st.last_seen = now
    return st


def reset_client(key):
    """
    Begin a fresh attempt.

    A spoof cooldown deliberately SURVIVES this call — otherwise "try again"
    would clear the evidence and let an attacker retry until a lucky sequence
    of frames slipped through.
    """
    st = _clients.get(key)
    latched_at, reason = (st.spoof_latched_at, st.spoof_reason) if st else (0.0, "")

    fresh = _ClientState()
    if latched_at and (time.time() - latched_at) < SPOOF_COOLDOWN_SECONDS:
        fresh.spoof_latched_at = latched_at
        fresh.spoof_reason = reason
    _clients[key] = fresh

    try:
        get_anti_spoof_engine().reset()
    except Exception:
        pass


def clear_client(key):
    """Fully drop a client's state — used after a successful login."""
    _clients.pop(key, None)


def flag_partial_match(key):
    """
    Recognition observed a face that matches an enrolled identity only
    partially — the signature of that person arriving through a screen.

    Fed back as strong evidence and latched, so the next frame returns SPOOF
    from the gate itself and recognition stops being consulted at all.
    """
    st = _get_client(key)
    st.suspicion += WEIGHT_PARTIAL_MATCH
    st.last_blink_at = 0.0
    if st.suspicion >= SPOOF_THRESHOLD and not st.spoof_latched_at:
        st.spoof_latched_at = time.time()
        st.spoof_reason = ("Face matches an enrolled user only partially, "
                           "consistent with a screen replay")
        print(f"[Liveness] SPOOF latched via partial match "
              f"(suspicion={st.suspicion:.2f})", flush=True)


# ──────────────────────────────────────────────────────────────
# Evidence gathering — each helper answers exactly one question
# ──────────────────────────────────────────────────────────────

def _screen_evidence(frame, bbox):
    """Is the face being presented on a display (phone / tablet / monitor)?"""
    try:
        return screen_detect.check(frame, bbox)
    except Exception as e:
        print(f"[Liveness] screen detector error: {e}", flush=True)
        return {"is_screen": False, "reason": ""}


def _texture_evidence(face_crop, bbox, frame, landmarks):
    """Does the passive texture/frequency model consider this a replay?"""
    try:
        return get_anti_spoof_engine().check(
            face_crop=face_crop, landmarks=landmarks,
            bbox=bbox, full_frame=frame)
    except Exception as e:
        print(f"[Liveness] anti-spoof engine error: {e}", flush=True)
        return {"is_spoof": False, "reason": "", "model_a_score": 1.0}


def _update_blink(state, blendshapes):
    """
    Track the open -> closed -> open transition of a real blink.

    Returns whether a blink was seen recently. Whether that MEANS anything is
    the pipeline's decision — during a spoof it does not.
    """
    bs = blendshapes or {}
    level = max(bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0))
    now = time.time()

    if not state.eye_closed and level >= BLINK_CLOSE_THRESHOLD:
        state.eye_closed = True
    elif state.eye_closed and level <= BLINK_OPEN_THRESHOLD:
        state.eye_closed = False
        state.last_blink_at = now

    return (now - state.last_blink_at) <= BLINK_FRESHNESS_SECONDS


# ──────────────────────────────────────────────────────────────
# The pipeline
# ──────────────────────────────────────────────────────────────

def evaluate(client_key, frame, bbox, face_crop, blendshapes, landmarks):
    """
    Decide whether this frame shows a real, present person.

    Returns a Verdict. Recognition must run only when verdict.is_live.
    """
    state = _get_client(client_key)

    # ── Latched spoof: same input always gives the same answer ──
    if state.in_cooldown():
        remaining = SPOOF_COOLDOWN_SECONDS - (time.time() - state.spoof_latched_at)
        print(f"[Liveness] LATCHED spoof, {remaining:.1f}s cooldown remaining", flush=True)
        return Verdict(SPOOF, state.spoof_reason or "Screen replay detected",
                       {"source": "latched", "cooldown_remaining": round(remaining, 1)})

    state.frames += 1

    # ── Evidence 1: screen presentation (most reliable) ────────
    screen = _screen_evidence(frame, bbox)
    screen_hit = bool(screen.get("is_screen"))
    if screen_hit:
        state.screen_hits += 1
        state.suspicion += WEIGHT_SCREEN

    # ── Evidence 2: passive texture model (noisier) ────────────
    texture = _texture_evidence(face_crop, bbox, frame, landmarks)
    model_a_hit = bool(texture.get("is_spoof"))
    if model_a_hit:
        state.model_a_hits += 1
        state.suspicion += WEIGHT_MODEL_A

    model_a_score = texture.get("model_a_score", 1.0)

    # A clean frame decays suspicion slowly — it never wipes it.
    if not screen_hit and not model_a_hit:
        state.suspicion = max(0.0, state.suspicion - CLEAN_FRAME_DECAY)

    # ── Evidence 3: blink (tracked always, trusted only if clean) ──
    has_blink = _update_blink(state, blendshapes)

    print(f"[Liveness] frame={state.frames} suspicion={state.suspicion:.2f}/{SPOOF_THRESHOLD} "
          f"screen_hits={state.screen_hits} modelA_hits={state.model_a_hits} "
          f"modelA={model_a_score:.3f} blink={has_blink}", flush=True)

    # ── Decision 1: accumulated evidence latches a spoof ────────
    if state.suspicion >= SPOOF_THRESHOLD:
        reason = (screen.get("reason") or texture.get("reason")
                  or "Screen replay detected")
        state.spoof_latched_at = time.time()
        state.spoof_reason = reason
        state.last_blink_at = 0.0      # a blink seen on a screen is a video's blink
        print(f"[Liveness] SPOOF latched: {reason}", flush=True)
        return Verdict(SPOOF, reason,
                       {"source": "accumulated", "suspicion": round(state.suspicion, 2),
                        "screen_hits": state.screen_hits,
                        "model_a_hits": state.model_a_hits})

    # ── Decision 2: fail closed — LIVE needs positive proof ────
    if state.frames < MIN_FRAMES_FOR_LIVE:
        return Verdict(PENDING, "Verifying liveness",
                       {"frames": state.frames, "needed": MIN_FRAMES_FOR_LIVE})

    if state.suspicion > LIVE_MAX_SUSPICION:
        return Verdict(PENDING, "Liveness inconclusive",
                       {"suspicion": round(state.suspicion, 2)})

    if not has_blink:
        return Verdict(PENDING, "Awaiting a natural blink",
                       {"model_a_score": model_a_score})

    return Verdict(LIVE, "Real live person confirmed",
                   {"suspicion": round(state.suspicion, 2),
                    "model_a_score": model_a_score})
