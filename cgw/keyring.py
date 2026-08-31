"""Key rotation and health tracking. In-memory only; restart = clean slate."""

import threading
import time

COOLDOWN_BASE = 60
COOLDOWN_MAX = 900
MAX_ATTEMPTS = 5

# a transport failure (timeout, DNS, reset, client hangup) says nothing about
# the key, so it gets a short flat cooldown instead of the exponential ladder
TRANSPORT_COOLDOWN = 5

# statuses that mean "this key is wrong", not "this key is busy"
DEAD_STATUSES = (401, 403)


class KeyState:
    __slots__ = ("label", "failures", "cooldown_until", "dead", "last_status")

    def __init__(self, label):
        self.label = label
        self.failures = 0
        self.cooldown_until = 0.0
        self.dead = False
        self.last_status = None

    def healthy(self, now=None):
        now = time.time() if now is None else now
        return (not self.dead) and now >= self.cooldown_until

    def status_text(self, now=None):
        now = time.time() if now is None else now
        if self.dead:
            return "dead"
        if now < self.cooldown_until:
            return "cool %ds" % int(self.cooldown_until - now)
        return "ok"


class KeyRing:
    """One per provider. Rebuilt when the provider's key list changes."""

    def __init__(self, keys, rotation="fill_first"):
        self.rotation = rotation if rotation in ("fill_first", "round_robin") else "fill_first"
        self.keys = [dict(k) for k in keys]
        self.state = [KeyState(k.get("label") or "k%d" % (i + 1)) for i, k in enumerate(self.keys)]
        self.cursor = 0
        self.lock = threading.Lock()

    @staticmethod
    def signature(keys, rotation):
        return (rotation, tuple((k.get("key"), k.get("label"), bool(k.get("enabled", True))) for k in keys))

    def _enabled_indexes(self):
        return [i for i, k in enumerate(self.keys) if k.get("enabled", True)]

    def try_order(self, now=None):
        """Indexes to attempt, best first, capped at MAX_ATTEMPTS."""
        now = time.time() if now is None else now
        with self.lock:
            enabled = self._enabled_indexes()
            healthy = [i for i in enabled if self.state[i].healthy(now)]
            if self.rotation == "round_robin" and healthy:
                # The cursor counts dispatches, not positions in `healthy`:
                # modulo-ing it by a list that shrinks as keys cool down made
                # rotation jump around and revisit the same key.
                start = self.cursor % len(healthy)
                healthy = healthy[start:] + healthy[:start]
                self.cursor = (self.cursor + 1) % max(1, len(enabled))
            order = list(healthy)
            if not order:
                # everything is cooling down: try the one that recovers soonest
                cooling = [i for i in enabled if not self.state[i].dead]
                cooling.sort(key=lambda i: self.state[i].cooldown_until)
                order = cooling[:1]
            return order[:MAX_ATTEMPTS]

    def key_at(self, idx):
        return self.keys[idx]["key"]

    def report_success(self, idx):
        with self.lock:
            st = self.state[idx]
            st.failures = 0
            st.cooldown_until = 0.0
            st.last_status = 200

    def report_failure(self, idx, status, now=None):
        now = time.time() if now is None else now
        with self.lock:
            st = self.state[idx]
            st.last_status = status
            if status in DEAD_STATUSES:
                st.dead = True
                return
            st.failures += 1
            if not status:
                # transport-level: not the key's fault, so don't punish it for
                # a minute — a client disconnect used to cost a good key 60s
                st.cooldown_until = now + TRANSPORT_COOLDOWN
                return
            st.cooldown_until = now + min(COOLDOWN_BASE * (2 ** (st.failures - 1)), COOLDOWN_MAX)

    def revive(self, idx):
        with self.lock:
            st = self.state[idx]
            st.dead = False
            st.failures = 0
            st.cooldown_until = 0.0

    def revive_all(self):
        with self.lock:
            for st in self.state:
                st.dead = False
                st.failures = 0
                st.cooldown_until = 0.0

    def summary(self, now=None):
        now = time.time() if now is None else now
        return [
            {"label": st.label, "status": st.status_text(now), "last_status": st.last_status}
            for st in self.state
        ]


class Registry:
    """Holds a KeyRing per provider name, rebuilding on config change."""

    def __init__(self):
        self.rings = {}
        self.sigs = {}
        self.lock = threading.Lock()

    def get(self, provider):
        name = provider["name"]
        sig = KeyRing.signature(provider.get("keys") or [], provider.get("rotation", "fill_first"))
        with self.lock:
            if self.sigs.get(name) != sig:
                self.rings[name] = KeyRing(provider.get("keys") or [], provider.get("rotation", "fill_first"))
                self.sigs[name] = sig
            return self.rings[name]

    def snapshot(self):
        with self.lock:
            return {name: ring.summary() for name, ring in self.rings.items()}

    def revive(self, name=None):
        """Clear dead/cooldown state for one provider's ring, or all of them.

        Returns the number of rings cleared. A 401/403 marks a key dead
        permanently, which is right for a wrong key and wrong for a provider
        blip, so there has to be a way back that isn't a restart.
        """
        with self.lock:
            targets = [r for n, r in self.rings.items() if name in (None, n)]
        for ring in targets:
            ring.revive_all()
        return len(targets)
