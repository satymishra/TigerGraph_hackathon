"""
TigerGraph client — shared connection utility for all pipeline scripts.
Handles token auth, GSQL execution, and REST upserts.
"""

import os
import requests
import json
import time

def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

_load_env()

HOST   = os.environ["TG_HOST"]
SECRET = os.environ["TG_SECRET"]
GRAPH  = os.environ["TG_GRAPH"]


class TGClient:
    def __init__(self):
        self.host   = HOST
        self.graph  = GRAPH
        self.secret = SECRET
        self.token  = None
        self.token_expiry = 0
        self._refresh_token()

    def _refresh_token(self, retries: int = 5):
        last_err = None
        for attempt in range(retries):
            try:
                r = requests.post(
                    f"{self.host}/gsql/v1/tokens",
                    json={"secret": self.secret, "lifetime": 36000},
                    timeout=30,
                )
                r.raise_for_status()
                data = r.json()
                self.token = data.get("results", {}).get("token") or data["token"]
                self.token_expiry = time.time() + 35000
                return
            except Exception as e:
                last_err = e
                if attempt < retries - 1:
                    wait = 5 * (attempt + 1)
                    print(f"  TG auth attempt {attempt+1} failed, retry in {wait}s: {e}")
                    time.sleep(wait)
        raise RuntimeError(f"TigerGraph auth failed after {retries} attempts: {last_err}")

    def _headers(self):
        if time.time() > self.token_expiry:
            self._refresh_token()
        return {"Authorization": f"Bearer {self.token}"}

    # ── GSQL ──────────────────────────────────────────────────
    def gsql(self, statement: str) -> str:
        r = requests.post(
            f"{self.host}/gsql/v1/statements",
            headers={**self._headers(), "Content-Type": "text/plain"},
            data=statement.encode("utf-8"),
            timeout=120,
        )
        return r.text

    # ── Upsert vertices ───────────────────────────────────────
    def upsert_vertices(self, vertex_type: str, vertices: list[dict]) -> dict:
        """
        vertices: list of dicts with 'id' key and attribute keys.
        TG REST++ upsert format.
        """
        payload = {vertex_type: {}}
        for v in vertices:
            vid = v.pop("id")
            payload[vertex_type][vid] = {"attributes": v}

        r = requests.post(
            f"{self.host}/restpp/graph/{self.graph}",
            headers=self._headers(),
            json={"vertices": payload},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()

    # ── Upsert edges ──────────────────────────────────────────
    def upsert_edges(self, edge_type: str, from_type: str, to_type: str,
                     edges: list[dict]) -> dict:
        """
        edges: list of dicts with 'from_id', 'to_id', and optional attribute keys.
        """
        payload = {from_type: {}}
        for e in edges:
            fid = e.pop("from_id")
            tid = e.pop("to_id")
            if fid not in payload[from_type]:
                payload[from_type][fid] = {edge_type: {to_type: {}}}
            payload[from_type][fid][edge_type][to_type][tid] = {"attributes": e}

        r = requests.post(
            f"{self.host}/restpp/graph/{self.graph}",
            headers=self._headers(),
            json={"edges": payload},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()

    # ── REST query ────────────────────────────────────────────
    def query(self, query_name: str, params: dict = None) -> dict:
        r = requests.get(
            f"{self.host}/restpp/query/{self.graph}/{query_name}",
            headers=self._headers(),
            params=params or {},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    # ── GET vertices by ID ────────────────────────────────────
    def get_vertex(self, vertex_type: str, vid: str) -> dict:
        r = requests.get(
            f"{self.host}/restpp/graph/{self.graph}/vertices/{vertex_type}/{vid}",
            headers=self._headers(),
            timeout=15,
        )
        r.raise_for_status()
        return r.json()

    # ── GET vertex count ──────────────────────────────────────
    def count_vertices(self, vertex_type: str = None) -> dict:
        url = f"{self.host}/restpp/graph/{self.graph}/vertices"
        if vertex_type:
            url += f"/{vertex_type}?count_only=true"
        r = requests.get(url, headers=self._headers(), timeout=15)
        return r.json()

    # ── Raw REST call ─────────────────────────────────────────
    def get(self, path: str, params: dict = None) -> requests.Response:
        return requests.get(
            f"{self.host}{path}",
            headers=self._headers(),
            params=params or {},
            timeout=30,
        )

    def post(self, path: str, payload: dict, timeout: int = 60) -> requests.Response:
        return requests.post(
            f"{self.host}{path}",
            headers=self._headers(),
            json=payload,
            timeout=timeout,
        )


# ── Quick smoke test ──────────────────────────────────────────
if __name__ == "__main__":
    print("Connecting to TigerGraph...")
    client = TGClient()
    print(f"Connected. Token: {client.token[:20]}...")

    # Check graph exists via GSQL ls
    result = client.gsql(f"USE GRAPH {GRAPH}\nls")
    print(f"\nGraph schema summary:\n{result[:800]}")
