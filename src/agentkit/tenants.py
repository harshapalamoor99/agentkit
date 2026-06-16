"""Multi-tenant brand metadata store + portfolio isolation (AC-7, AC-10).

Each corporate tenant (property-management company) has its own brand voice, legal
footer, and the set of properties it operates. At prompt-construction time we inject
ONLY the requesting tenant's guidelines, and we isolate few-shot retrieval to that
tenant's own records — a record from another tenant can never leak its brand, pricing,
or property names into this tenant's message (antitrust / portfolio isolation).

In production this is a metadata service / row-level-secured table keyed by tenant_id;
here it is an in-process registry seeded with a couple of tenants and overridable via a
JSON file (TENANT_REGISTRY_PATH) so it is data-driven, not hardcoded per request.
"""
from __future__ import annotations

import json
import os
from typing import Any

_DEFAULT_TENANTS: dict[str, dict[str, Any]] = {
    "oakridge_pm": {
        "tenant_id": "oakridge_pm",
        "display_name": "Oak Ridge Living",
        "brand_voice": "warm, concise, neighborly; no hype or pushy urgency",
        "legal_footer": "Oak Ridge is an Equal Housing Opportunity provider.",
        "properties": ["Oak Ridge Apartments", "Oak Ridge Townhomes"],
    },
    "summit_residential": {
        "tenant_id": "summit_residential",
        "display_name": "Summit Residential",
        "brand_voice": "upbeat, modern, amenity-forward",
        "legal_footer": "Summit Residential. Equal Housing Opportunity.",
        "properties": ["Summit Heights", "Summit Lofts"],
    },
}

_DEFAULT_TENANT_ID = "oakridge_pm"


def _load_registry() -> dict[str, dict[str, Any]]:
    path = os.getenv("TENANT_REGISTRY_PATH")
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {k: v for k, v in data.items() if isinstance(v, dict)}
        except (OSError, ValueError):
            pass
    return dict(_DEFAULT_TENANTS)


_REGISTRY = _load_registry()


def _tenant_id_for_record(record: dict[str, Any]) -> str | None:
    """Resolve the tenant id from explicit field, else by property ownership."""
    tid = record.get("tenant_id") or (record.get("input", {}) or {}).get("tenant_id")
    if isinstance(tid, str) and tid in _REGISTRY:
        return tid
    prop = (record.get("input", {}) or {}).get("property_name")
    if isinstance(prop, str):
        for t in _REGISTRY.values():
            if prop in t.get("properties", []):
                return t["tenant_id"]
    return None


def resolve_tenant(record: dict[str, Any]) -> dict[str, Any]:
    """Return the brand metadata for this record's tenant (never another tenant's)."""
    tid = _tenant_id_for_record(record) or _DEFAULT_TENANT_ID
    return _REGISTRY.get(tid, _REGISTRY[_DEFAULT_TENANT_ID])


def same_tenant(record_a: dict[str, Any], record_b: dict[str, Any]) -> bool:
    return (_tenant_id_for_record(record_a) or _DEFAULT_TENANT_ID) == \
           (_tenant_id_for_record(record_b) or _DEFAULT_TENANT_ID)


def foreign_property_names(record: dict[str, Any]) -> list[str]:
    """Property names belonging to OTHER tenants — these must never appear in output
    (cross-tenant leakage / antitrust isolation, AC-7/AC-10)."""
    tid = _tenant_id_for_record(record) or _DEFAULT_TENANT_ID
    names: list[str] = []
    for t in _REGISTRY.values():
        if t["tenant_id"] == tid:
            continue
        names.extend(t.get("properties", []))
    return names
