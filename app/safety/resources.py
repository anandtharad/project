"""Crisis and emergency resources by region (spec §12.4).

The resources live in a versioned, cited file (`resources_v1.toml`, next to
this module): for each supported region (IN, US, GB) and a generic fallback,
the emergency number and crisis lines, each with name, contact, hours, the
official source it was checked against and the date. The generic fallback
names no number ("your local emergency number"): no number works
everywhere. A list of known-outdated contacts (the closed Samaritans email
service, the retired KIRAN line) is kept for the output validator.

Resources are application text: code renders them into the reply guidance
and the fallback replies, so a model never has to recall a number. The file
is loaded and validated as a whole; any problem is a `ResourceError`, and an
unknown region code is an `UnknownRegionError` (both at startup).
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlsplit

RESOURCES_FILE = Path(__file__).with_name("resources_v1.toml")

# Region code for "no region configured" (SAFETY_REGION="").
GENERIC = ""
SUPPORTED_REGIONS: tuple[str, ...] = ("IN", "US", "GB")
_FILE_GENERIC = "generic"
# Hours that are not shown.
UNSTATED_HOURS = frozenset({"not stated by the source", "not applicable"})
# At most this many resources are shown at once (Samaritans' guidance for
# technology companies: two or three).
MAX_SHOWN = 3

Verification = Literal["official_page", "official_doc", "search_index"]
_VERIFICATIONS: frozenset[str] = frozenset({"official_page", "official_doc", "search_index"})
# Find A Helpline's terms forbid using its data or embedding it.
_FORBIDDEN_SOURCES = ("findahelpline",)

_URL = re.compile(r"https?://[^\s)>\]]+|www\.[^\s)>\]]+")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# A phone-like number: digit groups joined by single spaces or dashes.
_PHONE = re.compile(r"(?<![\w+])\+?\d+(?:[ -]\d+)*(?![\w])")
_SEMVER = re.compile(r"\d+\.\d+\.\d+")


class ResourceKind(StrEnum):
    EMERGENCY = "emergency"
    CRISIS_LINE = "crisis_line"
    TEXT_LINE = "text_line"
    ABUSE_SUPPORT = "abuse_support"
    DIRECTORY = "directory"


# When a region has no resource of a kind, the first substitute it has.
_SUBSTITUTES: dict[ResourceKind, tuple[ResourceKind, ...]] = {
    ResourceKind.EMERGENCY: (),
    ResourceKind.CRISIS_LINE: (ResourceKind.DIRECTORY,),
    ResourceKind.TEXT_LINE: (),
    ResourceKind.ABUSE_SUPPORT: (ResourceKind.CRISIS_LINE, ResourceKind.DIRECTORY),
    ResourceKind.DIRECTORY: (),
}


class ResourceError(ValueError):
    """The resources file is missing, malformed or fails validation."""


class UnknownRegionError(ValueError):
    """A region code with no resources (only IN, US, GB, or "" for generic)."""


@dataclass(frozen=True)
class Resource:
    id: str
    region: str  # an ISO code, or GENERIC
    kind: ResourceKind
    name: str
    contact: str
    hours: str
    checked: date
    source_url: str | None = None
    verification: Verification | None = None
    # A generic entry names no number ("your local emergency number").
    generic: bool = False
    notes: str = ""

    @property
    def line(self) -> str:
        """The text shown to the person: name, contact and (when known) hours."""
        text = f"{self.name}: {self.contact}"
        return text if self.hours in UNSTATED_HOURS else f"{text} ({self.hours})"


@dataclass(frozen=True)
class OutdatedResource:
    id: str
    name: str
    values: tuple[str, ...]
    reason: str
    source_url: str
    checked: date


def phone_numbers(text: str) -> tuple[str, ...]:
    """The digits of every phone-like number in `text` (URLs and emails are
    ignored): "call 116 123" -> ("116123",)."""
    text = _EMAIL.sub(" ", _URL.sub(" ", text))
    return tuple(re.sub(r"\D", "", m.group()) for m in _PHONE.finditer(text))


def urls(text: str) -> tuple[str, ...]:
    return tuple(m.group().rstrip(".,;:!?'\"") for m in _URL.finditer(text))


def emails(text: str) -> tuple[str, ...]:
    return tuple(m.group().lower().rstrip(".") for m in _EMAIL.finditer(text))


def base_domain(url: str) -> str:
    """The registrable part of a URL's host: telemanas.mohfw.gov.in ->
    mohfw.gov.in, www.crisistextline.org -> crisistextline.org."""
    host = urlsplit(url if "://" in url else f"https://{url}").hostname or ""
    labels = host.lower().split(".")
    second_level = {"gov", "co", "org", "ac", "net", "nic", "edu"}
    keep = 3 if len(labels) >= 3 and labels[-2] in second_level and len(labels[-1]) == 2 else 2
    return ".".join(labels[-keep:])


def normalize_region(region: str) -> str:
    """The canonical code for a configured region ("gb" -> "GB", "" -> GENERIC)."""
    code = region.strip().upper()
    if not code:
        return GENERIC
    if code not in SUPPORTED_REGIONS:
        raise UnknownRegionError(
            f"Unknown safety region {region!r}: use one of {', '.join(SUPPORTED_REGIONS)}, "
            "or leave it empty for the generic resources"
        )
    return code


@dataclass(frozen=True)
class ResourceDirectory:
    version: str
    verified_on: date
    review_due: date
    resources: tuple[Resource, ...]
    outdated: tuple[OutdatedResource, ...]

    def for_region(self, region: str) -> tuple[Resource, ...]:
        code = normalize_region(region)
        return tuple(r for r in self.resources if r.region == code)

    def select(
        self, region: str, kinds: Sequence[ResourceKind], *, limit: int = MAX_SHOWN
    ) -> tuple[Resource, ...]:
        """Resources of the requested kinds, in that order. A kind the region
        lacks is replaced by its first available substitute (an abuse line by
        a crisis line, a crisis line by a directory)."""
        available = self.for_region(region)
        chosen: dict[str, Resource] = {}
        for kind in kinds:
            for candidate in (kind, *_SUBSTITUTES[kind]):
                found = [r for r in available if r.kind is candidate]
                if found:
                    for resource in found:
                        chosen.setdefault(resource.id, resource)
                    break
        return tuple(chosen.values())[:limit]

    def render(self, region: str, kinds: Sequence[ResourceKind] | None = None) -> tuple[str, ...]:
        """Text lines for the reply guidance: every resource of the region, or
        those `select` picks for `kinds`."""
        chosen = self.for_region(region) if kinds is None else self.select(region, kinds)
        return tuple(r.line for r in chosen)

    def allowed_numbers(self, region: str) -> frozenset[str]:
        """Digits of every number the region's resources name."""
        return frozenset(n for r in self.for_region(region) for n in phone_numbers(r.contact))

    def allowed_domains(self, region: str) -> frozenset[str]:
        """Base domains of the region's contact links and official sources."""
        found: set[str] = set()
        for resource in self.for_region(region):
            found.update(base_domain(u) for u in urls(resource.contact))
            if resource.source_url:
                found.add(base_domain(resource.source_url))
        return frozenset(found)


# --- loading and validation -----------------------------------------------------

_RESOURCE_KEYS = {"id", "region", "kind", "name", "contact", "hours", "checked"}
_RESOURCE_OPTIONAL = {"source_url", "verification", "generic", "notes"}
_OUTDATED_KEYS = {"id", "name", "values", "reason", "source_url", "checked"}


def _text(entry: dict[str, Any], key: str, where: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ResourceError(f"{where}: {key!r} must be a non-empty string")
    return value.strip()


def _date(value: object, where: str) -> date:
    # tomllib gives datetime.date for a bare date; a datetime is not one here.
    if not isinstance(value, date) or type(value) is not date:
        raise ResourceError(f"{where}: expected a date (YYYY-MM-DD)")
    return value


def _https(url: str, where: str) -> str:
    if not url.startswith("https://") or not urlsplit(url).hostname:
        raise ResourceError(f"{where}: source_url must be an https URL")
    if any(word in url.lower() for word in _FORBIDDEN_SOURCES):
        raise ResourceError(f"{where}: Find A Helpline data may not be used (its terms forbid it)")
    return url


def _keys(entry: dict[str, Any], required: set[str], optional: set[str], where: str) -> None:
    missing = required - set(entry)
    unknown = set(entry) - required - optional
    if missing or unknown:
        raise ResourceError(f"{where}: missing {sorted(missing)}, unknown {sorted(unknown)}")


def _resource(entry: dict[str, Any], index: int, verified_on: date) -> Resource:
    where = f"resource #{index + 1}"
    _keys(entry, _RESOURCE_KEYS, _RESOURCE_OPTIONAL, where)
    resource_id = _text(entry, "id", where)
    where = f"resource {resource_id!r}"
    file_region = _text(entry, "region", where)
    region = GENERIC if file_region == _FILE_GENERIC else file_region
    if region != GENERIC and region not in SUPPORTED_REGIONS:
        raise ResourceError(f"{where}: unknown region {file_region!r}")
    try:
        kind = ResourceKind(_text(entry, "kind", where))
    except ValueError:
        raise ResourceError(f"{where}: unknown kind {entry['kind']!r}") from None
    generic = entry.get("generic", False)
    if not isinstance(generic, bool):
        raise ResourceError(f"{where}: 'generic' must be true or false")
    checked = _date(entry["checked"], where)
    if checked > verified_on:
        raise ResourceError(f"{where}: checked after the file's verified_on date")
    name, contact = _text(entry, "name", where), _text(entry, "contact", where)
    source_url = entry.get("source_url")
    verification = entry.get("verification")
    if generic:
        if region != GENERIC:
            raise ResourceError(f"{where}: generic entries belong to the generic region")
        # A generic entry must not invent a number, link or address.
        if phone_numbers(name + " " + contact) or urls(contact) or emails(contact):
            raise ResourceError(f"{where}: a generic entry must not name a number or a link")
    else:
        if not isinstance(source_url, str):
            raise ResourceError(f"{where}: needs the official source_url it was checked against")
        if verification not in _VERIFICATIONS:
            raise ResourceError(f"{where}: verification must be one of {sorted(_VERIFICATIONS)}")
    if source_url is not None:
        _https(_text(entry, "source_url", where), where)
    for link in urls(contact):
        _https(link, where)
    if region == GENERIC and phone_numbers(contact):
        raise ResourceError(f"{where}: generic resources must not name a number")
    return Resource(
        id=resource_id,
        region=region,
        kind=kind,
        name=name,
        contact=contact,
        hours=_text(entry, "hours", where),
        checked=checked,
        source_url=source_url,
        verification=cast(Verification | None, verification),
        generic=generic,
        notes=str(entry.get("notes", "")),
    )


def _outdated(entry: dict[str, Any], index: int, verified_on: date) -> OutdatedResource:
    where = f"outdated #{index + 1}"
    _keys(entry, _OUTDATED_KEYS, set(), where)
    values = entry["values"]
    if not isinstance(values, list) or not values or not all(isinstance(v, str) for v in values):
        raise ResourceError(f"{where}: 'values' must be a non-empty list of strings")
    checked = _date(entry["checked"], where)
    if checked > verified_on:
        raise ResourceError(f"{where}: checked after the file's verified_on date")
    return OutdatedResource(
        id=_text(entry, "id", where),
        name=_text(entry, "name", where),
        values=tuple(v.strip() for v in values),
        reason=_text(entry, "reason", where),
        source_url=_https(_text(entry, "source_url", where), where),
        checked=checked,
    )


def outdated_in(text: str, outdated: Iterable[OutdatedResource]) -> tuple[str, ...]:
    """Ids of the outdated resources `text` names (any formatting of a number)."""
    lowered = text.lower()
    found: list[str] = []
    for entry in outdated:
        for value in entry.values:
            digits = re.sub(r"\D", "", value)
            if "@" in value or not digits:
                hit = value.lower() in lowered
            else:
                separators = r"[\s().\-]{0,3}"
                pattern = r"(?<!\d)" + separators.join(digits) + r"(?!\d)"
                hit = re.search(pattern, text) is not None
            if hit:
                found.append(entry.id)
                break
    return tuple(found)


def _check_coverage(directory: ResourceDirectory) -> None:
    for region in (GENERIC, *SUPPORTED_REGIONS):
        kinds = {r.kind for r in directory.for_region(region)}
        label = region or _FILE_GENERIC
        if ResourceKind.EMERGENCY not in kinds:
            raise ResourceError(f"region {label}: no emergency resource")
        if not kinds & {ResourceKind.CRISIS_LINE, ResourceKind.DIRECTORY}:
            raise ResourceError(f"region {label}: no crisis line or directory")
    for resource in directory.resources:
        stale = outdated_in(resource.contact, directory.outdated)
        if stale:
            raise ResourceError(f"resource {resource.id!r} names an outdated contact {stale[0]!r}")


def load_resources(path: Path = RESOURCES_FILE) -> ResourceDirectory:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ResourceError(f"{path.name}: {exc}") from exc
    version = data.get("version")
    if not isinstance(version, str) or not _SEMVER.fullmatch(version):
        raise ResourceError(f"{path.name}: 'version' must be a version like 1.0.0")
    verified_on = _date(data.get("verified_on"), f"{path.name} verified_on")
    review_due = _date(data.get("review_due"), f"{path.name} review_due")
    if review_due <= verified_on:
        raise ResourceError(f"{path.name}: review_due must be after verified_on")
    changelog = data.get("changelog")
    if not isinstance(changelog, list) or not changelog:
        raise ResourceError(f"{path.name}: 'changelog' must not be empty")
    raw_resources = data.get("resource", [])
    raw_outdated = data.get("outdated", [])
    if not isinstance(raw_resources, list) or not isinstance(raw_outdated, list):
        raise ResourceError(f"{path.name}: [[resource]] and [[outdated]] must be tables")
    resources = tuple(_resource(e, i, verified_on) for i, e in enumerate(raw_resources))
    ids = [r.id for r in resources]
    if len(ids) != len(set(ids)):
        raise ResourceError(f"{path.name}: duplicate resource ids")
    directory = ResourceDirectory(
        version=version,
        verified_on=verified_on,
        review_due=review_due,
        resources=resources,
        outdated=tuple(_outdated(e, i, verified_on) for i, e in enumerate(raw_outdated)),
    )
    _check_coverage(directory)
    return directory


@lru_cache(maxsize=1)
def default_resources() -> ResourceDirectory:
    """The bundled resources file, loaded and validated once."""
    return load_resources()


def render_resources(region: str, kinds: Sequence[ResourceKind] | None = None) -> tuple[str, ...]:
    """Text lines for the reply guidance (see `ResourceDirectory.render`).
    An unknown region raises `UnknownRegionError`."""
    return default_resources().render(region, kinds)
