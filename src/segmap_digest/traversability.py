"""Traversability -- the seam an external going estimator plugs into.

Nothing in this file is a traversability model. It is the *socket*: one
contract, one registry, and one deliberately unclever placeholder that runs
until the real estimator is wired in. The real one is a black box owned
elsewhere; it wants a DEM and a DSM, and this repo currently has neither
(`LabelRaster.dem` exists and is usually None, `LabelRaster.dsm` does not exist
at all), which is why every `mean_slope` and `aspect_circvar` in the region
index is 0.0 today.

===========================================================================
IF YOU ARE HERE TO PLUG IN THE REAL ESTIMATOR, THIS IS THE WHOLE CONTRACT
===========================================================================

    class MyGoingModel:
        def estimate(self, request: TraversabilityRequest) -> TraversabilityResult:
            ...

    traversability.register_provider("mygoing", MyGoingModel())

and then either `traversability.estimate(raster, provider="mygoing")` or
`export RT_TRAVERSABILITY_PROVIDER=mygoing` and every existing caller picks it
up. No caller in this repo needs editing for that to happen -- that is the
entire point of the registry.

WHAT YOU ARE HANDED (every unit stated, because guessing one of these wrong is
silent and the output still looks like a map):

  request.labels     (H, W) uint8. DENSE TAXONOMY IDS, 0..N_CLASSES-1, indexing
                     `taxonomy.BY_ID`. These are NOT the sparse wire ids
                     0..241 that the Smart Terrain product emits -- `loader`
                     already translated them BY NAME and refused to guess on
                     any name the taxonomy does not define. If you have your own
                     class table keyed by wire id, map through `taxonomy.BY_ID[i].name`,
                     never through the integer.
  request.valid      (H, W) bool, or None meaning "every pixel is data". Nodata
                     on a real cropped export carries the integer 0, which is
                     ALSO the id of `Unclassified` (traffic 0.5). Scoring those
                     pixels makes a third of an arid AOI read as mediocre going.
                     Score them NaN (unmeasured), not 0.0 (measured and
                     impassable) -- see WHAT YOU MUST RETURN below; that
                     distinction is the entire reason `valid` exists.
                     Use `request.aoi` if you want valid AND mask pre-combined.
  request.dem        (H, W) float32 metres above the datum, already on the label
                     grid, or None. It may have been nearest-neighbour resampled
                     up from a much coarser source (`loader.load_dem`), which
                     quantises slope; say so in a note if that matters to you.
  request.dsm        (H, W) float32 metres, same grid -- top of canopy/roof, or
                     None. NOTHING IN THIS REPO PRODUCES ONE YET. The field
                     exists so the day it arrives the seam does not move.
  request.gsd        float, METRES PER PIXEL. Per-AOI: 0.3 for the synthetic
                     fixture, whatever the header said for a real export, and it
                     differs between AOIs in the same session. Do not hardcode a
                     resolution and do not infer one from array size.
  request.transform  rasterio Affine (or `loader.SimpleAffine`), or None.
  request.crs        rasterio CRS, or None. IF THE CRS IS GEOGRAPHIC THE
                     TRANSFORM IS IN DEGREES -- a 0.5 m/px export in EPSG:4326
                     has `transform.a` of ~5e-06. `request.gsd` is already
                     converted to metres (`loader.geotiff_gsd`); never derive a
                     distance from the affine coefficients without checking
                     `request.is_geographic` first.
  request.vehicle    str name, and `request.vehicle_params` the numbers behind
                     it: max_slope_deg, min_surface (a [0,1] taxonomy `traffic`
                     floor), wet_sensitivity, width_m (0.0 = no width gate).
                     THESE ARE GUESSES -- see `s4_products.VEHICLES`. If your
                     model owns better numbers, use your own and say so in a note.
  request.mask       optional (H, W) bool area of interest. This is the CALLER
                     restricting scope on purpose, not a data-availability
                     claim -- unlike `valid`, ground outside `mask` was still
                     measured, the caller just is not asking about it here.
                     Score what you like outside it, but you must still return
                     a full-shape raster; 0.0 is a reasonable default for
                     "not in scope" and is what the builtin does.
  request.wet        bool season flag, `request.osm` the burned OSM overlay when
                     one was joined (see `osm.burned`), `request.options` a
                     free-form dict this repo never inspects.

The arrays are shared views into the caller's raster. Read them; do not write
into them.

WHAT YOU MUST RETURN -- `TraversabilityResult`:

  score        (H, W) float32, exactly `request.labels.shape`. Every value is
               either in [0, 1] or NaN. 1.0 = freely traversable by that
               vehicle, 0.0 = impassable -- MEASURED and impassable, a
               confident answer. NaN = you never measured this pixel at all;
               it is unmeasured ground, not a confident zero, and is the
               correct thing to return for nodata rather than guessing 0.0.
               `_validate` rejects +/-inf and anything outside [0, 1] at the
               seam rather than letting a bad value propagate into a corridor
               search; NaN is the one non-finite value it lets through, on
               purpose. See `TraversabilityResult.unmeasured_fraction` for how
               a reader finds out how much of your raster this was.
  provider     your name, printed in reports so a reader knows which model
               produced the number.
  has_terrain  True only if elevation actually informed the score.
  notes        free text, printed verbatim in reports. Prefix with
               `ABSTAINED --` / `SCOPE --` to match the audit's vocabulary.

WHAT THIS REPO PROMISES YOU:

  * `has_terrain=False` IS A LEGITIMATE ANSWER, NOT A FAILURE. Downstream code
    already knows how to abstain on unmeasured terrain rather than report a
    confident zero -- `RegionIndex.has_terrain`, the S1 coverage caveat
    ("ABSTAINED -- no DEM: every mean_slope ... is 0.0, which is a valid-looking
    number meaning 'unmeasured'"), and S5 refusing a slope filter outright. A
    slope-free number dressed up as a real going estimate is precisely the
    failure mode this repo guards against everywhere else, so returning
    "I could not measure this" costs you nothing here.
  * Your `estimate` is called once per AOI with the full raster. No tiling, no
    resampling, no reordering behind your back.
  * Your notes reach the report unedited.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from .loader import LabelRaster
from .solutions import s4_products

# --- knobs, all of them plumbing rather than model ------------------------
#
# Where these should come from: nowhere else. Unlike the constants in
# `s4_products` these are not guesses about terrain -- they are names. The
# guesses live behind whichever provider is registered, which is the point of
# the split: a wrong number in here is a typo, a wrong number in there is a
# wrong map.

DEFAULT_PROVIDER = "builtin"

# Env override, so a deployment can swap the estimator without touching code or
# callers. Deliberately loud: an unknown name here raises rather than falling
# back to the builtin, because silently scoring an AOI with the placeholder when
# someone believed the real model was running is a wrong answer with a
# plausible number attached.
PROVIDER_ENV = "RT_TRAVERSABILITY_PROVIDER"

# Tolerance when checking a provider's raster is in [0, 1]. Wide enough for
# float32 rounding at the endpoints, narrow enough that a 0..255 raster or a
# stray -1 nodata sentinel is caught at the seam.
SCORE_EPS = 1e-6


# --- the request ----------------------------------------------------------

@dataclass(frozen=True)
class TraversabilityRequest:
    """Everything an external estimator could need, and nothing it cannot get.

    Frozen because it is handed to third-party code: whatever the provider does,
    the caller's view of what it asked for is unchanged.

    `dem` and `dsm` are separate fields rather than read off the raster so a
    caller can supply elevation the `LabelRaster` never carried -- and so this
    seam does not have to wait for `LabelRaster` to grow a `.dsm`.
    """

    raster: LabelRaster
    vehicle: str
    vehicle_params: Mapping[str, float]
    gsd: float                                  # metres per pixel, per-AOI
    dem: np.ndarray | None = None               # metres above datum, or None
    dsm: np.ndarray | None = None               # metres, top of canopy/roof
    mask: np.ndarray | None = None              # optional AOI restriction
    wet: bool = False                           # season flag
    osm: Any = None                             # burned OSM overlay, if joined
    options: Mapping[str, Any] = field(default_factory=dict)

    # Convenience views. Properties, not fields, so there is exactly one copy of
    # each array and no way for the two to disagree.

    @property
    def labels(self) -> np.ndarray:
        """(H, W) uint8 dense taxonomy ids -- see the module docstring."""
        return self.raster.labels

    @property
    def shape(self) -> tuple[int, int]:
        return self.raster.shape

    @property
    def valid(self) -> np.ndarray | None:
        return self.raster.valid

    @property
    def transform(self):
        return self.raster.transform

    @property
    def crs(self):
        return self.raster.crs

    @property
    def is_geographic(self) -> bool:
        """True when `transform` is in degrees, not metres."""
        crs = self.raster.crs
        return bool(getattr(crs, "is_geographic", False))

    @property
    def has_dem(self) -> bool:
        return self.dem is not None

    @property
    def has_dsm(self) -> bool:
        return self.dsm is not None

    @property
    def aoi(self) -> np.ndarray | None:
        """`valid` AND `mask`, or None when neither restricts anything."""
        if self.valid is None:
            return self.mask
        if self.mask is None:
            return self.valid
        return self.valid & self.mask

    @property
    def pixel_area_m2(self) -> float:
        return self.gsd * self.gsd

    @classmethod
    def build(cls, raster: LabelRaster, vehicle: str = "wheeled",
              dem: np.ndarray | None = None, dsm: np.ndarray | None = None,
              gsd: float | None = None, **kw) -> "TraversabilityRequest":
        """The normal way to make one: fills gsd and the vehicle table from the raster.

        `dem`/`dsm` of None mean "whatever the raster carries" -- today that is
        `raster.dem` and nothing, and it will be `raster.dsm` for free the day
        the loader grows one. A caller who genuinely wants the no-elevation
        answer from a raster that HAS a DEM constructs the request directly.
        """
        if vehicle not in s4_products.VEHICLES:
            raise KeyError(f"unknown vehicle {vehicle!r}; "
                           f"known: {sorted(s4_products.VEHICLES)}")
        v = s4_products.VEHICLES[vehicle]
        params = {"max_slope_deg": v.max_slope_deg, "min_surface": v.min_surface,
                  "wet_sensitivity": v.wet_sensitivity, "width_m": v.width_m}
        dem = raster.dem if dem is None else dem
        dsm = getattr(raster, "dsm", None) if dsm is None else dsm
        gsd = float(raster.gsd if gsd is None else gsd)
        if gsd <= 0:
            raise ValueError(f"gsd must be positive metres per pixel, got {gsd}")
        for name, arr in (("dem", dem), ("dsm", dsm), ("mask", kw.get("mask"))):
            if arr is not None and tuple(np.shape(arr)) != tuple(raster.shape):
                raise ValueError(f"{name} shape {np.shape(arr)} does not match the "
                                 f"label raster {tuple(raster.shape)}; resample it "
                                 f"onto the label grid first (see loader.load_dem)")
        return cls(raster=raster, vehicle=vehicle, vehicle_params=params,
                   gsd=gsd, dem=dem, dsm=dsm, **kw)


# --- the result -----------------------------------------------------------

@dataclass(frozen=True)
class TraversabilityResult:
    """A going raster plus the provenance a reader needs to trust it."""

    score: np.ndarray                  # (H, W) float32, [0, 1] or NaN (unmeasured)
    provider: str                      # who produced it
    has_terrain: bool                  # did elevation actually inform it
    notes: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return self.score.shape

    @property
    def unmeasured_fraction(self) -> float:
        """Fraction of `score` that is NaN -- genuinely unmeasured, not scored
        impassable. 0.0 for an empty raster or one with no NaN at all.

        A downstream reader (tilemap rendering, report text) needs this
        constantly and should never have to reach into `np.isnan(result.score)`
        itself -- that is exactly the kind of thing that gets forgotten once,
        quietly turning "we never measured this" back into "this is 0.0",
        which is the bug this property exists to make impossible to miss.
        """
        score = self.score
        if score.size == 0:
            return 0.0
        return float(np.isnan(score).mean())

    def caveat_block(self) -> str:
        """Report-ready header. Every note, verbatim, plus the terrain verdict.

        A provider's notes are the only place it can say what its number does
        NOT mean, so they are printed with the map rather than logged.
        """
        lines = [f"# traversability provider: {self.provider} "
                 f"(has_terrain={self.has_terrain})"]
        if not self.has_terrain:
            lines.append("#   elevation did not inform this raster -- treat it as "
                         "a surface-class map, not a going estimate")
        frac = self.unmeasured_fraction
        if frac > 0:
            lines.append(f"#   {frac:.1%} of this raster is unmeasured (NaN) -- "
                         f"treat that ground as unknown, not impassable")
        lines += [f"#   {n}" for n in self.notes]
        return "\n".join(lines)


# --- the contract ---------------------------------------------------------

@runtime_checkable
class TraversabilityProvider(Protocol):
    """One method. Implement it, register it, done -- see the module docstring.

    A Protocol rather than a hard ABC so the owner's estimator does not have to
    import this repo to satisfy it: any object with a matching `estimate` is a
    provider. Subclassing works too, and gives you an isinstance check for free.
    """

    def estimate(self, request: TraversabilityRequest) -> TraversabilityResult:
        """Score `request` for `request.vehicle`. Must not mutate the request."""
        ...


# --- the registry ---------------------------------------------------------

_PROVIDERS: dict[str, TraversabilityProvider] = {}


def register_provider(name: str, provider: TraversabilityProvider) -> None:
    """Make `provider` available as `name`. Re-registering a name replaces it."""
    if not isinstance(name, str) or not name:
        raise ValueError("provider name must be a non-empty string")
    if not isinstance(provider, TraversabilityProvider):
        raise TypeError(
            f"{provider!r} is not a TraversabilityProvider: it needs a callable "
            f"`estimate(request) -> TraversabilityResult`"
        )
    _PROVIDERS[name] = provider


def available_providers() -> list[str]:
    return sorted(_PROVIDERS)


def get_provider(name: str | None = None) -> TraversabilityProvider:
    """Resolve a provider: explicit name, else $RT_TRAVERSABILITY_PROVIDER, else builtin.

    Unknown names raise. There is no fallback on purpose -- see PROVIDER_ENV.
    """
    source = "argument"
    if name is None:
        name = os.environ.get(PROVIDER_ENV) or None
        source = f"${PROVIDER_ENV}"
    if name is None:
        name, source = DEFAULT_PROVIDER, "default"
    try:
        return _PROVIDERS[name]
    except KeyError:
        raise KeyError(
            f"no traversability provider named {name!r} (from {source}); "
            f"registered: {available_providers()}. Register it with "
            f"traversability.register_provider({name!r}, <your estimator>) before use."
        ) from None


# --- the builtin placeholder ----------------------------------------------
#
# Honest rather than clever, and it is meant to be replaced. With no DEM it is
# exactly `s4_products.trafficability` -- the class-based surface score, with the
# slope term inert at 1.0 -- and it says so. It does not interpolate, guess a
# regional slope, or blend in a plausible-looking constant, because a number
# that reads as a going estimate but never saw the ground is the failure this
# repo abstains from everywhere else (RegionIndex.has_terrain, the S1 ABSTAINED
# caveat, S5 refusing a slope filter with no DEM).

BUILTIN_NAME = "builtin"

_NO_DEM_NOTE = (
    "ABSTAINED -- no DEM: slope was unmeasured, so this is the class-based "
    "surface trafficability from s4_products only, with the slope term inert at "
    "1.0. A 30-degree scree slope of the same class scores identically to flat "
    "ground. Do not read it as a going estimate; attach a DEM, or register the "
    "real estimator ($" + PROVIDER_ENV + ")."
)

_DSM_NOTE = (
    "SCOPE -- a DSM was supplied and the builtin does not use it. Vegetation "
    "height, wall and embankment height, and anything else in DSM-minus-DEM are "
    "the external provider's job; this placeholder scores surface class and bare "
    "slope only."
)


class BuiltinProvider:
    """The placeholder. Class surface score, plus DEM slope when there is a DEM."""

    name = BUILTIN_NAME

    def estimate(self, request: TraversabilityRequest) -> TraversabilityResult:
        notes: list[str] = []

        # s4 reads elevation off the raster, so hand it a raster carrying the
        # DEM this request actually specifies. `replace` is a field copy: the
        # arrays are shared, not duplicated.
        raster = request.raster
        if request.dem is not raster.dem:
            raster = replace(raster, dem=request.dem)

        score = s4_products.trafficability(
            raster, vehicle=request.vehicle, wet=request.wet, osm=request.osm,
        )

        has_terrain = request.has_dem
        if has_terrain:
            v = request.vehicle_params
            notes.append(
                f"slope came from the DEM: full score below "
                f"{s4_products.SLOPE_FREE_DEG:g} deg, falling linearly to 0 at the "
                f"{request.vehicle} limit of {v['max_slope_deg']:g} deg. Bare-earth "
                f"gradient only -- no roughness, step height, or curvature."
            )
        else:
            notes.append(_NO_DEM_NOTE)
        if request.has_dsm:
            notes.append(_DSM_NOTE)

        if request.is_geographic:
            notes.append(
                f"SCOPE -- this raster is in a geographic CRS, so its transform is "
                f"in degrees; every distance here used gsd = {request.gsd:.3f} m/px "
                f"as converted by loader.geotiff_gsd, not the affine."
            )
        if request.raster.subset_note:
            notes.append(request.raster.subset_note)

        # Nodata carries class id 0 (Unclassified, traffic 0.5) -- the same
        # reasoning as s4_products.compute: left alone, a sparse export reads
        # as moderately driveable open ground and a corridor search drives
        # across it. The fix is NOT to score it 0.0 either -- that is
        # identical to a genuine obstacle (cliff, water) and a reader of the
        # raster cannot tell "no data here" from "impassable here" apart.
        # NaN is the distinct signal: `valid` means the sensor never covered
        # this pixel at all, which is a data-availability fact, not a going
        # estimate, so it becomes NaN (see `unmeasured_fraction`).
        #
        # `mask`, by contrast, is the caller narrowing scope on purpose --
        # ground that WAS measured (it is still inside `valid`) but that they
        # are not asking about right now. That is a real "not in scope"
        # answer rather than an "unknown" one, so it keeps the old
        # all-zero convention below; only `valid` graduates to NaN.
        if request.mask is not None:
            notes.append(f"SCOPE -- scored inside the supplied mask only "
                         f"({float(request.mask.mean()):.1%} of the extent); "
                         f"everything outside it is 0.")
        valid = request.valid
        if valid is not None:
            score = np.where(valid, score, np.nan)
        if request.mask is not None:
            outside_mask = ~request.mask
            if valid is not None:
                # Nodata already went to NaN above; do not stomp that back to
                # 0.0 just because it also happens to sit outside the mask.
                outside_mask = outside_mask & valid
            score = np.where(outside_mask, 0.0, score)

        return TraversabilityResult(
            score=np.clip(score, 0.0, 1.0).astype(np.float32),
            provider=self.name, has_terrain=has_terrain, notes=notes,
        )


register_provider(BUILTIN_NAME, BuiltinProvider())


# --- convenience ----------------------------------------------------------

def estimate(raster: LabelRaster, vehicle: str = "wheeled",
             dem: np.ndarray | None = None, dsm: np.ndarray | None = None,
             provider: str | None = None, **kw) -> TraversabilityResult:
    """Score a raster without building the request by hand.

        res = traversability.estimate(tile, vehicle="tracked")
        print(res.caveat_block())

    `dem`/`dsm` default to whatever the raster carries; `provider` defaults to
    $RT_TRAVERSABILITY_PROVIDER, then to the builtin placeholder. Remaining
    keywords (`mask`, `wet`, `osm`, `options`, `gsd`) go to the request.
    """
    request = TraversabilityRequest.build(raster, vehicle=vehicle, dem=dem,
                                          dsm=dsm, **kw)
    impl = get_provider(provider)
    return _validate(impl.estimate(request), request)


def _validate(result: TraversabilityResult,
              request: TraversabilityRequest) -> TraversabilityResult:
    """Enforce the output half of the contract, at the seam.

    A third-party estimator that returns a 0..255 raster, +/-inf, or a
    transposed array is a bug that would otherwise surface a thousand lines
    later as a corridor through a cliff. NaN is let through on purpose: it is
    the contract's way of saying "genuinely unmeasured", not a bug -- see the
    module docstring's WHAT YOU MUST RETURN section and
    `TraversabilityResult.unmeasured_fraction`. Cheap to catch the rest here,
    and the error names the contract it broke. float64 is accepted and cast --
    that one is a formality, not a mistake.
    """
    if not isinstance(result, TraversabilityResult):
        raise TypeError(f"provider returned {type(result).__name__}, expected a "
                        f"TraversabilityResult")
    score = np.asarray(result.score)
    if score.shape != request.shape:
        raise ValueError(f"provider {result.provider!r} returned shape {score.shape}, "
                         f"expected {request.shape} (the full label grid, even when "
                         f"a mask was supplied)")
    if np.isinf(score).any():
        raise ValueError(f"provider {result.provider!r} returned +/-inf; the "
                         f"contract is [0, 1], or NaN for a pixel that was never "
                         f"measured -- inf is neither")
    finite = ~np.isnan(score)
    if finite.any():
        lo, hi = float(score[finite].min()), float(score[finite].max())
        if lo < -SCORE_EPS or hi > 1.0 + SCORE_EPS:
            raise ValueError(f"provider {result.provider!r} returned values in "
                             f"[{lo:.4g}, {hi:.4g}]; the contract is [0, 1] where 1 is "
                             f"freely traversable and 0 impassable (NaN is allowed, "
                             f"and exempt from this range check)")
    if score.dtype == np.float32:
        return result
    return replace(result, score=score.astype(np.float32))
