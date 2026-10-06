"""Resolve and validate audit run settings from CLI flags, YAML and defaults.

Precedence for ``n_splits`` / ``n_repeats`` / ``n_perm``:

1. explicit CLI value (or an explicit ``run_full_audit`` argument);
2. smoke-mode defaults (3 splits × 1 repeat; 5 permutations) — config values
   are not applied and this is recorded;
3. YAML config;
4. full-mode defaults (5 splits × 10 repeats; 100 permutations).

Smoke does **not** override an explicit CLI ``--n-perm`` / ``--n-splits`` /
``--n-repeats``. Other settings: CLI > YAML > default.

Unknown keys and values the audit cannot honour raise ``ConfigError``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCOPES = ("primary", "sensitivity")
POLICIES = ("raw", "quarantine_out_of_range", "grouped_duplicates")

FULL_EVAL_DEFAULTS: dict[str, int] = {"n_splits": 5, "n_repeats": 10}
SMOKE_EVAL_DEFAULTS: dict[str, int] = {"n_splits": 3, "n_repeats": 1}
EVAL_MINIMUMS: dict[str, int] = {"n_splits": 2, "n_repeats": 1}

FULL_N_PERM_DEFAULT = 100
SMOKE_N_PERM_DEFAULT = 5

# sklearn RandomState / numpy SeedSequence accept [0, 2**32).
MIN_SEED = 0
MAX_SEED = 2**32 - 1

TOP_LEVEL_KEYS = frozenset(
    {
        "scope",
        "tail_quarantine_from",
        "near_dup_row_limit",
        "n_perm",
        "seed",
        "smoke",
        "sensitivity_policy",
        "persistent_error_threshold",
        "eval",
    }
)
EVAL_KEYS = frozenset({"n_splits", "n_repeats", "seed"})


class ConfigError(ValueError):
    """Invalid or unsupported audit configuration."""


def validate_bool(name: str, value: Any) -> bool:
    if type(value) is not bool:
        raise ConfigError(f"{name} must be a bool, got {value!r}")
    return value


def validate_enum(name: str, value: Any, allowed: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ConfigError(f"{name} must be one of {list(allowed)}, got {value!r}")
    return value


def validate_int(
    name: str,
    value: Any,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name} must be <= {maximum}, got {value}")
    return value


def validate_positive_int(name: str, value: Any, minimum: int) -> int:
    return validate_int(name, value, minimum=minimum)


def validate_seed(name: str, value: Any) -> int:
    return validate_int(name, value, minimum=MIN_SEED, maximum=MAX_SEED)


def validate_file_config(
    file_cfg: dict[str, Any], *, persistent_error_threshold: float
) -> dict[str, Any]:
    """Reject unknown keys and values the audit would otherwise silently ignore.

    Returns the ``eval`` sub-mapping (empty if absent).
    """
    if not isinstance(file_cfg, dict):
        raise ConfigError(f"config root must be a mapping, got {type(file_cfg).__name__}")
    unknown = sorted(set(file_cfg) - TOP_LEVEL_KEYS)
    if unknown:
        raise ConfigError(f"unknown config keys: {unknown}")

    eval_cfg = file_cfg.get("eval", {})
    if eval_cfg is None:
        eval_cfg = {}
    if not isinstance(eval_cfg, dict):
        raise ConfigError("config 'eval' must be a mapping")
    unknown_eval = sorted(set(eval_cfg) - EVAL_KEYS)
    if unknown_eval:
        raise ConfigError(f"unknown config keys under 'eval': {unknown_eval}")

    for key in ("n_splits", "n_repeats"):
        if key in eval_cfg:
            validate_positive_int(f"eval.{key}", eval_cfg[key], EVAL_MINIMUMS[key])

    if "seed" in eval_cfg:
        validate_seed("eval.seed", eval_cfg["seed"])
        if "seed" in file_cfg and file_cfg["seed"] != eval_cfg["seed"]:
            raise ConfigError(
                f"config seed ({file_cfg['seed']!r}) and eval.seed ({eval_cfg['seed']!r}) "
                "disagree; the audit uses a single seed"
            )

    if "scope" in file_cfg:
        validate_enum("scope", file_cfg["scope"], SCOPES)
    if "sensitivity_policy" in file_cfg:
        validate_enum("sensitivity_policy", file_cfg["sensitivity_policy"], POLICIES)
    if "smoke" in file_cfg:
        validate_bool("smoke", file_cfg["smoke"])
    if "seed" in file_cfg:
        validate_seed("seed", file_cfg["seed"])
    if "n_perm" in file_cfg:
        validate_int("n_perm", file_cfg["n_perm"], minimum=1)
    if "tail_quarantine_from" in file_cfg:
        validate_int("tail_quarantine_from", file_cfg["tail_quarantine_from"], minimum=1)
    if "near_dup_row_limit" in file_cfg:
        validate_int("near_dup_row_limit", file_cfg["near_dup_row_limit"], minimum=1)

    if "persistent_error_threshold" in file_cfg:
        value = file_cfg["persistent_error_threshold"]
        if value != persistent_error_threshold:
            raise ConfigError(
                f"persistent_error_threshold={value!r} is not configurable; "
                f"the audit uses {persistent_error_threshold}"
            )
    return eval_cfg


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML mapping. PyYAML is required; no silent fallback parser."""
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError(
            "PyYAML is required to load --config. Install the project with the "
            "default dependencies (PyYAML is a core requirement)."
        ) from exc
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        raise ConfigError(f"config file is empty: {path}")
    if not isinstance(data, dict):
        raise ConfigError(
            f"config root must be a mapping, got {type(data).__name__}: {path}"
        )
    return data


def _pick(
    cli_val: Any,
    file_cfg: dict[str, Any],
    key: str,
    default: Any,
    *,
    extra_maps: tuple[dict[str, Any], ...] = (),
) -> tuple[Any, str]:
    if cli_val is not None:
        return cli_val, "cli"
    if key in file_cfg:
        return file_cfg[key], f"config:{key}"
    for extra in extra_maps:
        if key in extra:
            return extra[key], f"config:{key}"
    return default, "default"


def resolve_eval_settings(
    *,
    smoke: bool,
    cli_n_splits: int | None,
    cli_n_repeats: int | None,
    file_eval: dict[str, Any],
) -> dict[str, Any]:
    """Return effective ``n_splits`` / ``n_repeats`` with the source of each value."""
    validate_bool("smoke", smoke)
    cli = {"n_splits": cli_n_splits, "n_repeats": cli_n_repeats}
    resolved: dict[str, Any] = {"sources": {}, "notes": []}
    for key in ("n_splits", "n_repeats"):
        if cli[key] is not None:
            value = validate_positive_int(
                f"--{key.replace('_', '-')}", cli[key], EVAL_MINIMUMS[key]
            )
            source = "cli"
        elif smoke:
            value = SMOKE_EVAL_DEFAULTS[key]
            source = "smoke_default"
            if key in file_eval:
                resolved["notes"].append(
                    f"config eval.{key}={file_eval[key]} not applied: smoke mode uses "
                    f"{value} (pass --{key.replace('_', '-')} to override)"
                )
        elif key in file_eval:
            value = validate_positive_int(f"eval.{key}", file_eval[key], EVAL_MINIMUMS[key])
            source = f"config:eval.{key}"
        else:
            value = FULL_EVAL_DEFAULTS[key]
            source = "full_default"
        resolved[key] = value
        resolved["sources"][key] = source
    return resolved


def resolve_n_perm(
    *,
    smoke: bool,
    cli_n_perm: int | None,
    file_cfg: dict[str, Any],
) -> dict[str, Any]:
    requested: int | None
    requested_source: str
    if cli_n_perm is not None:
        requested = validate_int("n_perm", cli_n_perm, minimum=1)
        requested_source = "cli"
    elif "n_perm" in file_cfg:
        requested = validate_int("n_perm", file_cfg["n_perm"], minimum=1)
        requested_source = "config:n_perm"
    else:
        requested = FULL_N_PERM_DEFAULT
        requested_source = "full_default"

    note = None
    if cli_n_perm is not None:
        effective = requested
        source = "cli"
    elif smoke:
        effective = SMOKE_N_PERM_DEFAULT
        source = "smoke_default"
        if requested != effective:
            note = (
                f"n_perm requested={requested} ({requested_source}) not applied: "
                f"smoke mode uses {effective} (pass --n-perm to override)"
            )
    else:
        effective = requested
        source = requested_source
    return {
        "n_perm": effective,
        "n_perm_requested": requested,
        "source": source,
        "requested_source": requested_source,
        "note": note,
    }


@dataclass
class RunSettings:
    scope: str
    sensitivity_policy: str
    smoke: bool
    run_generalization: bool
    seed: int
    n_splits: int
    n_repeats: int
    n_perm: int
    n_perm_requested: int
    tail_quarantine_from: int
    near_dup_row_limit: int
    sources: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_config_block(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "eval": {
                "n_splits": self.n_splits,
                "n_repeats": self.n_repeats,
                "seed": self.seed,
                "scoring": "f1_macro",
            },
            "n_perm": self.n_perm,
            "n_perm_requested": self.n_perm_requested,
            "seed": self.seed,
            "tail_quarantine_from": self.tail_quarantine_from,
            "near_dup_row_limit": self.near_dup_row_limit,
            "sensitivity_policy": self.sensitivity_policy,
            "run_generalization": self.run_generalization,
            "smoke": self.smoke,
        }


def resolve_run_settings(
    *,
    file_cfg: dict[str, Any] | None = None,
    persistent_error_threshold: float,
    cli_scope: str | None = None,
    cli_sensitivity_policy: str | None = None,
    cli_smoke: bool | None = None,
    cli_seed: int | None = None,
    cli_n_perm: int | None = None,
    cli_n_splits: int | None = None,
    cli_n_repeats: int | None = None,
    cli_tail_quarantine_from: int | None = None,
    cli_near_dup_row_limit: int | None = None,
    cli_run_generalization: bool | None = None,
) -> RunSettings:
    file_cfg = dict(file_cfg or {})
    eval_cfg = validate_file_config(
        file_cfg, persistent_error_threshold=persistent_error_threshold
    )

    smoke_raw, smoke_source = _pick(cli_smoke, file_cfg, "smoke", False)
    smoke = validate_bool("smoke", smoke_raw)

    scope_raw, scope_source = _pick(cli_scope, file_cfg, "scope", "primary")
    scope = validate_enum("scope", scope_raw, SCOPES)

    policy_raw, policy_source = _pick(
        cli_sensitivity_policy, file_cfg, "sensitivity_policy", "raw"
    )
    policy = validate_enum("sensitivity_policy", policy_raw, POLICIES)

    seed_raw, seed_source = _pick(cli_seed, file_cfg, "seed", 0, extra_maps=(eval_cfg,))
    seed = validate_seed("seed", seed_raw)

    tail_raw, tail_source = _pick(
        cli_tail_quarantine_from, file_cfg, "tail_quarantine_from", 1100
    )
    tail = validate_int("tail_quarantine_from", tail_raw, minimum=1)

    near_raw, near_source = _pick(
        cli_near_dup_row_limit, file_cfg, "near_dup_row_limit", 1100
    )
    near = validate_int("near_dup_row_limit", near_raw, minimum=1)

    gen_raw = True if cli_run_generalization is None else cli_run_generalization
    gen = validate_bool("run_generalization", gen_raw)

    eval_settings = resolve_eval_settings(
        smoke=smoke,
        cli_n_splits=cli_n_splits,
        cli_n_repeats=cli_n_repeats,
        file_eval=eval_cfg,
    )
    perm = resolve_n_perm(smoke=smoke, cli_n_perm=cli_n_perm, file_cfg=file_cfg)

    max_grouped_seed = seed + max(0, eval_settings["n_repeats"] - 1)
    if max_grouped_seed > MAX_SEED:
        raise ConfigError(
            f"seed={seed} with n_repeats={eval_settings['n_repeats']} would use "
            f"random_state seed+repeat up to {max_grouped_seed}, which exceeds "
            f"{MAX_SEED} (sklearn/numpy seed range)"
        )

    notes = list(eval_settings["notes"])
    if perm["note"]:
        notes.append(perm["note"])

    sources = {
        "scope": scope_source,
        "sensitivity_policy": policy_source,
        "smoke": smoke_source,
        "seed": seed_source,
        "tail_quarantine_from": tail_source,
        "near_dup_row_limit": near_source,
        "run_generalization": "cli" if cli_run_generalization is not None else "default",
        "n_splits": eval_settings["sources"]["n_splits"],
        "n_repeats": eval_settings["sources"]["n_repeats"],
        "n_perm": perm["source"],
        "n_perm_requested": perm["requested_source"],
    }
    return RunSettings(
        scope=scope,
        sensitivity_policy=policy,
        smoke=smoke,
        run_generalization=gen,
        seed=seed,
        n_splits=eval_settings["n_splits"],
        n_repeats=eval_settings["n_repeats"],
        n_perm=perm["n_perm"],
        n_perm_requested=perm["n_perm_requested"],
        tail_quarantine_from=tail,
        near_dup_row_limit=near,
        sources=sources,
        notes=notes,
    )


def settings_to_dict(settings: RunSettings) -> dict[str, Any]:
    return asdict(settings)
