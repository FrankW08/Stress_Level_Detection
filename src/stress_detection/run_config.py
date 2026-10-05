"""Resolve audit run settings from CLI flags, a YAML config mapping and mode defaults.

Precedence for ``n_splits`` / ``n_repeats``:

1. explicit CLI value (``--n-splits`` / ``--n-repeats``), in any mode;
2. smoke mode: smoke defaults (3 splits x 1 repeat); config ``eval.*`` values are
   not applied and this is recorded in ``notes``;
3. config ``eval.n_splits`` / ``eval.n_repeats``;
4. full-mode defaults (5 splits x 10 repeats).

Unknown config keys and values the audit cannot honour raise ``ConfigError``
instead of being ignored.
"""

from __future__ import annotations

from typing import Any

FULL_EVAL_DEFAULTS: dict[str, int] = {"n_splits": 5, "n_repeats": 10}
SMOKE_EVAL_DEFAULTS: dict[str, int] = {"n_splits": 3, "n_repeats": 1}
EVAL_MINIMUMS: dict[str, int] = {"n_splits": 2, "n_repeats": 1}

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


def validate_positive_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def validate_file_config(
    file_cfg: dict[str, Any], *, persistent_error_threshold: float
) -> dict[str, Any]:
    """Reject unknown keys and values the audit would otherwise silently ignore.

    Returns the ``eval`` sub-mapping (empty if absent).
    """
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
        validate_positive_int("eval.seed", eval_cfg["seed"], 0)
        if "seed" in file_cfg and file_cfg["seed"] != eval_cfg["seed"]:
            raise ConfigError(
                f"config seed ({file_cfg['seed']!r}) and eval.seed ({eval_cfg['seed']!r}) "
                "disagree; the audit uses a single seed"
            )

    if "persistent_error_threshold" in file_cfg:
        value = file_cfg["persistent_error_threshold"]
        if value != persistent_error_threshold:
            raise ConfigError(
                f"persistent_error_threshold={value!r} is not configurable; "
                f"the audit uses {persistent_error_threshold}"
            )
    return eval_cfg


def resolve_eval_settings(
    *,
    smoke: bool,
    cli_n_splits: int | None,
    cli_n_repeats: int | None,
    file_eval: dict[str, Any],
) -> dict[str, Any]:
    """Return effective ``n_splits`` / ``n_repeats`` with the source of each value."""
    cli = {"n_splits": cli_n_splits, "n_repeats": cli_n_repeats}
    resolved: dict[str, Any] = {"sources": {}, "notes": []}
    for key in ("n_splits", "n_repeats"):
        if cli[key] is not None:
            value = validate_positive_int(f"--{key.replace('_', '-')}", cli[key], EVAL_MINIMUMS[key])
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
