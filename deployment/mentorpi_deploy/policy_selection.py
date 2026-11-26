"""Resolve a complete local bundle by name."""


def resolve_policy(root, name):
    models = (root / "models").resolve()
    directory = (models / name).resolve()
    if directory.parent != models:
        raise ValueError(f"Expected a policy directory directly under {models}")
    for filename in ("policy.ts", "contract.json", "parity_cases.npz"):
        if not (directory / filename).is_file():
            raise ValueError(f"Missing policy file: {directory / filename}")
    return directory
