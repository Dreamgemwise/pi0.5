# openpi

Local workspace for OpenPI experiments and tooling.

## Environment

Create the default conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate openpi-dev
```

For RLDS training, use:

```bash
conda env create -f environment-rlds.yml
conda activate openpi-rlds
```

## Notes

- The project installs both the root package and `packages/openpi-client` in editable mode.
- Example scripts and local model paths use repository-relative paths such as `./models/...`.
