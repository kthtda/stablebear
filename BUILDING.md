## Bump version number

- In pyproject.toml, change [project].version

## Minimal module build

Use CMake 3.30.3 or newer. Build and install directly with CMake; no preliminary
pip installation of the project is required. See [AGENTS.md](AGENTS.md#build--development)
for the configure, build, and install commands.

## Building the documentation

Go to the `docs` directory and run `Make` on Linux/OSX, or `make.bat` on Windows.

This will create `_build` containing the documentation.
