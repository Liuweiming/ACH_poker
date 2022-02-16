## What is OpenSpiel?

OpenSpiel is a collection of environments and algorithms for research in general
reinforcement learning and search/planning in games. OpenSpiel also includes
tools to analyze learning dynamics and other common evaluation metrics. Games
are represented as procedural extensive-form games, with some natural
extensions.

**Open Spiel supports**

*   Single and multi-player games
*   Fully observable (via observations) and imperfect information games (via
    information states and observations)
*   Stochasticity (via explicit chance nodes mostly, even though implicit
    stochasticity is partially supported)
*   n-player normal-form "one-shot" games and (2-player) matrix games
*   Sequential and simultaneous move games
*   Zero-sum, general-sum, and cooperative (identical payoff) games

**Multi-language support**

*   C++17
*   Python 3
*   A subset of the features are available in Swift.

The games and utility functions (e.g. exploitability computation) are written in
C++. These are also available using
[pybind11](https://pybind11.readthedocs.io/en/stable/) Python (2.7 and 3)
bindings.

The methods names are in `CamelCase` in C++ and `snake_case` in Python (e.g.
`state.ApplyAction` in C++ will be `state.apply_action` in Python). See the
pybind11 definition in `open_spiel/python/pybind11/pyspel.cc` for the full
mapping between names.

For algorithms, many are written in both languages, even if some are only
available from Python.

**Platforms**

OpenSpiel has been tested on Linux (Debian 10 and Ubuntu 19.04), MacOS, and
[Windows 10 (through Windows Subsystem for Linux)](windows.md).
