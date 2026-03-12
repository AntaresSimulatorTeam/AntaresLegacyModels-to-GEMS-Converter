# AntaresLegacyModels-to-GEMS-Converter

[![License: MPL v2](https://img.shields.io/badge/License-MPLv2-blue.svg)](https://www.mozilla.org/en-US/MPL/2.0/)
[![Python](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)

A Python tool that converts [Antares Simulator](https://antares-simulator.readthedocs.io/en/latest/) legacy studies into the [GEMS](https://gems-energy.readthedocs.io/en/latest/) format.

## Background

[Antares Simulator](https://antares-simulator.readthedocs.io/en/latest/) is an open-source power system simulator developed by RTE to assess the adequacy and economic performance of interconnected energy systems. Historically, the mathematical models of its components — thermal clusters, hydro units, short-term storage, and so on — were defined implicitly inside the solver's code.

[GEMS](https://gems-energy.readthedocs.io/en/latest/) (Generic Energy Modelling Scheme) is a high-level modelling language that brings these model definitions out of the codebase and into plain YAML files, where variables, parameters, and constraints are expressed using natural mathematical syntax. A GEMS model library can be used directly with [Antares Simulator's dynamic modeler](https://antares-simulator.readthedocs.io/en/latest/user-guide/modeler/01-overview-modeler/) or with the [GemsPy](https://github.com/AntaresSimulatorTeam/GemsPy) Python interpreter.

## Purpose

This tool automates the translation of Antares Simulator studies into valid GEMS system files (YAML), making it straightforward to migrate existing studies to the new modelling framework — with no manual rewriting required.

## Installation

```bash
git clone https://github.com/AntaresSimulatorTeam/AntaresLegacyModels-to-GEMS-Converter.git
cd AntaresLegacyModels-to-GEMS-Converter
pip install -r requirements.txt
```


## Related projects

| Project | Description |
|---|---|
| [Antares Simulator](https://github.com/AntaresSimulatorTeam/Antares_Simulator) | Open-source power system simulator |
| [GEMS](https://gems-energy.readthedocs.io/en/latest/) | Graph-based algebraic modelling language |
| [GemsPy](https://github.com/AntaresSimulatorTeam/GemsPy) | Python interpreter for the GEMS modelling framework |
| [PyPSA-to-GEMS-Converter](https://github.com/AntaresSimulatorTeam/PyPSA-to-GEMS-Converter) | Converter from PyPSA models to GEMS format |

## License

[Mozilla Public License v2.0](LICENSE)
