<!--
Copyright (C) 2021-2022 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# PyRoboCOP

PyRoboCOP is a lightweight Python-based package  for  control  and  optimization  of  robotic  systems  described by nonlinear Differential Algebraic Equations (DAEs)

## Features

The main features of the package are:

- Direct transcript from optimal control problem to NLP
- Contact modeling by complementarity constraints
- Obstacle avoidance by complementarity constraints
- Automatic Differentiation for sparse derivatives
- Open-source and based on only open source toolchain

- Support for Trajectory Optimization problems
- Support for minimum time problems
- Support for optimization over fixed mode sequence problems with unknown sequence time horizons
- Support for parameter estimation in linear complementarity systems.


## Installation

1. Download or git clone PyRoboCOP.

2. Install PyRoboCOP in a virtual environment using anaconda using instructions provided in [PyRoboCOPInstallationSteps.md](PyRoboCOPInstallationSteps.md)

## Usage

1. All codes should be run inside the virtual python environment in Installation.

2. All the codes should be run from the directory PyRoboCOP. This folder contains the following folders

    ```
    a. Envs
    b. Solvers
    c. Simulations
    d. Results
    ```

3. Set the PYTHONPATH to the directory PyRoboCOP. We have added a bash file which can be used to set the PYTHONPATH.

    ```
    cd PyRoboCOP
    source setpath.sh
    ```

4. To run an OCP, for example, the inverted pendulum swing-up, run the following command

    ```
    python Simulations/main_pendulum.py
    ```

5. Every script in the Simulations folder solves an OCP for the dynamical system in the OCP. All the results reported in the paper could be generated using the scripts in the Simulations folder.

6. To run the comparisons between casadi and pyomo, run the following command (for the acrobot system).

    ```
    python Simulations/main_acrobot.py
    python Simulations/Casadi/acrobot.py
    python Simulations/pyomo/acrobot.py
    python Results/Acrobot/plot_acrobot.py
    ```

## Citation

If you use the software, please cite the following  ([MERL Technical Report](https://www.merl.com/publications/docs/TR2022-057.pdf)):

```
@inproceedings{Raghunathan2022may,
author = {Raghunathan, Arvind and Jha, Devesh K. and Romeres, Diego},
title = {PYROBOCOP: Python-based Robotic Control & Optimization Package for Manipulation},
booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
year = 2022,
month = may,
url = {https://www.merl.com/publications/TR2022-057}
}
```

([ArXiv](https://arxiv.org/pdf/2106.03220.pdf)):
```
@article{raghunathan2021pyrobocop,
  title={PYROBOCOP: Python-based robotic control \& optimization package for manipulation and collision avoidance},
  author={Raghunathan, Arvind U and Jha, Devesh K and Romeres, Diego},
  journal={arXiv preprint arXiv:2106.03220},
  year={2021}
}
```

## Related Links

[MERL Research Page](https://www.merl.com/research/license/PyRoboCOP)

## Contact

If you have any question please contact us at:

* Arvind U. Raghunathan: raghunathan@merl.com
* Devesh K. Jha: jha@merl.com
* Diego Romeres: romeres@merl.com

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## License

Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files:
```
Copyright (c) 2021-2022 Mitsubishi Electric Research Laboratories (MERL).

SPDX-License-Identifier: AGPL-3.0-or-later
```

Uses [ipopt](https://github.com/coin-or/Ipopt) which has `EPL-2.0` license.
