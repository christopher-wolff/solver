# Nash Equilibrium Solver

This repository contains a simple [0,1] poker solver implemented using Apple's
[MLX](https://github.com/ml-explore/mlx) framework. The solver discretises card
values and bet sizes, runs gradient-based training to approximate a Nash
equilibrium and serialises the resulting strategies to JSON.

## Setup

It is recommended to run the solver in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the solver with

```bash
python3 solver.py [--device cpu|gpu]
```
The device flag is optional and defaults to running on the CPU.

The script prints progress, saves `p1_strategy.json` and `p2_strategy.json` and
reports the final expected value (EV) for both players along with the average
bet for each card value.

Run the tests with

```bash
pytest -q
```
