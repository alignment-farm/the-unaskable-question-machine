"""Explicit finite EBM reference: p(x|task) proportional to exp(-E(x)/T).

The energy is hand-specified (constraint violations), not learned. Single-site
Metropolis inference is approximate; exhaustive enumeration is a separate oracle.
This adapter accepts structured tasks, not language prompts. No neural EBM claims.
"""
import itertools
import math
import random
from src.experiments.tasks import ConstraintTask


def exact_distribution(task: ConstraintTask, temperature: float = 1.0) -> dict:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if task.n > 16:
        raise ValueError("exact enumeration limited to 16 bits")
    states = list(itertools.product((0, 1), repeat=task.n))
    energies = [task.energy(s) for s in states]
    weights = [math.exp(-e / temperature) for e in energies]
    partition = sum(weights)
    return {"states": states, "energies": energies,
            "probabilities": [w / partition for w in weights],
            "partition": partition, "minimum_energy": min(energies)}


def metropolis(task: ConstraintTask, steps: int, temperature: float, seed: int) -> dict:
    if steps < 0 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("steps must be nonnegative and temperature finite and positive")
    rng = random.Random(seed)
    state = tuple(rng.randrange(2) for _ in range(task.n))
    initial = state
    energy = task.energy(state)
    best, best_energy = state, energy
    trajectory = [energy]
    accepted = 0
    for _ in range(steps):
        index = rng.randrange(task.n)
        candidate = tuple(1 - b if i == index else b for i, b in enumerate(state))
        next_energy = task.energy(candidate)
        delta = next_energy - energy
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            state, energy = candidate, next_energy
            accepted += 1
        if energy < best_energy:
            best, best_energy = state, energy
        trajectory.append(energy)
    return {"initial_bits": initial, "bits": best, "energy": best_energy, "final_bits": state, "final_energy": energy,
            "trajectory": trajectory, "accepted": accepted,
            "energy_evaluations": steps + 1, "seed": seed, "steps": steps,
            "temperature": temperature, "selection": "lowest energy visited (optimization, not equilibrium sample)",
            "model_kind": "hand-specified finite EBM; no learned weights"}
