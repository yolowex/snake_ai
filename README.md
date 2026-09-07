# Snake -- Human vs. Neural Network

A pretty pygame-ce Snake game built directly on top of `snake_engine.py`'s
`SnakeGame` class -- all movement, collision, food, and scoring logic comes
straight from the engine; `main.py` is purely rendering + input.

## Setup

```bash
pip install pygame-ce numpy
python main.py
```

## Controls

- **Human Control** button (or it's on by default): steer with Arrow keys / WASD
- **Neural Network** button: hands control to `NeuralNetworkAgent`, a small
  MLP that reads the engine's 11-value state vector and outputs an action.
  It ships with **random, untrained weights**, so it currently plays
  randomly/poorly -- it's wired up and ready for a real trained model.
- `Space` -- pause / resume, or restart after game over
- `R` -- restart anytime
- `Esc` -- quit

## Plugging in a trained model

Train an agent (e.g. a small DQN) against `SnakeGame` or the vectorized
`VecSnakeGame` for speed, using the same 11-value state as input and the
3-way `{straight, right, left}` action space as output. Save the resulting
weights with:

```python
np.savez("weights.npz", W1=W1, b1=b1, W2=W2, b2=b2)
```

Then in `main.py`, after `self.agent = NeuralNetworkAgent()`, add:

```python
self.agent.load_weights("weights.npz")
```

The sidebar will automatically switch its "AI STATUS" readout from
`UNTRAINED` to `TRAINED`.
