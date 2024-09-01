# Genetic Algorithm for Private Key Search

This project uses a genetic algorithm combined with machine learning to search for a Bitcoin private key that closely matches a target RIPEMD-160 hash.

## How It Works

- **Initial Population**: Random private keys are generated.
- **Fitness Function**: The fitness function evaluates how close a generated hash is to the target hash.
- **Crossover and Mutation**: Genetic operators are applied to evolve the population towards the target.
- **Quantum Tunneling Optimization**: Occasionally, the algorithm makes significant changes to escape local minima.
- **Machine Learning**: An MLPRegressor model is trained to predict fitness and guide the search.

## How to Run

.
.
.
.
