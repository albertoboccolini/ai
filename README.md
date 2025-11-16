# ai

A simple implementation of neural networks with no dependencies. The scope of this project is to have a frictionless method to understand the core of neural networks and learn how to use them for simple tasks. The repository will follow my actual AI university course with me trying to do the actual implementations of the theory.

## Getting Started

This project requires **Go version 1.25.2 or higher**. Make sure you have a compatible version installed. If needed, download the latest version from [https://go.dev/dl/](https://go.dev/dl/)

1. Clone the repository

2. Test the current state of the project 

    ```bash
    go run main.go
    ```

3. Have fun and learn by experimenting with the code

## Project Structure

```
ai/
|--- main.go            # Entry point of the application with example usage
|--- go.mod
|--- predictors/
│   |--- predictor.go   # Training function for a single perceptron and predictor interfaces
|--- utils/
|   |--- utils.go       # Perceptron implementation, activation function and some other utilities
|--- README.md
```

## Contributing

If you notice any errors or bugs, feel free to open an issue or submit a PR with corrections. I am not a mathematician or an AI expert, just someone who wants to understand this subject, so I'm always open to contributions.
