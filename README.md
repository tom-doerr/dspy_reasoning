<h1 align="center">Reasoning Pipeline Experiments</h1>

<p align="center">
  <strong>Exploring AI reasoning capabilities using DSPy</strong>
</p>

<p align="center">
  <a href="https://github.com/tom-doerr/dspy_reasoning/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square&logo=opensourceinitiative&logoColor=white" alt="License">
  </a>
  <a href="https://github.com/tom-doerr/dspy_reasoning/issues">
    <img src="https://img.shields.io/github/issues/tom-doerr/dspy_reasoning?style=flat-square&logo=github&logoColor=white" alt="Issues">
  </a>
  <a href="https://github.com/tom-doerr/dspy_reasoning/pulls">
    <img src="https://img.shields.io/github/issues-pr/tom-doerr/dspy_reasoning?style=flat-square&logo=github&logoColor=white" alt="Pull Requests">
  </a>
  <a href="https://github.com/tom-doerr/dspy_reasoning">
    <img src="https://img.shields.io/github/stars/tom-doerr/dspy_reasoning?style=flat-square&logo=github&logoColor=white" alt="Stars">
  </a>
  <a href="https://github.com/tom-doerr/dspy_reasoning/commits/main">
    <img src="https://img.shields.io/github/last-commit/tom-doerr/dspy_reasoning?style=flat-square&logo=github&logoColor=white" alt="Last Commit">
  </a>
</p>

<p align="center">
  This project explores how DSPy can be used to implement and analyze AI reasoning processes.
  It's a work in progress for experimenting with different reasoning approaches and patterns.
</p>

<div align="center">
  <a href="#what-it-does">What it does</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#license">License</a>
</div>

## What it does

- Implements iterative reasoning processes
- Analyzes reasoning patterns and logical validity
- Tracks reasoning performance metrics
- Provides detailed reasoning analysis

## Current Limitations

This is an experimental project with several known limitations:

- Reasoning quality depends heavily on the underlying model
- Analysis capabilities are still basic
- Performance metrics are simple
- Needs more diverse test cases

## System Components

### 1. Math Multiplication Optimizer
- Implements multiplication solver using DSPy Chain-of-Thought
- Uses MIPROv2 for optimization
- Generates random multiplication problems for training
- Evaluates accuracy on validation set
- Example usage:
  ```bash
  python3 math_multiplication_optimizer.py
  ```

### 2. Residual Pipeline Optimizer
- Optimizes search-replace pipelines using BootstrapFewShot or MIPROv2
- Supports both standard and iterative pipeline types
- Tracks optimization history and best configurations
- Example usage:
  ```bash
  python3 residual_pipeline_optimizer.py --pipeline-type standard --optimizer mipro
  ```

### 3. Jeopardy Dataset Generator
Generates challenging Jeopardy-style questions across multiple categories:
- Creates initial questions and hints
- Produces more challenging final questions
- Saves dataset to `jeopardy_dataset.json`

### 2. Reasoning Pipeline
Implements iterative reasoning with:
- Context tracking and history
- Objective achievement analysis
- Formal logical fallacy detection:
  - Affirming the consequent
  - Denying the antecedent
  - Undistributed middle
  - Illicit major/minor
- Mathematical validation
- Termination logic

### 3. Benchmark System
Measures pipeline performance by:
- Running reasoning pipeline on generated questions
- Verifying answers using semantic matching
- Tracking metrics:
  - Accuracy
  - Iterations per question
  - Processing time
  - Fallacy detection rates

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tom-doerr/dspy_reasoning.git
   cd dspy_reasoning
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```bash
   export DSPY_MODEL=deepseek/deepseek-chat
   ```

4. (Optional) Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

## Usage

1. Generate Jeopardy questions:
   ```bash
   ./jeopardy_dataset.py -n 50
   ```

2. Run reasoning pipeline benchmark:
   ```bash
   ./benchmark.py
   ```

3. View results in `reasoning_benchmark.json`

## Configuration

Customize settings in the scripts:
- `jeopardy_dataset.py`: Adjust categories and question count
- `reasoning_pipeline.py`: Modify reasoning parameters
- `benchmark.py`: Change evaluation metrics

## Performance Metrics

The system tracks comprehensive performance metrics across material batches:

- **Batch Processing Time**: Average time per batch
- **Batch Accuracy**: Percentage of correct solutions per batch
- **Iteration Efficiency**: Average reasoning iterations per problem
- **Fallacy Detection Rate**: Percentage of detected logical fallacies
- **Objective Achievement**: Success rate in meeting problem objectives

Benchmark results include aggregate statistics across all batches:
- Overall accuracy
- Average iterations per question
- Total processing time
- Fallacy detection rates
- Objective achievement scores

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

Please make sure to update tests as appropriate and follow the coding style of the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
