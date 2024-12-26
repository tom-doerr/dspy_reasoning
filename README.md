<h1 align="center">Reasoning Pipeline Experiments</h1>

<p align="center">
  <strong>Exploring AI reasoning capabilities through iterative problem-solving</strong>
</p>

<p align="center">
  <a href="https://github.com/tom-doerr/dspy_reasoning/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  </a>
  <a href="https://github.com/tom-doerr/dspy_reasoning/issues">
    <img src="https://img.shields.io/github/issues/tom-doerr/dspy_reasoning" alt="Issues">
  </a>
  <a href="https://github.com/tom-doerr/dspy_reasoning/pulls">
    <img src="https://img.shields.io/github/issues-pr/tom-doerr/dspy_reasoning" alt="Pull Requests">
  </a>
  <a href="https://github.com/tom-doerr/dspy_reasoning">
    <img src="https://img.shields.io/github/stars/tom-doerr/dspy_reasoning?style=social" alt="Stars">
  </a>
</p>

<p align="center">
  This project explores AI reasoning capabilities by implementing a simple pipeline for solving 
  mathematical and logical problems. It's a playground for experimenting with iterative reasoning 
  processes and analyzing their effectiveness.
</p>

<div align="center">
  <a href="#key-features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#license">License</a>
</div>

## Key Features

- **Jeopardy Question Generation**: Creates challenging Jeopardy-style questions with hints
- **Reasoning Pipeline**: Implements iterative reasoning with formal logical fallacy detection
- **Answer Verification**: Uses semantic matching to verify answer correctness
- **Benchmarking**: Measures pipeline performance on generated questions
- **Formal Logic Analysis**: Detects and scores common logical fallacies

## Experimental Setup

The system processes problems in batches to explore reasoning patterns:

- **Batch Size**: Small sets of problems (typically 5-10)
- **Iterative Processing**: Sequential reasoning across problems
- **Basic Analysis**: Simple metrics collection
- **Manual Validation**: Human review of solutions

## System Components

### 1. Jeopardy Dataset Generator
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