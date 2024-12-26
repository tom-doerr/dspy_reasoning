# Reasoning Pipeline Benchmark

This project implements and benchmarks an AI reasoning pipeline for solving complex problems, with a focus on logical reasoning and mathematical problem-solving.

## Features

- **Jeopardy Question Generation**: Creates challenging Jeopardy-style questions with hints
- **Reasoning Pipeline**: Implements iterative reasoning with formal logical fallacy detection
- **Answer Verification**: Uses semantic matching to verify answer correctness
- **Benchmarking**: Measures pipeline performance on generated questions
- **Formal Logic Analysis**: Detects and scores common logical fallacies

## Components

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
   git clone https://github.com/tom-doerr/reasoning-pipeline.git
   cd reasoning-pipeline
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```bash
   export DSPY_MODEL=deepseek/deepseek-chat
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

## Results

Benchmark results include:
- Overall accuracy
- Average iterations per question
- Total processing time
- Fallacy detection rates
- Objective achievement scores

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
