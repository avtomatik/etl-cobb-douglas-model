# ETL Cobb-Douglas Model

This project implements an ETL (Extract, Transform, Load) pipeline for processing economic datasets and analyzing them using a Cobb-Douglas production function. The goal is to transform and visualize various economic data, particularly focusing on capital, labor, and product information for different economic datasets.

## Datasets

The project uses several datasets related to Cobb-Douglas production functions, including datasets for historical U.S. economic data such as capital, labor, and physical product series.

## Key Features

* **Extract**: The `Dataset` enums allow extraction of data from various sources (local CSV files or URLs).
* **Transform**: The core analysis is done using the Cobb-Douglas production function, including:

  * Capital-Labor intensity
  * Labor productivity
  * Fixed Assets Turnover
  * Product trend lines (moving averages)
* **Visualize**: Generates various visualizations such as charts for actual vs computed product, deviations, and relative productivities.

## Requirements

* Python 3.12 or later
* Required Python packages:

  * `matplotlib`
  * `numpy`
  * `pandas`
  * `requests`

## Setup

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd etl-cobb-douglas-model
   ```

2. Create and activate a Python virtual environment:

   ```bash
   uv venv --python 3.12
   source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
   ```

3. Install the required dependencies:

   ```bash
   uv sync
   ```

4. Ensure your data is in the correct folder (`data/raw/`). The data files should be `.zip` files as indicated by the dataset names in the code.

## Usage

To run the project, use the following command:

```bash
uv run python main.py
```

This will:

* Load the datasets defined in the `combine_cobb_douglas()` function.
* Perform the transformation (e.g., calculating labor and capital productivity).
* Generate visualizations comparing actual vs theoretical production curves, deviations, and relative productivity of labor and capital.

## How It Works

1. **Data Extraction**: The `Dataset` enums define various datasets and URLs, with a function to load them into Pandas DataFrames.
2. **Data Transformation**: The `combine_cobb_douglas` function pulls data for the relevant economic variables (capital, labor, product) and prepares them for analysis.
3. **Analysis**: The `transform_cobb_douglas` function performs calculations like:

   * Labor-Capital intensity
   * Labor productivity
   * Fixed assets turnover
   * Computed vs actual product comparisons
4. **Visualization**: The `plot_cobb_douglas` function generates several plots, including:

   * Progress in manufacturing over time
   * Theoretical vs actual curves of production
   * Deviations of actual product from trend lines
   * Relative productivities of labor and capital

## Folder Structure

```plaintext
etl-cobb-douglas-model/
│
├── data/
│   └── raw/          # Folder for raw dataset files
│
├── .venv/            # Python virtual environment
│
├── main.py           # Main script to run the ETL and analysis
│
└── README.md         # This README file
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
