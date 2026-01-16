# ETL Cobb-Douglas Model

This project implements an ETL (Extract, Transform, Load) pipeline for processing economic datasets and analyzing them using the Cobb-Douglas production function. The main goal is to transform and visualize various economic data, particularly focusing on capital, labor, and product information, for different economies and periods. The visualizations generated closely follow the plots presented in C.W. Cobb and P.H. Douglas's 1928 paper, "A Theory of Production."

## Key Features

* **Ingest**: Loads raw economic data into a DuckDB warehouse.
* **Transform**: Uses DBT for transforming the data according to the Cobb-Douglas production function.
* **Visualize**: Generates plots that match the ones presented in Cobb and Douglas's 1928 paper.

## Datasets

The project uses several datasets related to the Cobb-Douglas production function, such as historical U.S. economic data (capital, labor, and physical product series). The project is flexible enough to accept additional datasets for other economies and time periods, which will be added in the future.

## Requirements

* Python 3.12 or later
* Required Python packages:

  * `dbt-duckdb>=1.10.0`
  * `duckdb>=1.4.3`
  * `matplotlib>=3.10.8`
  * `pandas>=2.3.3`
  * `pyyaml>=6.0.3`

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/avtomatik/etl-cobb-douglas-model.git
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

4. Ensure your data is in the correct folder (`data/raw/`). The data files should be `.zip` files as indicated by the dataset names in the `sources.yml`.

## Usage

### Ingest Data into DuckDB Warehouse

To ingest data into your DuckDB warehouse, run the following command:

```bash
uv run python -m ingest.main
```

This will:

* Load the raw data from the `.zip` files located in the `data/raw/` directory.
* Store the data into a DuckDB warehouse under the `raw` schema.

### Transformation with DBT

To transform the raw data using DBT and the Cobb-Douglas production function, run:

```bash
dbt run --profiles-dir dbt --project-dir dbt
```

This will:

* Transform the raw data into intermediate, staging, and final tables according to the Cobb-Douglas production function and other relevant calculations.
* The transformations are defined in the `dbt/models/` directory.

You can specify different sets of triplets for different scenarios using the `--vars` flag. For example:

```bash
dbt run --vars '{active_cobb_douglas_spec: historical_douglas}' --profiles-dir dbt --project-dir dbt
```

Where you can change the value of `active_cobb_douglas_spec` to different configurations (e.g., `historical_nber`, `historical_douglas`, etc.).

### Visualize Results

To visualize the results, run the following command in **Spyder** (or another interactive environment that supports matplotlib visualization):

```bash
%pwd
# Ensure the current working directory is set to the project root, where 'etl-cobb-douglas-model' is located.
# Output should be: '/path/to/etl-cobb-douglas-model'

%run -m viz.run_all
```

This will:

* Generate the plots as presented in Cobb and Douglas’s 1928 paper.
* **Note**: Plots are not displayed when running `uv run python -m viz.run_all` due to non-interactive backend limitations (e.g., `FigureCanvasAgg`). It is recommended to run the visualizations in Spyder or another IDE with an interactive plot window.
* Ensure you’re in the **project root directory** (`/path/to/etl-cobb-douglas-model`) before running the script, or use the `%cd` command to change to the correct directory.

---

## Folder Structure


```plaintext
etl-cobb-douglas-model/
│
├── core/               # Core configuration and data handling
│   ├── config.py       # Configuration settings
│   └── data.py         # Data handling functions
│
├── data/               # Data storage and configuration
│   ├── processed/      # Processed data (e.g., intermediate results)
│   ├── raw/            # Raw data (e.g., .zip files)
│   └── sources.yml     # Source configurations (paths, columns, etc.)
│
├── dbt/                # DBT project for data transformations
│   ├── dbt_project.yml # DBT project settings
│   ├── models/         # DBT models (SQL transformations)
│   │   ├── intermediate/
│   │   │   ├── int_inputs.sql
│   │   │   ├── int_normalized_data.sql
│   │   │   └── int_productivity_metrics.sql
│   │   ├── marts/
│   │   │   ├── cobb_douglas_estimates.sql
│   │   │   └── cobb_douglas_series.sql
│   │   ├── sources.yml # DBT source configurations
│   │   └── staging/
│   │       ├── stg_douglas.sql
│   │       ├── stg_usa_cobb_douglas.sql
│   │       └── stg_uscb.sql
│   └── profiles.yml    # DBT profiles for connection settings
│
├── ingest/             # Data ingestion scripts
│   └── main.py         # Main ingestion script
│
├── LICENSE.md          # Project license
├── pyproject.toml      # Python project settings
├── README.md           # This README file
├── uv.lock             # uvicorn lock file
└── viz/                # Visualization scripts
    ├── plot.py         # Plotting functions
    └── run_all.py      # Script to run all visualizations
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

---
