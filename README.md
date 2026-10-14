# CustomerIQ — Customer Segmentation & Marketing Intelligence

CustomerIQ is a small Python web app for exploring customer data and grouping customers with K-Means clustering. It is built with Streamlit and is intended as a learning project in data analysis and machine learning.

## What it currently does

- Accepts a customer dataset in CSV format.
- Shows a basic dataset overview and preview.
- Lets the user choose two numerical features.
- Supports automatic or manual selection of the number of clusters.
- Scales the selected features and applies K-Means.
- Displays a scatter plot with customer segments and cluster centres.
- Summarizes each segment and shows rule-based marketing suggestions.
- Exports the segmented customer data and segment summary as CSV files.

## Tech stack

- Python
- Streamlit
- pandas and NumPy
- scikit-learn (StandardScaler and K-Means)
- Matplotlib and Seaborn

## Run locally

1. Install Python 3.10 or newer.
2. Clone or download this repository.
3. Open a terminal in the project folder.
4. Create and activate a virtual environment (recommended).
5. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

6. Start the app:

   ```bash
   streamlit run app.py
   ```

7. Open the local URL printed in the terminal and upload a CSV file with at least two usable numerical columns.

## How the segmentation works

The app standardizes the two selected numerical features before fitting K-Means. It then summarizes the clusters and assigns readable customer labels using the selected dataset's feature medians. These labels and marketing suggestions are simple rules for demonstration; they are not predictions from a trained marketing model.

## Current limitations

- The current visualization uses two selected numerical features.
- The customer labels and marketing recommendations are heuristic, not validated business advice.
- Results depend on the dataset, selected features, and chosen number of clusters.
- Do not upload confidential or personally identifiable customer data to a public demo.

## Planned improvements

- Add clearer cluster-quality evaluation and explain how the suggested number of clusters is chosen.
- Add tests for data cleaning and segmentation.
- Improve validation and error messages for unusual CSV files.
- Add example data that contains no personal information.
- Improve accessibility and mobile layout.

## Learning notes

This is an evolving project. Before presenting or extending it, make sure you can explain the data-cleaning steps, feature scaling, K-Means, the limitations of median-based labels, and how the exported files are produced. Add your own screenshots, design decisions, and improvements as you work on it.

**Author:** [Your name or GitHub username]

**Project status:** Learning project / prototype.
