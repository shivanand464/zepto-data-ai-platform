# Data Pipeline

## Overview

This module implements an end-to-end data engineering pipeline using the public `books.toscrape.com` scraping-practice website.

The pipeline performs the following steps:

1. Scrape book data using `requests` and `BeautifulSoup`.
2. Scrape all pages from three selected book categories.
3. Clean the scraped fields using pandas.
4. Convert GBP prices to INR using the required fixed project rate.
5. Store the cleaned data in a normalized SQLite database.
6. Execute SQL queries demonstrating the required SQL operations.
7. Validate SQL results against pandas operations.

## Dataset

The pipeline scrapes three categories:

* Travel — 11 books
* Mystery — 32 books
* Historical Fiction — 26 books

Total books scraped: **69**

The scraper automatically follows pagination links until the final page of each selected category.

## Scraped Fields

The raw scraper collects:

* `title`
* `price`
* `star_rating`
* `availability`
* `category`

## Data Cleaning

The following transformations are applied:

### Price

The original price is scraped as text containing a currency symbol.

Non-numeric characters are removed and the result is converted to the `float` column `price_gbp`.

If a price fails to parse, the median `price_gbp` is used for imputation.

### Rating

The text ratings `One`, `Two`, `Three`, `Four`, and `Five` are mapped to integers from 1 to 5.

If an unexpected rating fails to parse, the median rating is used and rounded to the nearest integer.

### Availability

The availability text is converted to the boolean column `in_stock`.

Values containing `In stock` are represented as `True`; other values are represented as `False`.

### Currency Conversion

The project-defined fixed conversion rate is:

**1 GBP = 105.50 INR**

The `price_inr` column is calculated as:

```text
price_inr = price_gbp × 105.50
```

No external currency API is used.

## Data Quality

The final dataset contains:

* 69 rows
* 3 categories
* 0 missing values in the cleaned fields

Cleaned column types include:

* `price_gbp` — float
* `rating` — integer
* `in_stock` — boolean
* `price_inr` — float

## Database Design

The data is stored in SQLite using two normalized tables.

### `categories`

| Column          | Type    | Constraint       |
| --------------- | ------- | ---------------- |
| `category_id`   | INTEGER | Primary Key      |
| `category_name` | TEXT    | UNIQUE, NOT NULL |

### `books`

| Column        | Type    | Constraint  |
| ------------- | ------- | ----------- |
| `book_id`     | INTEGER | Primary Key |
| `title`       | TEXT    | NOT NULL    |
| `price_gbp`   | REAL    | NOT NULL    |
| `price_inr`   | REAL    | NOT NULL    |
| `rating`      | INTEGER | NOT NULL    |
| `in_stock`    | INTEGER | NOT NULL    |
| `category_id` | INTEGER | Foreign Key |

The relationship is:

```text
categories.category_id
        ↑
        |
books.category_id
```

The category name is stored once in the `categories` table rather than repeated for every book.

## SQL Queries

Six SQL queries are implemented in `queries.py`.

| Query   | Demonstrated operation |
| ------- | ---------------------- |
| Query 1 | `SELECT` and `WHERE`   |
| Query 2 | `ORDER BY`             |
| Query 3 | `LIMIT`                |
| Query 4 | `DISTINCT`             |
| Query 5 | `BETWEEN`              |
| Query 6 | `JOIN`                 |

The executed SQL statements and their outputs are saved in:

```text
query_results.txt
```

## Pandas Validation

The SQL results are read into pandas using `pd.read_sql()`.

The JOIN result is independently reproduced using:

```python
pd.merge()
```

The SQL JOIN and pandas merge results were compared and produced:

```text
JOIN SQL and pd.merge() equivalent: True
```

This confirms that the relational JOIN and its pandas equivalent produce matching results.

## Files

```text
data_pipeline/
├── books.db
├── check_database.py
├── queries.py
├── query_results.txt
├── scrape_and_load.py
├── validate_pandas.py
└── README.md
```

## How to Run

From the project root:

### 1. Run the complete scraping and database pipeline

```powershell
python data_pipeline\scrape_and_load.py
```

This scrapes the selected categories, cleans the data, performs the GBP-to-INR conversion, and recreates the SQLite database.

### 2. Verify the database

```powershell
python data_pipeline\check_database.py
```

### 3. Run the SQL queries

```powershell
python data_pipeline\queries.py
```

The query output is saved to:

```text
data_pipeline/query_results.txt
```

### 4. Validate SQL against pandas

```powershell
python data_pipeline\validate_pandas.py
```

The final validation should report:

```text
JOIN SQL and pd.merge() equivalent: True
```
