import pandas as pd

def fill_missing_values(df, numeric_columns):
    """Fill missing values with the mean of their respective columns for numeric columns only."""
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    return df


def remove_outliers(df, columns):
    """Remove outliers based on the IQR method for numeric columns only."""
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        criteria = ((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))
        df = df[~criteria]
    return df


def main():
    # Load the data
    file_path = r'Tetuan City power consumption.csv'  # Adjust the file path as needed
    data = pd.read_csv(file_path)

    # Select numeric columns excluding 'DateTime'
    numeric_columns = data.select_dtypes(include=['number']).columns

    # Fill missing values
    data_filled = fill_missing_values(data, numeric_columns)

    # Remove outliers
    data_cleaned = remove_outliers(data_filled, numeric_columns)

    # Output the cleaned data to a new CSV file, without modifying the original file
    output_file_path = r'Tetuan City power consumption_cleaned.csv'  # Adjust the output file path as needed
    data_cleaned.to_csv(output_file_path, index=False)
    print(f"Cleaned data has been saved to {output_file_path}")


if __name__ == "__main__":
    main()
