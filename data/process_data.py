import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load messages and categories datasets"""
    
    # Load CSV files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge dataframes
    df = messages.merge(categories, on="id")

    return df



def clean_data(df):
    """Clean dataset"""

    # Split categories column into multiple columns
    categories = df.categories.str.split(";", expand=True)

    # Assign new column names
    row = categories.iloc[0]
    category_colnames = [cat.split("-")[0] for cat in row]
    categories.columns = category_colnames

    # Extract numeric values from new columns
    for column in categories:
        # Extract the last character of the string
        categories[column] = categories[column].str[-1]
        
        # Convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # Replace values 2 => 1
    categories.replace(2, 1, inplace=True)

    # Drop the original categories column
    df.drop(columns="categories", axis=1, inplace=True)

    # Concatenate new categories column to the merged dataframe
    df = pd.concat([df, categories], axis=1)

    # Drop columns with only zeroes
    empty_cols = [col for col in category_colnames if df[col].sum() == 0]
    df.drop(empty_cols, axis=1, inplace=True)

    # Drop duplicate rows
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """Save clean dataframe to sqlite database"""

    # Create sqlite database
    engine = create_engine("sqlite:///" + database_filename)

    # Save dataframe to table 'disasters'
    df.to_sql("disasters", engine, index=False, if_exists="replace")  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()