import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge the messages and categories datasets from CSV files.
    
    Args:
        messages_filepath (str): Filepath to the messages CSV file.
        categories_filepath (str): Filepath to the categories CSV file.
    
    Returns:
        df (DataFrame): Merged dataframe containing messages and categories.
    """
    # Load messages and categories datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge the datasets on the 'id' column
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    """
    Clean the merged dataframe by processing the categories and removing duplicates.
    
    Args:
        df (DataFrame): Merged dataframe of messages and categories.
    
    Returns:
        df (DataFrame): Cleaned dataframe with separate category columns and no duplicates.
    """
    # Split the 'categories' column into separate columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Extract category names from the first row
    row = categories.iloc[0]
    category_colnames = [x.split('-')[0] for x in row]
    categories.columns = category_colnames
    
    # Convert category values to numeric (0 or 1)
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)
    
    # Drop the original 'categories' column from the dataframe
    df = df.drop('categories', axis=1)
    
    # Concatenate the new category columns to the dataframe
    df = pd.concat([df, categories], axis=1)
    
    # Drop rows where 'related' is 2
    df = df[df['related'] != 2]
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
    """
    Save the cleaned dataframe to an SQLite database.
    
    Args:
        df (DataFrame): Cleaned dataframe to be saved.
        database_filename (str): Filepath to the SQLite database.
    """
    # Create an SQLite engine
    engine = create_engine(f'sqlite:///{database_filename}')
    
    # Save the dataframe to a table named 'Messages'
    df.to_sql('Messages', engine, index=False, if_exists='replace')

def main():
    """
    Main function to execute the ETL pipeline:
    - Load data from CSV files
    - Clean the data
    - Save the cleaned data to an SQLite database
    """
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