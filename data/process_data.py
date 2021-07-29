import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''A function to load the data from .csv files (messages and categories)
    and merge them.
    Args:
    - messages_filpath: A string of the messages dataset file path.
    - categories_filepath: A string of the categories dataset file path.
    Returns:
    df: Data frame of merged datasets.
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on='id')
    return df


def clean_data(df):
    '''A function to clean the data and handle the missing and duplicated data.
    Args:
    - df: A data frame of merged datasets.
    Returns:
    df: Data frame of clean data ready to be saved in database.
    '''
    categories = df['categories'].str.split(pat = ';',expand=True)
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x:x[:-2])
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].apply(lambda x:x[-1])
        categories[column] = categories[column].astype(int)

    df.drop(columns='categories',inplace=True)
    #replace 2 in 'related' column with 1 to make it binary.
    categories.related = np.where(categories['related']!=0,1,0)
    df = pd.concat([df,categories],axis=1)
    df.drop_duplicates(inplace=True)
    classes_list = list(df.columns[4:])
    df.dropna(subset=classes_list,how='any',inplace=True)

    return df


def save_data(df, database_filename):
    '''A function to save the dataset in a database sqlite file.
    Args:
    - df: A data frame of the clean dataset.
    - database_filename: string of the database file name.
    Returns:
    - None.
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages-categories', engine, index=False, if_exists='replace')


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
