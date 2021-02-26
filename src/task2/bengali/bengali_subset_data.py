import pandas as pd

"""
This script samples the Bengali hatespeech data equivalent to the Hindi hatespeech dataset.
"""

def subset_data(n_positive, n_negative, df, split_col, random_seed=42):
    ## sampling the required samples for each label
    positive_df = df[df[split_col] == 1].sample(n_positive, random_state=random_seed)
    negative_df = df[df[split_col] == 0].sample(n_negative, random_state=random_seed)
    
    ## creating list of frames for concatenation
    frames = [positive_df, negative_df]
    
    return pd.concat(frames)

def save_data(df, path):
    df.to_csv(path) ## writing to file
    print('data saved to file')

def main():
    data = pd.read_csv('data/bengali_hatespeech.csv')

    ## sampling same distribution of labels as was given in hindi dataset
    bengali_subset_df = subset_data(n_positive=2469, n_negative=2196, df=data, split_col='hate')
    save_data(df=bengali_subset_df, path='data/bengali_hatespeech_subset.csv')


if __name__ == "__main__":
    main()

