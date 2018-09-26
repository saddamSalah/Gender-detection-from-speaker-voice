import pandas as pd
import os


def make_filenames_csv( path):
    fs = os.listdir(str(path))
    wavs = ['testset/' + str(i) for i in fs if i.endswith('.wav')]
    df = pd.DataFrame(wavs) #, Gender])
    df.to_csv('../test_data.csv', header=['filename'], index=False)


def make_name_gender_csv(data_set_path):


    Gender = []
    output_data = {}
    items = []
    input_data = pd.read_csv(data_set_path)
    vals = input_data.values
    for i in range(len(vals)):
        for item in vals[i]:
            Gender.append(item.split('/')[0])
            items.append('data/' + str(item))
    output_data['Path'] = items
    output_data['Gender'] = Gender
    output_df = pd.DataFrame.from_dict(output_data)
    output_df.to_csv('input_dataset.csv', columns=['Path', 'Gender'], index=False)
    input_data = pd.read_csv('input_dataset.csv')
    print("Number of male: {}".format(input_data[input_data.Gender == 'male'].shape[0]))
    print("Number of female: {}".format(input_data[input_data.Gender == 'female'].shape[0]))


make_filename_csv('../testset')
make_name_gender_csv('input_dataset.csv')