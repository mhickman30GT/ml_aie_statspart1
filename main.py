import datetime
import os
import copy
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


# GLOBAL VARIABLES
TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
PATH = os.path.dirname(os.path.realpath(__file__))


class DataSet:
    """ Class holding values for dataset """

    def __init__(self, file, direct):
        """ Constructor for Dataset """
        self.data_name = f'Deaths In Custody Data Set'
        self.out_dir = direct
        self.file = file
        self.csv = pd.read_csv(self.file)
        self.reduced = pd.DataFrame()
        self.x_data = pd.DataFrame()
        self.toxic = pd.DataFrame()
        self.last_steps = pd.DataFrame()
        self.hetero_data = pd.DataFrame()
        self.bi_data = pd.DataFrame()
        self.homo_data = pd.DataFrame()
        self.features = list()

    def process(self):
        """ Processes data set """
        # Separate classification labels
        self.x_data = self.csv.drop(["Wiki_ID", "TOXICITY"], 1)
        self.toxic = self.csv["TOXICITY"]
        self.features = list(self.x_data.columns)

    def reduce(self):
        """ Reduces data set by 50% via random sampling """
        # Find any columns with only one unique value
        xcopy = copy.deepcopy(self.x_data)
        reduced = pd.DataFrame(columns=['Sexuality', 'Gender', 'Race', 'National Origin', 'Religion', 'Age', 'Disability'])
        new_toxic = list()
        last_list = list()
        hetero_list = list()
        bi_list = list()
        homo_list = list()

        for index, row in xcopy.iterrows():
            # Parse out sexuality row
            sexuality = 0
            if row.heterosexual or row.straight:
                sexuality = 1
            elif row.bisexual:
                sexuality = 2
            elif (row.gay or row.lesbian or row.queer or row.lgbt
                 or row.lgbtq or row.homosexual):
                 sexuality = 3

            # Parse out gender
            gender = 0
            if row.male:
                gender = 1
            elif row.female:
                gender = 2
            elif row.nonbinary:
                gender = 3
            elif row.transgender or row.trans:
                gender = 4

            # Parse out race
            race = 0
            if row.white:
                race = 1
            elif row.asian:
                race = 2
            elif row.hispanic or row.latino or row.latina or row.latinx:
                race = 3
            elif row.black or row.africanamerican:
                race = 4

            # Parse out national origin
            national_origin = 0
            if row.american:
                national_origin = 1
            elif row.canadian:
                national_origin = 2
            elif row.european:
                national_origin = 3
            elif row.mexican:
                national_origin = 4
            elif row.middleeastern:
                national_origin = 5
            elif row.indian:
                national_origin = 6
            elif row.chinese:
                national_origin = 7
            elif row.japanese:
                national_origin = 8
            elif row.african:
                national_origin = 9

            # Parse out religion
            religion = 0
            if row.christian or row.catholic or row.protestant:
                religion = 1
            elif row.jewish:
                religion = 2
            elif row.muslim:
                religion = 3
            elif row.buddhist:
                religion = 4
            elif row.sikh:
                religion = 5
            elif row.taoist:
                religion = 6

            # Parse out age
            age = 0
            if row.millenial:
                age = 1
            elif row.middleaged:
                age = 2
            elif row.old or row.older or row.elderly:
                age = 3

            # Parse out disability
            disability = 0
            if row.deaf:
                disability = 1
            elif row.blind:
                disability = 2
            elif row.paralyzed:
                disability = 3

            total = sexuality + gender + race + national_origin + religion + age + disability

            if total != 0:
                values = {'Sexuality': sexuality, 'Gender': gender, 'Race': race, 'National Origin': national_origin,
                          'Religion': religion, 'Age': age, 'Disability': disability}
                row_to_add = pd.Series(values)
                new_toxic.append(self.toxic[index])
                #if self.toxic[index] == 52671.0:
                    #print(f'Total {total} s {sexuality} g {gender} r {race} no {national_origin} r {religion} a {age} d {disability}')
                reduced = reduced.append(row_to_add, ignore_index=True)

                # Grab data set for final steps
                if sexuality != 0:
                    last_list.append(self.toxic[index])
                    if sexuality == 1:
                        hetero_list.append(self.toxic[index])
                    elif sexuality == 2:
                        bi_list.append(self.toxic[index])
                    else:
                        homo_list.append(self.toxic[index])

        self.toxic = pd.Series(new_toxic, name='toxicity')
        self.last_steps = pd.Series(last_list, name='sexuality')
        self.hetero_data = pd.Series(hetero_list, name='hetero')
        self.bi_data = pd.Series(bi_list, name='bi')
        self.homo_data = pd.Series(homo_list, name='homo')

        return reduced


def main():
    """ Main """
    # Process directories
    dir_name = os.path.join(os.path.join(PATH, "out"), "ML_Stats_Part1")
    out_dir = os.path.join(dir_name, TIME)
    os.makedirs(out_dir)

    # Initialize data class
    dataset = DataSet(os.path.join(os.path.join(PATH, "data"), "toxity_per_attribute.csv"), out_dir)
    dataset.process()

    # Core count for multi-proc
    core_count = round(multiprocessing.cpu_count() * .75)

    # Data Reduction
    reduced_data = dataset.reduce()
    dataset.toxic = dataset.toxic.reset_index()
    #dataset.total_data = reduced_data.append(dataset.toxic, 'Toxicity')
    #print(dataset.total_data)

    # Correlation Coefficents
    for name, col in reduced_data.items():
        coeff = col.astype("float64").corr(dataset.toxic.toxicity.astype("float64"))
        print(f'{name}: {coeff}')

    # Correlation Graphs
    fig, ax = plt.subplots()
    plt.scatter(x=reduced_data.Sexuality, y=dataset.toxic.toxicity, s=.05)
    plt.title("Toxicity over Sexuality Protected Class Data")
    plt.grid(True)
    fig.savefig(os.path.join(dataset.out_dir, f"tox_se.png"))

    fig, ax = plt.subplots()
    plt.scatter(x=reduced_data['National Origin'], y=dataset.toxic.toxicity, s=.05)
    plt.title("Toxicity over National Origin Protected Class Data")
    plt.grid(True)
    fig.savefig(os.path.join(dataset.out_dir, f"tox_no.png"))

    fig, ax = plt.subplots()
    plt.scatter(x=reduced_data.Age, y=dataset.toxic.toxicity, s=.05)
    plt.title("Toxicity over Age Protected Class Data")
    plt.grid(True)
    fig.savefig(os.path.join(dataset.out_dir, f"tox_age.png"))


    # Calculate Toxic Mean and std
    sum_tox = dataset.toxic.toxicity.sum()
    mean = sum_tox/dataset.toxic.toxicity.count()
    #print(f'Total: {sum_tox} Max: {dataset.toxic.toxicity.max()}')
    print(f'Toxicity Mean: {mean}')
    std = np.std(dataset.toxic.toxicity)
    print(f'Toxicity STD: {std}')
    error = (1.96 * std)/(np.sqrt(dataset.toxic.toxicity.count()))
    print(f'Toxicity Margin of Error:{error}')

    # Range of std values
    pos_std = norm.cdf(2 * std, mean, std)
    neg_std = norm.cdf(-2 * std, mean, std)
    print(f'95% range: {neg_std} - {pos_std}')

    # Sample data
    sampled_data_60 = dataset.toxic.toxicity.sample(frac=0.6, replace=False, random_state=1)
    sampled_data_10 = dataset.toxic.toxicity.sample(frac=0.1, replace=False, random_state=1)

    # Calculate Sampled Mean and std
    sum_tox = sampled_data_60.sum()
    mean_60 = sum_tox/sampled_data_60.count()
    print(f'60% Sampled Mean: {mean_60}')
    std_60 = np.std(sampled_data_60)
    print(f'60% Sampled STD: {std_60}')

    # Calculate Sampled Mean and std
    sum_tox = sampled_data_10.sum()
    mean_10 = sum_tox/sampled_data_10.count()
    print(f'10% Sampled Mean: {mean_10}')
    std_10 = np.std(sampled_data_10)
    print(f'10% Sampled STD: {std_10}')

    # Sample data
    sampled_data_60 = dataset.last_steps.sample(frac=0.6, replace=False, random_state=1)
    sampled_data_10 = dataset.last_steps.sample(frac=0.1, replace=False, random_state=1)

    # Calculate Toxic Mean and std
    sum_tox = dataset.last_steps.sum()
    sexual_mean = sum_tox/dataset.last_steps.count()
    print(f'Sexuality Toxicity Mean: {sexual_mean}')
    sexual_std = np.std(dataset.last_steps)
    print(f'Sexuality Toxicity STD: {sexual_std}')

    # Calculate Sampled Mean and std
    sum_tox = sampled_data_60.sum()
    sexual_mean_60 = sum_tox/sampled_data_60.count()
    print(f'60% Sexuality Sampled Mean: {sexual_mean_60}')
    sexual_std_60 = np.std(sampled_data_60)
    print(f'60% Sexuality Sampled STD: {sexual_std_60}')

    # Calculate Sampled Mean and std
    sum_tox = sampled_data_10.sum()
    sexual_mean_10 = sum_tox/sampled_data_10.count()
    print(f'10% Sexuality Sampled Mean: {sexual_mean_10}')
    sexual_std_10 = np.std(sampled_data_10)
    print(f'10% Sexuality Sampled STD: {sexual_std_10}')

    # Sample data
    sampled_data_60 = dataset.hetero_data.sample(frac=0.6, replace=False, random_state=1)
    sampled_data_10 = dataset.hetero_data.sample(frac=0.1, replace=False, random_state=1)

    # Calculate Toxic Mean and std
    sum_tox = dataset.hetero_data.sum()
    hetero_mean = sum_tox/dataset.hetero_data.count()
    print(f'Sexuality Toxicity Mean: {hetero_mean}')
    hetero_std = np.std(dataset.hetero_data)
    print(f'Sexuality Toxicity STD: {hetero_std}')

    # Calculate Sampled Mean and std
    sum_tox = sampled_data_60.sum()
    hetero_mean_60 = sum_tox/sampled_data_60.count()
    print(f'60% Hetero Sampled Mean: {hetero_mean_60}')
    hetero_std_60 = np.std(sampled_data_60)
    print(f'60% Hetero Sampled STD: {hetero_std_60}')

    # Calculate Sampled Mean and std
    sum_tox = sampled_data_10.sum()
    hetero_mean_10 = sum_tox/sampled_data_10.count()
    print(f'10% Hetero Sampled Mean: {hetero_mean_10}')
    hetero_std_10 = np.std(sampled_data_10)
    print(f'10% Hetero Sampled STD: {hetero_std_10}')

    # Sample data
    sampled_data_60 = dataset.bi_data.sample(frac=0.6, replace=False, random_state=1)
    sampled_data_10 = dataset.bi_data.sample(frac=0.1, replace=False, random_state=1)

    # Calculate Toxic Mean and std
    sum_tox = dataset.bi_data.sum()
    bi_mean = sum_tox/dataset.bi_data.count()
    print(f'Bi Toxicity Mean: {bi_mean}')
    bi_std = np.std(dataset.bi_data)
    print(f'Bi Toxicity STD: {bi_std}')

    # Calculate Sampled Mean and std
    sum_tox = sampled_data_60.sum()
    bi_mean_60 = sum_tox/sampled_data_60.count()
    print(f'60% Bi Sampled Mean: {bi_mean_60}')
    bi_std_60 = np.std(sampled_data_60)
    print(f'60% Bi Sampled STD: {bi_std_60}')

    # Calculate Sampled Mean and std
    sum_tox = sampled_data_10.sum()
    bi_mean_10 = sum_tox/sampled_data_10.count()
    print(f'10% Bi Sampled Mean: {bi_mean_10}')
    bi_std_10 = np.std(sampled_data_10)
    print(f'10% Bi Sampled STD: {bi_std_10}')

    # Sample data
    sampled_data_60 = dataset.homo_data.sample(frac=0.6, replace=False, random_state=1)
    sampled_data_10 = dataset.homo_data.sample(frac=0.1, replace=False, random_state=1)

    # Calculate Toxic Mean and std
    sum_tox = dataset.homo_data.sum()
    homo_mean = sum_tox/dataset.homo_data.count()
    print(f'Homo Toxicity Mean: {homo_mean}')
    homo_std = np.std(dataset.homo_data)
    print(f'Homo Toxicity STD: {homo_std}')

    # Calculate Sampled Mean and std
    sum_tox = sampled_data_60.sum()
    homo_mean_60 = sum_tox/sampled_data_60.count()
    print(f'60% Homo Sampled Mean: {homo_mean_60}')
    homo_std_60 = np.std(sampled_data_60)
    print(f'60% Homo Sampled STD: {homo_std_60}')

    # Calculate Sampled Mean and std
    sum_tox = sampled_data_10.sum()
    homo_mean_10 = sum_tox/sampled_data_10.count()
    print(f'10% Homo Sampled Mean: {homo_mean_10}')
    homo_std_10 = np.std(sampled_data_10)
    print(f'10% Homo Sampled STD: {homo_std_10}')

    fig, ax = plt.subplots()
    x_axis = 0

    plt.bar(x_axis + 1.3, homo_std, 0.1, label='Homo Sexual STD')
    plt.bar(x_axis + 1.2, homo_mean, 0.1, label='Homo Sexual Mean')
    plt.bar(x_axis + 1.0, bi_std, 0.1, label='Bi Sexual STD')
    plt.bar(x_axis + 0.9, bi_mean, 0.1, label='Bi Sexual Mean')
    plt.bar(x_axis + 0.7, hetero_std, 0.1, label='Hetero Sexual STD')
    plt.bar(x_axis + 0.6, hetero_mean, 0.1, label='Hetero Sexual Mean')
    plt.bar(x_axis + 0.4, sexual_std, 0.1, label='Sexuality STD')
    plt.bar(x_axis + 0.3, sexual_mean, 0.1, label='Sexuality Mean')
    plt.bar(x_axis + 0.1, std, 0.1, label='Population STD')
    plt.bar(x_axis + 0.0, mean, 0.1, label='Population Mean')

    plt.xticks([.1, .3, .6, .9, 1.2],['Population', 'Sexuality', 'HeteroSexual', 'BiSexual', 'HomoSexual'])
    plt.xlabel("Data from Population and Sexuality Protected Class")
    plt.ylabel("Mean and Standard Deviation in Toxicity")
    plt.title("Means and Standard Deviations of Toxicity Data")
    ax.xaxis_date()
    ax.autoscale(tight=True)
    plt.legend()
    fig.savefig(os.path.join(dataset.out_dir, f"final_graph.png"))


if __name__ == "__main__":
    main()
