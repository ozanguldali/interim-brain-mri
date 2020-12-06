import csv
from random import randrange, shuffle
from shutil import copyfile
from math import ceil
import numpy as np

from util.file_util import prepare_directory

ROOT_DIR = "/Users/ozanguldali/Documents/thesis/modelsWithLASSO/"
SOURCE_DIR = "/Users/ozanguldali/Documents/thesis/covid-chestxray-dataset/images/"
CSV_PATH = "/Users/ozanguldali/Documents/thesis/modelsWithLASSO/metadata.csv"

train_covid_19_folder = ROOT_DIR+'dataset_notunique/train/COVID-19/'
train_non_covid_19_folder = ROOT_DIR+'dataset_notunique/train/non-COVID-19/'
test_covid_19_folder = ROOT_DIR+'dataset_notunique/test/COVID-19/'
test_non_covid_19_folder = ROOT_DIR+'dataset_notunique/test/non-COVID-19/'

# whole dataset_unique list init
covid_chestxray_dataset = []


def dataset_investigate():
    covid_patient_ids = []
    non_covid_patient_ids = []

    # read metadata file
    with open(CSV_PATH, mode='r') as csv_data:
        reader = csv.DictReader(csv_data, delimiter=',')

        for row in reader:
            covid_chestxray_dataset.append(row)

    # filter dataset_unique for COVID-19 patients having sex and age info, and PA X-ray image
    for data in covid_chestxray_dataset:
        if "COVID-19" in str(data["finding"]) \
                and data["sex"] != "" \
                and data["age"] != "" \
                and data["modality"] == "X-ray" \
                and (data["view"] == "PA"):

            covid_patient_ids.append(data["patientid"])

    print("not unique covid: ", len(covid_patient_ids))
    print("unique covid: ", len(np.unique(covid_patient_ids)))
    print()

    # filter dataset_unique for non-COVID-19 patients or healthy people having sex and age info, and PA X-ray image
    for data in covid_chestxray_dataset:
        if "COVID-19" not in str(data["finding"]) \
                and data["sex"] != "" \
                and data["age"] != "" \
                and data["modality"] == "X-ray" \
                and (data["view"] == "PA"):

            non_covid_patient_ids.append(data["patientid"])

    print("not unique non-covid: ", len(non_covid_patient_ids))
    print("unique non-covid: ", len(np.unique(non_covid_patient_ids)))
    print()

    # check whether there exists any common data
    common_ids = set(covid_patient_ids).intersection(set(non_covid_patient_ids))

    common_ids = list(common_ids)

    print("common ids: ", len(common_ids))

    if len(common_ids) > 0:
        # remove common id info from control group
        if len(non_covid_patient_ids) > len(covid_patient_ids):
            for common_id in common_ids:
                non_covid_patient_ids.remove(common_id)
        elif len(covid_patient_ids) > len(non_covid_patient_ids):
            for common_id in common_ids:
                covid_patient_ids.remove(common_id)
        else:
            for common_id in common_ids:
                non_covid_patient_ids.remove(common_id)


    print("not unique covid: ", len(covid_patient_ids))
    print("unique covid: ", len(np.unique(covid_patient_ids)))
    print()
    print("not unique non-covid: ", len(non_covid_patient_ids))
    print("unique non-covid: ", len(np.unique(non_covid_patient_ids)))
    print()


def construct_dataset(unique=False, balanced=False, reset=False, create=False):
    if reset:
        prepare_directory(train_covid_19_folder)
        prepare_directory(train_non_covid_19_folder)
        prepare_directory(test_covid_19_folder)
        prepare_directory(test_non_covid_19_folder)

    # id list and whole data list inits
    covid_dataset = []
    non_covid_dataset = []
    covid_patient_ids = []
    non_covid_patient_ids = []

    # read metadata file
    with open(CSV_PATH, mode='r') as csv_data:
        reader = csv.DictReader(csv_data, delimiter=',')

        for row in reader:
            covid_chestxray_dataset.append(row)

# ---------------------------------------------------------------------------------------------------------------------

    # filter dataset_unique for COVID-19 patients having sex and age info, and PA X-ray image
    for data in covid_chestxray_dataset:
        if "COVID-19" in str(data["finding"]) \
                and data["sex"] != "" \
                and data["age"] != "" \
                and data["modality"] == "X-ray" \
                and (data["view"] == "PA"):

            if unique:
                if data["patientid"] not in covid_patient_ids:
                    covid_patient_ids.append(data["patientid"])
                    covid_dataset.append(
                        {
                            "id": data["patientid"],
                            "sex": data["sex"],
                            "age": data["age"],
                            "finding": "COVID-19",
                            "fileName": data["filename"]
                        }
                    )
            else:
                covid_patient_ids.append(data["patientid"])
                covid_dataset.append(
                    {
                        "id": data["patientid"],
                        "sex": data["sex"],
                        "age": data["age"],
                        "finding": "COVID-19",
                        "fileName": data["filename"]
                    }
                )

    # filter dataset_unique for non-COVID-19 patients or healthy people having sex and age info, and PA X-ray image
    for data in covid_chestxray_dataset:
        if "COVID-19" not in str(data["finding"]) \
                and data["sex"] != "" \
                and data["age"] != "" \
                and data["modality"] == "X-ray" \
                and (data["view"] == "PA"):

            if unique:
                if data["patientid"] not in non_covid_patient_ids:
                    non_covid_patient_ids.append(data["patientid"])
                    non_covid_dataset.append(
                        {
                            "id": data["patientid"],
                            "sex": data["sex"],
                            "age": data["age"],
                            "finding": "non-COVID-19",
                            "fileName": data["filename"]
                        }
                    )
            else:
                non_covid_patient_ids.append(data["patientid"])
                non_covid_dataset.append(
                    {
                        "id": data["patientid"],
                        "sex": data["sex"],
                        "age": data["age"],
                        "finding": "non-COVID-19",
                        "fileName": data["filename"]
                    }
                )

    print("not unique covid: ", len(covid_patient_ids))
    print("unique covid: ", len(np.unique(covid_patient_ids)))
    print()
    print("not unique non-covid: ", len(non_covid_patient_ids))
    print("unique non-covid: ", len(np.unique(non_covid_patient_ids)))
    print()

# ---------------------------------------------------------------------------------------------------------------------

    # check whether there exists any common data
    common_ids = set(covid_patient_ids).intersection(set(non_covid_patient_ids))

    common_ids = list(common_ids)

    if len(common_ids) > 0:
        # remove common id info from small dataset
        if len(non_covid_patient_ids) >= len(covid_patient_ids):
            for common_id in common_ids:
                non_covid_patient_ids.remove(common_id)
            non_covid_dataset = remove_refuse_info_list_from_list(common_ids, "id", non_covid_dataset)
        elif len(covid_patient_ids) > len(non_covid_patient_ids):
            for common_id in common_ids:
                covid_patient_ids.remove(common_id)
            covid_dataset = remove_refuse_info_list_from_list(common_ids, "id", covid_dataset)

    print("not unique covid: ", len(covid_patient_ids))
    print("unique covid: ", len(np.unique(covid_patient_ids)))
    print()
    print("not unique non-covid: ", len(non_covid_patient_ids))
    print("unique non-covid: ", len(np.unique(non_covid_patient_ids)))
    print()

# ---------------------------------------------------------------------------------------------------------------------

    if balanced:
        # case train: experiment group is 'larger' than control group
        if len(covid_patient_ids) > len(non_covid_patient_ids):
            # elect data from larger dataset_unique to build the balance
            covid_patient_ids, covid_dataset = elect_from_larger_dataset(small=non_covid_patient_ids,
                                                                         large=covid_patient_ids, dataset=covid_dataset)

            print("not unique covid: ", len(covid_patient_ids))
            print("unique covid: ", len(np.unique(covid_patient_ids)))

        # case 2: experiment group is 'smaller' than control group
        elif len(covid_patient_ids) < len(non_covid_patient_ids):
            # elect data from larger dataset_unique to build the balance
            non_covid_patient_ids, non_covid_dataset = elect_from_larger_dataset(small=covid_patient_ids,
                                                                                 large=non_covid_patient_ids,
                                                                                 dataset=non_covid_dataset)

            print("not unique non-covid: ", len(non_covid_patient_ids))
            print("unique non-covid: ", len(np.unique(non_covid_patient_ids)))

        # case 3: experiment group has equal size with control group -> which is okay

# ---------------------------------------------------------------------------------------------------------------------

    # construct the dataset_unique and shuffle it
    ids = covid_patient_ids + non_covid_patient_ids

    shuffle(covid_dataset)
    shuffle(non_covid_dataset)
    dataset = covid_dataset + non_covid_dataset
    shuffle(dataset)

    # get size of train and test sets
    total_size = len(dataset)
    covid_size = len(covid_dataset)
    non_covid_size = len(non_covid_dataset)

    covid_train_size = ceil(int(covid_size * 4 / 5))
    covid_train_size = covid_train_size + 1 if covid_train_size % 2 != 0 else covid_train_size
    covid_test_size = covid_size - covid_train_size

    non_covid_train_size = ceil(int(non_covid_size * 4 / 5))
    non_covid_train_size = non_covid_train_size + 1 if non_covid_train_size % 2 != 0 else non_covid_train_size
    non_covid_test_size = non_covid_size - non_covid_train_size

    train_size = covid_train_size + non_covid_train_size # int(total_size * 4 / 5)
    test_size = total_size - train_size

    print("\ntrain set size: ", train_size)
    print("\ntest set size: ", test_size)

    # construct train and test sets
    train_ids = []
    train_set = []
    test_ids = []
    test_set = []

    covid_iter = 0
    non_covid_iter = 0
    for i in range(train_size):
        if i < covid_train_size:
            train_set.append(covid_dataset[covid_iter])
            covid_iter += 1
        else:
            train_set.append(non_covid_dataset[non_covid_iter])
            non_covid_iter += 1

    # covid_iter = 0
    # non_covid_iter = 0
    for i in range(test_size):
        if i < covid_test_size:
            test_set.append(covid_dataset[covid_iter])
            covid_iter += 1
        else:
            test_set.append(non_covid_dataset[non_covid_iter])
            non_covid_iter += 1

    print('\ntrain set:')
    for train in train_set:
        print(''.join([train["id"], ' -> ', train["fileName"], ' -> ', train["finding"]]))

    print('\ntest set:')
    for test in test_set:
        print(''.join([test["id"], ' -> ', test["fileName"], ' -> ', test["finding"]]))

    if create:
        construct_related_base_directory(train_set, train_covid_19_folder, "COVID-19")
        construct_related_base_directory(train_set, train_non_covid_19_folder, "non-COVID-19")
        construct_related_base_directory(test_set, test_covid_19_folder, "COVID-19")
        construct_related_base_directory(test_set, test_non_covid_19_folder, "non-COVID-19")


def elect_from_larger_dataset(small, large, dataset):
    elected = []
    rand = []
    rand_range = len(small)

    for _ in range(rand_range):
        r = randrange(rand_range)
        while r in rand:
            r = randrange(rand_range)
        rand.append(r)
        elected.append(large[r])

    refuse = list(set(large) - set(elected))

    dataset = remove_refuse_info_list_from_list(refuse, "id", dataset)

    return elected, dataset


def remove_refuse_info_list_from_list(refuse, key, target):
    temp_list = []
    for data in target:
        if data[key] not in refuse:
            temp_list.append(data)

    target.clear()
    target.extend(temp_list)
    temp_list.clear()

    return target


def construct_related_base_directory(dataset, folder, sub_folder):
    for data in dataset:
        if data["finding"] == sub_folder:
            source = SOURCE_DIR + data["fileName"]
            destination = folder + data["fileName"]
            copyfile(source, destination)


if __name__ == '__main__':
    # construct_dataset(unique=False, balanced=False, reset=True, create=True)
    dataset_investigate()
