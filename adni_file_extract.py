import os
import pickle


def iterate_files(folder):
    """Iterates through a folder with folders in it, used for data.

    Parameters:
        folder (str): directory of folder

    Returns:
        file_dict (dict): {folder name: files containing the word "adni" in the folder}
    """
    file_dict = {}
    for folder_name in os.listdir(folder):
        if folder_name.startswith("adni"):
            folder_path = os.path.join(folder, folder_name)
            file_names = []
            for filename in os.listdir(folder_path):
                if not filename.startswith("."):
                    file_path = os.path.join(folder_path, filename)
                    if os.path.isfile(file_path):
                        file_names.append(filename)
            file_dict[folder_name] = sorted(file_names)
    with open("file_dict.pkl", "wb") as f:
        pickle.dump(file_dict, f)
    return file_dict


def count_files(file_dict):
    """Counts the number of files in the file_dict.

    Parameters:
        file_dict (dict): dict of folders and file names.
    Returns:
        count(int): count
    """
    count = 0
    for value in file_dict.values():
        count += len(value)
    return count


def clean_bim(filename, output_filename):
    """
    Cleans a .bim file by removing SNP entries that are invalid.

    Parameters:
        filename (str): Path to the input .bim file.
        output_filename (str): Path to save the cleaned output file.

    Removes lines where the SNP ID does not start with "rs" or alleles are marked as "0".
    """

    with open(filename, 'r') as f, open(output_filename, 'w') as out_f:
        for line in f:
            splt = line.rstrip().split()
            if not splt[1].startswith("rs") or splt[4] == "0" or splt[5] == "0":
                continue
            out_f.write("\t".join(splt) + "\n")


def get_diagnosis(diagnosis_file):
    """
    Parses a diagnosis file and returns a dictionary mapping patient IDs to diagnosis labels.

    Parameters:
        diagnosis_file (str): Path to the diagnosis CSV file.

    Returns:
        dict: Dictionary mapping patient IDs to their diagnoses.
    """
    diagnosis_dict = {}
    with open(diagnosis_file) as f:
        for line in f:
            diagnosis = -1
            line = line.replace('"', "")
            if line.startswith("subject"):
                continue
            splt = line.strip().split(",")
            if splt[1] == "CN":
                diagnosis = 0
            elif splt[1] == "AD":
                diagnosis = 1
            diagnosis_dict[splt[0]] = diagnosis
    with open("diagnosis_dict.pkl", "wb") as f:
        pickle.dump(diagnosis_dict, f)
    return diagnosis_dict


def get_features(filename, diagnosis_dict, feature_dict, lookup_dict, snp_dict, snp_ref):
    """
    Extracts SNP features from a given .bed/.bim/.fam file set and updates the feature dictionary.

    Parameters:
        filename (str): The base filename used for loading .bed, .bim, and .fam files.
        diagnosis_dict (dict): Patient diagnoses mapped by ID.
        feature_dict (dict): Dictionary to populate with patient feature vectors.
        lookup_dict (dict): Maps patient IDs to sample indices.
        snp_dict (dict): Dictionary of valid SNPs.
        snp_ref (dict): Reference alleles for each SNP.

    Modifies:
        feature_dict (dict): Adds new patient feature entries.
    """
    subject_id = ""
    count = 0
    feature_list = []
    snp_list = []
    with open(filename) as f:
        for line in f:
            if count == 0:
                count += 1
                continue
            feature_val = -1
            splt = line.strip().split(",")
            if count == 1:
                count += 1
                subject_id = splt[1]
            else:
                allele_pair = (splt[9], splt[10])
                if allele_pair == ("A", "A"):
                    feature_val = 0
                elif allele_pair == ("A", "B") or allele_pair == ("B", "A"):
                    feature_val = 1
                elif allele_pair == ("B", "B"):
                    feature_val = 2
                if splt[4] not in snp_dict.keys() or splt[4] not in snp_ref.keys():
                    continue
                else:
                    feature_list.append(feature_val)
                    snp_list.append(splt[4])
        feature_dict[tuple(feature_list)] = diagnosis_dict[subject_id]
        lookup_dict[subject_id] = tuple(feature_list)
    return feature_dict, lookup_dict, snp_list


def get_feature_dict(folder, file_dict, diagnosis_dict, snp_dict, snp_ref):
    """
    Builds a complete feature dictionary by iterating through all valid files in the folder.

    Parameters:
        folder (str): Path to the folder containing SNP files.
        file_dict (dict): Dictionary mapping subfolders to files.
        diagnosis_dict (dict): Patient diagnoses.
        snp_dict (dict): Dictionary of valid SNPs.
        snp_ref (dict): Reference alleles for SNPs.

    Returns:
        dict: A feature dictionary mapping patient IDs to SNP feature vectors.
    """
    feature_dict = {}
    lookup_dict = {}
    ret_snp_list = []
    count = 0
    for key, value in file_dict.items():
        for subject_file in value:
            subject = subject_file[:-4]
            if diagnosis_dict[subject] == -1:
                continue
            feature_dict, lookup_dict, snp_list = get_features(folder + "/" + key + "/" + subject_file,
                                                               diagnosis_dict, feature_dict, lookup_dict, snp_dict,
                                                               snp_ref)
            if count == 0:
                ret_snp_list = snp_list
            count += 1
            # print(str(count) + " " + str(snp_list == ret_snp_list))
    with open("feature_dict.pkl", "wb") as f:
        pickle.dump(feature_dict, f)
    with open("lookup_dict.pkl", "wb") as f1:
        pickle.dump(lookup_dict, f1)
    with open("snp_list.pkl", "wb") as f2:
        pickle.dump(ret_snp_list, f2)
    return feature_dict, lookup_dict, ret_snp_list


# FILE_DICT = iterate_files("/Volumes/seagate")
# DIAGNOSIS_DICT = get_diagnosis("diagnosis.csv")
#
# with open("snp_ref.pkl", "rb") as F:
#     SNP_REF = pickle.load(F)
# with open("snp_dict.pkl", "rb") as F2:
#     SNP_DICT = pickle.load(F2)
#
# FEATURE_DICT, LOOKUP_DICT, SNP_LIST = get_feature_dict("/Volumes/seagate",
#                                                        FILE_DICT, DIAGNOSIS_DICT, SNP_DICT, SNP_REF)

with open("file_dict.pkl", "rb") as F:
    FILE_DICT = pickle.load(F)
with open("diagnosis_dict.pkl", "rb") as F1:
    DIAGNOSIS_DICT = pickle.load(F1)
with open("feature_dict.pkl", "rb") as F2:
    FEATURE_DICT = pickle.load(F2)
with open("lookup_dict.pkl", "rb") as F3:
    LOOKUP_DICT = pickle.load(F3)
with open("snp_list.pkl", "rb") as F4:
    SNP_LIST = pickle.load(F4)
with open("snp_dict.pkl", "rb") as F5:
    SNP_DICT = pickle.load(F5)

# FILE_DICT = iterate_files("/Volumes/seagate")
# DIAGNOSIS_DICT = get_diagnosis("diagnosis.csv")
# FEATURE_DICT, LOOKUP_DICT, SNP_LIST = get_feature_dict("/Volumes/seagate",
#                                              FILE_DICT, DIAGNOSIS_DICT, SNP_DICT)

print(len(FEATURE_DICT))
print(len(LOOKUP_DICT))
AD_COUNT = 0
CU_COUNT = 0
for KEY, VALUE in FEATURE_DICT.items():
    if VALUE == 0:
        CU_COUNT += 1
    if VALUE == 1:
        AD_COUNT += 1
print("AD Count: ", AD_COUNT)
print("CU Count: ", CU_COUNT)
