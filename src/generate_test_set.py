import os
import random
import shutil

def generate_test_set():

    '''
    This function is used to generate a test set by randomly selecting a number of files
    from the train directory and moving them to the test directory.
    :return: None
    '''

    # Path to the directory containing your files
    source_directory = "../assets/train/"

    # Path to the directory where you want to move the selected files
    destination_directory = "../assets/test/"

    # Number of files you want to select randomly
    num_files_to_select = 508 # 10% of the total number of files

    # List all files in the source directory
    all_files = os.listdir(source_directory)

    # Randomly select 508 files
    selected_files = random.sample(all_files, num_files_to_select)

    # Move selected files to the destination directory
    for file_name in selected_files:
        source_file_path = os.path.join(source_directory, file_name)
        destination_file_path = os.path.join(destination_directory, file_name)
        shutil.move(source_file_path, destination_file_path)

    print("Selected files moved successfully!")

def generate_test_mask():
    '''
    This function is used to move the corresponding masks of the selected test files from the train_masks
    :return:  None
    '''
    path = '../assets/test/'
    files = get_file_names_in_folder(path)
    source_directory = "../assets/train_masks/"
    destination_directory = "../assets/test_masks/"
    # print(files)

    for file_name in files:
        print(file_name)
        source_file_path = os.path.join(source_directory, file_name)
        destination_file_path = os.path.join(destination_directory, file_name)
        shutil.move(source_file_path, destination_file_path)

    print("Selected files moved successfully!")


def generate_validation_set():
    '''
    This function is used to generate a validation set by randomly selecting a number of files
    from the train directory and moving them to the validation directory.
    :return:  None
    '''
    # Path to the directory containing your files
    source_directory = "../assets/train/"

    # Path to the directory where you want to move the selected files
    destination_directory = "../assets/validation/"

    # Number of files you want to select randomly
    num_files_to_select = 1374 # 30% of remaining files (remaining files = total - 0.1*total)

    # List all files in the source directory
    all_files = os.listdir(source_directory)

    # Randomly select 1374 files
    selected_files = random.sample(all_files, num_files_to_select)

    # Move selected files to the destination directory
    for file_name in selected_files:
        source_file_path = os.path.join(source_directory, file_name)
        destination_file_path = os.path.join(destination_directory, file_name)
        shutil.move(source_file_path, destination_file_path)

    print("Selected files moved successfully!")

def get_validation_mask():
    '''
    This function is used to move the corresponding masks of the selected validation files from the train_masks
    :return:
    '''
    path = '../assets/validation/'
    files = get_file_names_in_folder(path)
    source_directory = "../assets/train_masks/"
    destination_directory = "../assets/validation_masks/"
    # print(files)

    for file_name in files:
        print(file_name)
        source_file_path = os.path.join(source_directory, file_name)
        destination_file_path = os.path.join(destination_directory, file_name)
        shutil.move(source_file_path, destination_file_path)

    print("Selected files moved successfully!")

def count_files_in_directory(directory):
    return len(os.listdir(directory))


def get_file_names_in_folder(folder_path):
    files = []

    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            # files.append(file_name)
            name, extension = os.path.splitext(file_name)
            # print(name)
            # print(extension)
            if extension.lower() == '.jpg':
                files.append(name+'_mask.gif')

    return files




# if __name__ == "__main__":
#     generate_test_set()
#     generate_test_mask()
#     generate_validation_set()
#     get_validation_mask()
#     print("Number of files in train directory:", count_files_in_directory("../assets/train/"))
#     print("Number of files in train masks directory:", count_files_in_directory("../assets/train_masks/"))
#     print("Number of files in test directory:", count_files_in_directory("../assets/test_masks/"))
#     print("Number of files in validation directory:", count_files_in_directory("../assets/validation_masks/"))

