# Hacked up script for removing the license headers from the
import os

from tqdm import tqdm


def get_all_files_with_extension(ext, dir_path="."):
    possible_files = os.listdir(dir_path)
    files_with_given_ext = []

    for possible_file_name in possible_files:
        possible_file = os.path.join(dir_path, possible_file_name)

        if os.path.isfile(possible_file):
            # add the file only if it has the required extension
            if possible_file_name.endswith(ext):
                # this is one of the files
                files_with_given_ext.append(possible_file)
        else:
            files_with_given_ext.extend(
                get_all_files_with_extension(ext, possible_file)
            )

    return files_with_given_ext


def remove_license(files_list, top_lines=6):
    for file_ in tqdm(files_list):
        with open(file_, "r") as reader, open(file_ + "_.tmp", "w") as writer:

            # extract the first line:
            first_line = reader.readline()

            if (
                first_line.strip()
                != "# Copyright (c) 2019, NVIDIA Corporation. All rights reserved."
            ):

                continue

            for _ in range(top_lines - 1):
                next(reader)

            for line in reader:
                writer.write(line)
        os.remove(file_)
        os.rename(file_ + "_.tmp", file_)


print(" removing the Licences from the python files ... ")
remove_license(get_all_files_with_extension(".py"))
