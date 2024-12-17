import os
import json
import re
import subprocess
import nibabel as nib
import shutil

# BIDS Metadata Manager

# Overview:
# This script is designed to validate, modify, and preprocess BIDS-compliant neuroimaging datasets.
# It focuses on updating metadata files (`.json`) and ensuring NIfTI files (`.nii.gz`) meet BIDS
# standards. This is particularly useful for datasets containing functional MRI (fMRI), anatomical
# data, and field mapping (fmap) data.
#
# The script performs the following key tasks:
# 1. **File Renaming**: Standardizes functional file names to match the expected BIDS convention.
# 2. **Metadata Modification**:
#    - Adds or updates metadata fields such as `PhaseEncodingDirection`, `SliceTiming`, and
#    `TotalReadoutTime` for functional data.
#    - Ensures field mapping files contain proper `IntendedFor` references and `EchoTime1`/`EchoTime2`.
# 3. **fMRI Slicing**: If functional MRI data contains 49 slices, it slices the data down to 38
# slices using `fslroi` (FSL command-line tool).
# 4. **Validation Preparation**: Ensures the dataset is ready for BIDS validation.


class JsonManipulation:
    """Functions for manipulating json files"""
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.data = None

    def load_json(self):
        with open(self.input_path, 'r') as f:
            self.data = json.load(f)

    def save_json(self):
        with open(self.output_path, 'w') as f:
            json.dump(self.data, f, indent=4)


class BidsMetadataModifier:
    """
    BIDSMetadataModifier

    This class provides a collection of methods to modify BIDS (Brain Imaging Data Structure)
    metadata for various neuroimaging data types, including functional MRI (fMRI) and field mapping (fmap) data.

    The methods in this class ensure consistency with common BIDS specifications by adding or updating
    necessary metadata fields such as `PhaseEncodingDirection`, `SliceTiming`, `TotalReadoutTime`, and more.
    It is designed to facilitate preprocessing and organization of neuroimaging datasets, especially when
    metadata is incomplete or missing.

    Methods:
    - `modify_phase`: Updates or sets the `PhaseEncodingDirection` field.
    - `modify_func_slice_timing`: Ensures `SliceTiming` exists and applies default timing for 38-slice data.
    - `modify_func_total_readout_time`: Verifies and sets the `TotalReadoutTime` field.
    - `modify_fmap_intendfor`: Adds or updates the `IntendedFor` field for field map files.
    - `modify_fmap_pd`: Ensures `EchoTime1` and `EchoTime2` are properly set for field mapping data.

    Usage:
    Create an instance of the class and call the appropriate method, passing in the metadata dictionary
    and any required parameters such as the subject ID.
    """
    def __init__(self):
        pass

    # For all func
    def modify_phase(self, data):  # noqa
        if "PhaseEncodingAxis" in data:
            key = data["PhaseEncodingAxis"]
            data["PhaseEncodingDirection"] = key
            del data["PhaseEncodingAxis"]
        elif "PhaseEncodingAxis" not in data:
            data["PhaseEncodingDirection"] = "j"
        return data

    # For 38 slice func
    def modify_func_slice_timing(self, data, sub): # noqa
        if "SliceTiming" in data:
            print('SliceTiming Exists in {}'.format(sub))
        else:
            data["SliceTiming"] = [
                0,
                0.065,
                0.1325,
                0.1975,
                0.2625,
                0.3275,
                0.3925,
                0.4575,
                0.5225,
                0.5875,
                0.6525,
                0.72,
                0.785,
                0.85,
                0.915,
                0.98,
                1.045,
                1.11,
                1.175,
                1.24,
                1.305,
                1.3725,
                1.4375,
                1.5025,
                1.5675,
                1.6325,
                1.6975,
                1.7625,
                1.8275,
                1.8925,
                1.9575,
                2.025,
                2.09,
                2.155,
                2.22,
                2.285,
                2.35,
                2.415
            ]
        return data

    def modify_func_total_readout_time(self, data, sub): # noqa
        if "TotalReadoutTime" in data and data["TotalReadoutTime"] == 0.0377995:
            print("TotalReadoutTime already exists in {}".format(sub))
        elif "TotalReadoutTime" not in data:
            data["TotalReadoutTime"] = 0.0377995
        return data

    # For fmap
    def modify_fmap_intendfor(self, data, sub): # noqa
        if "IntendedFor" not in data:
            data["IntendedFor"] = "func/{}_task-rest_bold.nii.gz".format(sub)
        elif "IntendedFor" in data:
            data["IntendedFor"] = 'func/{}_task-rest_bold.nii.gz'.format(sub)
        return data

    def modify_fmap_pd(self, data): # noqa
        if "EchoTime1" not in data:
            data["EchoTime1"] = 0.00492
        elif "EchoTime1" in data:
            data["EchoTime1"] = 0.00492

        if "EchoTime2" not in data:
            data["EchoTime2"] = 0.00738
        elif "EchoTime2" in data:
            data["EchoTime2"] = 0.00738
        return data


class WorkflowManager(JsonManipulation, BidsMetadataModifier):
    """
    A class to manage and process BIDS-compliant neuroimaging data directories and metadata.

    This class includes methods to validate and modify metadata in JSON files, ensuring consistency with
    BIDS specifications for functional (func), anatomical (anat), and field mapping (fmap) data. It also
    handles file structure validation and updates specific fields as needed.

    Methods:
    - meet_the_pattern: Validates if directory names match the specified pattern.
    - get_sub_seq: Retrieves all subject directories that meet the specified pattern.
    - get_anatfunc_seq: Filters subject directories containing anat and func folders.
    - get_anatfuncfmap_seq: Filters subject directories containing anat, func, and fmap folders.
    - manipulate_phase_encoding: Replaces "PhaseEncodingAxis" with "PhaseEncodingDirection" in JSON
    metadata for all relevant files.
    - manipulate_anatfuncfmap: Enhances metadata for func and fmap files, adding slice timing, total
    readout time, and echo time values.
    - manipulate_fmap_intendfor: Updates the "IntendedFor" field in fmap JSON files to link them to the
    correct functional data.
    """
    def __init__(self, source_path: str, prefix: str, digit: int, suffix: str,):
        JsonManipulation.__init__(self, source_path, source_path)  # input and output path are same here.
        BidsMetadataModifier.__init__(self)

        self.source_path = source_path
        self.prefix = prefix
        self.digit = digit
        self.suffix = suffix

        self.sub_seq = self.get_sub_seq()
        self.anatfunc_seq = self.get_anatfunc_seq()
        self.anatfuncfmap_seq = self.get_anatfuncfmap_seq()

    def meet_the_pattern(self, name: str) -> bool:
        """To determine if a pattern is meet or not."""
        pattern = rf'{re.escape(self.prefix)}(\d{{{self.digit}}}){re.escape(self.suffix)}'
        match = re.search(pattern, name)
        if match:
            return True
        else:
            return False

    def get_sub_seq(self) -> list:
        """To get a list of all subject directory name"""
        sub_seq = [d for d in os.listdir(self.source_path) if self.meet_the_pattern(d)]
        return sub_seq

    def get_anatfunc_seq(self) -> list:
        """To get a list of all subject directory which contain anat and func only"""
        anatfunc_seq = []
        for sub in self.sub_seq:
            content = [d for d in os.listdir(os.path.join(self.source_path, sub))]
            if 'fmap' not in content:
                anatfunc_seq.append(sub)
        return anatfunc_seq

    def get_anatfuncfmap_seq(self) -> list:
        """To get a list of all subject directory which contain anat, func and fmap"""
        anatfuncfmap_seq = []
        for sub in self.sub_seq:
            if not os.path.isdir(os.path.join(self.source_path, sub)):
                continue
            else:
                content = [d for d in os.listdir(os.path.join(self.source_path, sub))]
                if 'fmap' in content:
                    anatfuncfmap_seq.append(sub)
        return anatfuncfmap_seq

    def change_func_files_name(self, name: str) -> None:
        """
        To make sure functions files in format as f'{sub}{name}.nii.gz'
        :param name: suffix of functional file
        :return: None (files will be changed)
        """
        # Loop for every subject
        for sub in self.sub_seq:
            if os.path.isdir(os.path.join(self.source_path, sub)):
                subfolder_path = os.path.join(self.source_path, sub)
            else:
                continue

            func_files_lst = [file for file in os.listdir(os.path.join(subfolder_path, 'func'))
                              if not file.startswith('.')]

            #
            # print(func_files_lst)
            #

            for file in func_files_lst:
                if file.endswith('.nii.gz'):
                    try: # noqa
                        os.rename(os.path.join(subfolder_path, 'func', file),
                                  os.path.join(subfolder_path, 'func', f'{sub}{name}.nii.gz'))
                    except FileNotFoundError:
                        print(f"Error: {file} not found.")
                    except Exception as e:
                        print(f"Error renaming {file}: {e}")
                elif file.endswith('.json'):
                    try: # noqa
                        os.rename(os.path.join(subfolder_path, 'func', file),
                                  os.path.join(subfolder_path, 'func', f'{sub}{name}.json'))
                    except FileNotFoundError:
                        print(f"Error: {file} not found.")
                    except Exception as e:
                        print(f"Error renaming {file}: {e}")

    @staticmethod
    def slice_func_49to38(file_path: str, subject: str) -> None:
        """
        Cut the slice number from 49 to 38 by using FSLroi.
        """
        # Make sure fslroi exists in the environment
        if not shutil.which("fslroi"):
            raise EnvironmentError("Error: fslroi command not found. Ensure FSL is installed and added to PATH.")

        # Construct the fslroi command
        command = [
            "fslroi",
            file_path,
            file_path,
            "0", "-1", "0", "-1", "0", "38", "0", "-1"
        ]

        # Print status
        print(f"Processing {subject}/func...")

        # Run the command
        try:
            subprocess.run(command, check=True)
            print(f"{subject}/func complete")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {subject}/func: {e}")

    @staticmethod
    def count_func_slice(file_path, subject: str) -> [int, None]:
        try:
            # Load the NIfTI file
            img = nib.load(file_path)
            # Return the number of slices in the Z dimension
            return img.shape[2] # noqa
        except FileNotFoundError:
            print(f"Error: File not found for subject {subject}: {file_path}")
            return None
        except IndexError:
            print(f"Error: File {file_path} for subject {subject} has invalid dimensions.")
            return None
        except Exception as e:
            print(f"Unexpected error for subject {subject}: {e}")
            return None

    # 8-1
    def manipulate_phase_encoding(self) -> None:
        """
        for all data, we need to change ["PhaseEncodingAxis": "j"], into ["PhaseEncodingDirection": "j"].
        Here subfolder stand for "sub-002" likely folder, bids_type stand for anat, func or fmap folder
        :return: None(File info will be changed)
        """
        # Loop for every subject
        for sub in self.sub_seq:
            # Here subfolder stand for "sub-002" likely folder, bids_type stand for anat, func or fmap.
            if os.path.isdir(os.path.join(self.source_path, sub)):
                subfolder_path = os.path.join(self.source_path, sub)
            else:
                continue
            bids_type = [t_folder for t_folder in os.listdir(subfolder_path)
                         if os.path.isdir(os.path.join(subfolder_path, t_folder))]

            for t in bids_type:
                t_path = os.path.join(subfolder_path, t)
                file_lst = [file for file in os.listdir(t_path) if not file.startswith('.')]

                for file in file_lst:
                    if file.endswith('.json'):
                        json_manipulator = JsonManipulation(input_path=os.path.join(t_path, file),
                                                            output_path=os.path.join(t_path, file))
                        json_manipulator.load_json()
                        json_manipulator.data = self.modify_phase(json_manipulator.data)
                        json_manipulator.save_json()

            print('Manipulation Complete for {}.'.format(sub))

    # 8-2
    def manipulate_anatfuncfmap(self) -> None:
        """
        for data after 2023.07: the func part: we need to add slice timing and totalreadouttime;
        (we can choose the same slice timing if there is a lack for all of them, and the totalreadouttime is the same),
        the fmap part: we need to add two echo time and change the intended for filed into the content below.
        :return: None(File info will be changed)
        """
        for sub in self.anatfuncfmap_seq:

            if os.path.isdir(os.path.join(self.source_path, sub)):
                subfolder_path = os.path.join(self.source_path, sub)
            else:
                continue
            bids_type = [t_folder for t_folder in os.listdir(subfolder_path) if not t_folder.startswith('.')]

            for t in bids_type:
                t_path = os.path.join(subfolder_path, t)

                # func data manipulation
                if t == 'func':
                    file_lst = [file for file in os.listdir(t_path) if not file.startswith('.')]
                    for file in file_lst:
                        if file.endswith('.json'):
                            json_manipulator = JsonManipulation(input_path=os.path.join(t_path, file),
                                                                output_path=os.path.join(t_path, file))
                            json_manipulator.load_json()
                            json_manipulator.data = self.modify_func_slice_timing(json_manipulator.data, sub)
                            json_manipulator.data = self.modify_func_total_readout_time(json_manipulator.data, sub)
                            json_manipulator.save_json()

                # fmap data manipulation
                if t == 'fmap':
                    file_lst = [file for file in os.listdir(t_path) if not file.startswith('.')]
                    for file in file_lst:
                        if 'phasediff' in file and file.endswith('.json'):
                            json_manipulator = JsonManipulation(input_path=os.path.join(t_path, file),  # noqa
                                                                output_path=os.path.join(t_path, file))  # noqa
                            json_manipulator.load_json()
                            json_manipulator.data = self.modify_fmap_intendfor(json_manipulator.data, sub)
                            json_manipulator.data = self.modify_fmap_pd(json_manipulator.data)
                            json_manipulator.save_json()

                        elif 'phasediff' not in file and file.endswith('.json'):
                            json_manipulator = JsonManipulation(input_path=os.path.join(t_path, file),
                                                                output_path=os.path.join(t_path, file))
                            json_manipulator.load_json()
                            json_manipulator.data = self.modify_fmap_intendfor(json_manipulator.data, sub)
                            json_manipulator.save_json()

                    #     if 'phasediff' in file and file.endswith('.json'):
                    #         json_manipulator = JsonManipulation(input_path=os.path.join(t_path, file), # noqa
                    #                                             output_path=os.path.join(t_path, file)) # noqa
                    #         json_manipulator.load_json()
                    #         json_manipulator.data = self.modify_fmap_intendfor(json_manipulator.data, sub)
                    #         json_manipulator.data = self.modify_fmap_pd(json_manipulator.data)
                    #         json_manipulator.save_json()
                    #         break
                    # else:
                    #     print("No phasediff file exists in {}".format(sub))

            print('Manipulation Complete for {}.'.format(sub))

    # Selective alternative
    def manipulate_fmap_intendfor(self) -> None:
        """
        In particular, only change the INTENDED_FOR info in fmap json files.
        :return: None(File info will be changed)
        """
        # for sub in self.anatfuncfmap_seq:
        #     fmapfolder_path = os.path.join(self.source_path, sub, 'fmap')
        #     if not os.path.isdir(fmapfolder_path):
        #         continue
        #     else:
        #         file_lst = [file for file in os.listdir(fmapfolder_path) if not file.startswith('.')]
        #         for file in file_lst:
        #             if 'phasediff' in file and file.endswith('.json'):
        #                 json_manipulator = JsonManipulation(input_path=os.path.join(fmapfolder_path, file), # noqa
        #                                                     output_path=os.path.join(fmapfolder_path, file)) # noqa
        #                 json_manipulator.load_json()
        #                 json_manipulator.data = self.modify_fmap_intendfor(json_manipulator.data, sub)
        #                 json_manipulator.data = self.modify_fmap_pd(json_manipulator.data)
        #                 json_manipulator.save_json()
        #
        #             elif 'phasediff' not in file and file.endswith('.json'):
        #                 json_manipulator = JsonManipulation(input_path=os.path.join(fmapfolder_path, file),
        #                                                     output_path=os.path.join(fmapfolder_path, file))
        #                 json_manipulator.load_json()
        #                 json_manipulator.data = self.modify_fmap_intendfor(json_manipulator.data, sub)
        #                 json_manipulator.save_json()
        #
        #     print('Manipulation Complete for {}.'.format(sub))
        pass

    # 8-3
    def execute_func_slicing(self) -> None:
        """
        for the data after 2023.07, if we chose  "SequenceName": "*epfid2d1_64" for function MRI data.
        We will found it includes two different slice number, 38 and 49. But for preprocessing, we can
        only choose 38 slice number. So we may need to cut the slice number from 49 to 38 by using FSLroi.
        :return: None(File info will be changed)
        """
        # Change functional files name
        self.change_func_files_name('_task-rest_bold')

        # Loop for every subject
        for sub in self.sub_seq:
            if os.path.isdir(os.path.join(self.source_path, sub)):
                subfolder_path = os.path.join(self.source_path, sub)
            else:
                continue

            # Get path of function nifti file
            func_nifti_path = os.path.join(subfolder_path, 'func', f'{sub}_task-rest_bold.nii.gz')

            # Slicing from 49 to 38 in Z-dimension
            slice_count = self.count_func_slice(func_nifti_path, sub)
            if not slice_count:
                print(f'Error: Failed to read {sub}_task-rest_bold.nii.gz in {func_nifti_path}')
            elif slice_count == 38:
                pass
            elif slice_count == 49:
                self.slice_func_49to38(func_nifti_path, sub)


# ---------------------------Parameters-------------------------
source_dir = "/Users/yanyuqi/Desktop/Proj/data_Bids"
sub_prefix = 'sub-hc'
sub_digit = 3
sub_suffix = ''
# ----------------------Parameters---End------------------------


# ---------------------------Execution-------------------------------------------
workflow_manager = WorkflowManager(source_dir, sub_prefix, sub_digit, sub_suffix)

# 8.3 (Need to execute 8.3 before others)
workflow_manager.execute_func_slicing()

# 8.1
workflow_manager.manipulate_phase_encoding()

# 8.2
workflow_manager.manipulate_anatfuncfmap()
