import os

brain_region_list = ['left_ALM', 'left_BLA', 'left_ECT', 'left_Medulla', 'left_Midbrain', 'left_Striatum', 'left_Thalamus'] +\
                    ['right_ALM', 'right_BLA', 'right_ECT', 'right_Medulla', 'right_Midbrain', 'right_Striatum', 'right_Thalamus']

def map_brain_to_days_and_vice_versa(brain_region_list):
    mapping_brain_to_days = {}
    mapping_days_to_brain = {}

    for brain_region in brain_region_list:
        dir_path = f"../data/raw_data/neural_data_{brain_region}"
        entries = os.listdir(dir_path)
        extracted_contents = []
        for file_name in entries:
            # Verify file naming convention; adjust the indices as necessary.
            assert file_name.startswith('neural_data_')
            assert file_name.endswith(f'_{brain_region}.mat')
            # Remove 'neural_data_' from the start and f'_{brain_region}.mat' from the end.
            num = int(file_name[len('neural_data_') : - (len(brain_region) + 5)])
            extracted_contents.append(num)
        extracted_contents.sort()

        mapping_brain_to_days[brain_region] = extracted_contents
        for day in extracted_contents:
            if day not in mapping_days_to_brain.keys():
                mapping_days_to_brain[day] = [brain_region]
            else:
                mapping_days_to_brain[day].append(brain_region)

    return mapping_brain_to_days, mapping_days_to_brain