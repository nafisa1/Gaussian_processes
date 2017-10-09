import numpy as np
import utils
import xlrd

def get_data(filename, id_name, smiles_name, output_name, descriptor_names, output_type='pic50', upper_threshold=None):
    workbook = xlrd.open_workbook(filename, on_demand=True)
    sheet = workbook.sheet_by_index(0)
    
    properties = sheet.row_values(0)
    reordered_descriptor_names = []
    descriptors = []
    for i,item in enumerate(properties):
        if item == id_name:
            names = sheet.col_values(i)[1:]
        elif item == output_name:
            output = sheet.col_values(i)[1:]
        elif item == smiles_name:
            smiles = sheet.col_values(i)[1:]
        elif str(item) in descriptor_names:
            print item, i
            reordered_descriptor_names.append(item)
            descriptors.append(sheet.col_values(i)[1:])
    descriptors = np.asarray(descriptors).T
    print len(names),"compounds initially"

    new_names = []
    new_output = []
    new_smiles = []
    new_desc = []

    for i, name in enumerate(names):
        if name != '' and output[i] != '' and output[i] < upper_threshold:
            new_names.append(name[4:])
            new_output.append(output[i])
            new_desc.append(list(descriptors[i]))
            new_smiles.append(str(smiles[i]).split()[0])
    print len(new_names),"compounds left after removing missing compounds and inactive compounds"
    
    zipped = zip(new_names, new_output, new_smiles, new_desc)
    zipped.sort()
    new_names, new_output, new_smiles, new_desc = zip(*zipped)
    new_desc = np.asarray(new_desc)
    print 'Compounds have been sorted by EVOTEC ID'

    if output_type == 'pic50':
        pic50s = utils.pIC50(new_output, -9)

    filtered_s, filtered_pic50, filtered_d = utils.enantiomers(new_smiles, pic50s, new_desc)
    return filtered_s, filtered_pic50, filtered_d
