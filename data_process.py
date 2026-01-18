import os
import numpy as np
import torch
import pandas as pd
import csv
import xml.etree.ElementTree as ET
import shutil


def dict_to_xml(data, root_name="root"):
    def build(parent, data):
        if isinstance(data, dict):
            for key, value in data.items():
                child = ET.SubElement(parent, key)
                build(child, value)
        elif isinstance(data, list):
            for item in data:
                item_elem = ET.SubElement(parent, "item")
                build(item_elem, item)
        else:
            parent.text = str(data)

    root = ET.Element(root_name)
    build(root, data)
    return ET.ElementTree(root)

# -----BSE形貌参数模板-----
BSE_mor_template = {'Pixel_Scale_nm_per_px': None,
                    'BSE_Ag_Coverage': None,
                    'BSE_Porosity': None,
                    'BSE_Effective_LCC_Ratio': None,
                    'BSE_Connectivity': None,
                    'BSE_Ag_Mean_Width_um': None,
                    'BSE_Ag_Bottleneck_Width_um': None,
                    'BSE_Ag_Entropy': None}

# -----SE形貌参数模板-----
SE_mor_template = {'Pixel_Scale_nm_per_px': None,
                   'SE_Porosity': None,
                   'SE_Network_Density_um': None,
                   'SE_Norm_Euler_Number': None,
                   'SE_Mean_Neck_Width_um': None,
                   'SE_Min_Bottleneck_Width_um': None,
                   'SE_Texture_Contrast': None,
                   'SE_Texture_Homogeneity': None}

# -----非截面性能参数模板-----
perf_nor_template = {'Surf_NeckWidth_um': None,
                     'Surf_NetDensity_um': None,
                     'Surf_Porosity': None,
                     'Surf_Tex_RMS_Roughness': None,
                     'Surf_Tex_Contrast': None,
                     'Surf_Tex_Homogeneity': None,
                     'Surf_Tex_Correlation': None,
                     'Surf_Alignment': None}

# -----截面参数模板-----
perf_cro_template = {'CS_Tex_RMS_Roughness': None,
                     'CS_Tex_Contrast': None,
                     'CS_Tex_Homogeneity': None,
                     'CS_Tex_Correlation': None,
                     'CS_Alignment': None,
                     'CS_Horizontal': None}

# -----路径相关信息-----
dataset_path = r'E:\PythonProject\AI4Materials\dataset'
mor_csv_path = os.path.join(dataset_path, 'Universal_Quantification_Results.csv')
main_perf_csv_path = os.path.join(dataset_path, '电导率汇总（主任务）.csv')
aux_perf_csv_path = os.path.join(dataset_path, '特征提取（辅助任务）.csv')
img_folder = os.path.join(dataset_path, 'dataset_8patches_512')

# -----创建数据集文件夹-----
materials_names = []
target_dataset_path = r'E:\PythonProject\AI4Materials\new_dataset'
main_perf_csv = pd.read_csv(main_perf_csv_path)
for i in range(main_perf_csv.shape[0]):
    materials_names.append(main_perf_csv.iloc[i, 0])

for materials_name in materials_names:
    if not os.path.exists(os.path.join(target_dataset_path, materials_name)):
        os.mkdir(os.path.join(target_dataset_path, materials_name))
        os.mkdir(os.path.join(target_dataset_path, materials_name, 'images'))
        os.mkdir(os.path.join(target_dataset_path, materials_name, 'morphological_parameters'))
        os.mkdir(os.path.join(target_dataset_path, materials_name, 'performance_parameters'))
        os.mkdir(os.path.join(target_dataset_path, materials_name, 'images', 'SE'))
        os.mkdir(os.path.join(target_dataset_path, materials_name, 'images', 'BSE'))
        os.mkdir(os.path.join(target_dataset_path, materials_name, 'images', 'CS'))
        os.mkdir(os.path.join(target_dataset_path, materials_name, 'morphological_parameters', 'SE'))
        os.mkdir(os.path.join(target_dataset_path, materials_name, 'morphological_parameters', 'BSE'))

# -----读取电导率参数相关数据并写为.xml后缀文件-----
main_perf_csv = pd.read_csv(main_perf_csv_path)
print('电导率参数规模: ', main_perf_csv.shape)
for i in range(main_perf_csv.shape[0]):
    main_perf_dict = {}
    xml_path = os.path.join(target_dataset_path, main_perf_csv.iloc[i, 0])
    main_perf_dict['Conductivity'] = main_perf_csv.iloc[i, 1]
    tree = dict_to_xml(main_perf_dict, root_name="annotation")
    ET.indent(tree, space="  ", level=0)
    tree.write(os.path.join(xml_path, '%s.xml'%(main_perf_csv.iloc[i, 0].split('.')[0])), encoding="utf-8", xml_declaration=False)
    print('已处理样件: %s' %(main_perf_csv.iloc[i, 0]))


# # -----读取形貌参数相关数据并写为.xml后缀文件-----
# mor_csv = pd.read_csv(mor_csv_path)
# print('形貌参数规模: ', mor_csv.shape)
# for i in range(mor_csv.shape[0]):
#     mor_dict = {}
#     for materials_name in materials_names:
#         str_len = len(materials_name)
#         # -----处理形貌参数-----
#         if mor_csv.iloc[i, 0][0:str_len] == materials_name and mor_csv.iloc[i, 0][str_len] == '-':
#             img_name = mor_csv.iloc[i, 0]
#             if mor_csv.iloc[i, 1] == 'BSE':
#                 mor_dict['Image_File'] = mor_csv.iloc[i, 0]
#                 mor_dict['Pixel_Scale_nm_per_px'] = mor_csv.iloc[i, 2]
#                 mor_dict['BSE_Ag_Coverage'] = mor_csv.iloc[i, 10]
#                 mor_dict['BSE_Porosity'] = mor_csv.iloc[i, 11]
#                 mor_dict['BSE_Effective_LCC_Ratio'] = mor_csv.iloc[i, 12]
#                 mor_dict['BSE_Connectivity'] = mor_csv.iloc[i, 13]
#                 mor_dict['BSE_Ag_Mean_Width_um'] = mor_csv.iloc[i, 14]
#                 mor_dict['BSE_Ag_Bottleneck_Width_um'] = mor_csv.iloc[i, 15]
#                 mor_dict['BSE_Ag_Entropy'] = mor_csv.iloc[i, 16]
#                 img_path = os.path.join(target_dataset_path, materials_name, 'images', 'BSE')
#                 xml_path = os.path.join(target_dataset_path, materials_name, 'morphological_parameters', 'BSE')
#             else:
#                 mor_dict['Image_File'] = mor_csv.iloc[i, 0]
#                 mor_dict['Pixel_Scale_nm_per_px'] = mor_csv.iloc[i, 2]
#                 mor_dict['SE_Porosity'] = mor_csv.iloc[i, 3]
#                 mor_dict['SE_Network_Density_um'] = mor_csv.iloc[i, 4]
#                 mor_dict['SE_Norm_Euler_Number'] = mor_csv.iloc[i, 5]
#                 mor_dict['SE_Mean_Neck_Width_um'] = mor_csv.iloc[i, 6]
#                 mor_dict['SE_Min_Bottleneck_Width_um'] = mor_csv.iloc[i, 7]
#                 mor_dict['SE_Texture_Contrast'] = mor_csv.iloc[i, 8]
#                 mor_dict['SE_Texture_Homogeneity'] = mor_csv.iloc[i, 9]
#                 img_path = os.path.join(target_dataset_path, materials_name, 'images', 'SE')
#                 xml_path = os.path.join(target_dataset_path, materials_name, 'morphological_parameters', 'SE')
#
#             tree = dict_to_xml(mor_dict, root_name="annotation")
#             ET.indent(tree, space="  ", level=0)
#             tree.write(os.path.join(xml_path, '%s.xml'%(img_name.split('.')[0])), encoding="utf-8", xml_declaration=False)
#             shutil.copy(os.path.join(img_folder, img_name), os.path.join(img_path, img_name))
#             print('已处理图像: %s' %(img_name))
#             break
#
# # -----读取性能参数相关数据并写为.xml后缀文件
# aux_perf_csv = pd.read_csv(aux_perf_csv_path)
# print('性能参数规模: ', aux_perf_csv.shape)
# for i in range(aux_perf_csv.shape[0]):
#     aux_perf_dict = {}
#     for materials_name in materials_names:
#         str_len = len(materials_name)
#         if aux_perf_csv.iloc[i, 0][2:] == materials_name:
#             if aux_perf_csv.iloc[i, 2] == 'Surface':
#                 bse_img_name = aux_perf_csv.iloc[i, 3]
#                 se_img_name = aux_perf_csv.iloc[i, 4]
#                 aux_perf_dict['SE_Image_File'] = aux_perf_csv.iloc[i, 3]
#                 aux_perf_dict['BSE_Image_File'] = aux_perf_csv.iloc[i, 4]
#                 aux_perf_dict['Surf_NeckWidth_um'] = aux_perf_csv.iloc[i, 6]
#                 aux_perf_dict['Surf_NetDensity_um'] = aux_perf_csv.iloc[i, 7]
#                 aux_perf_dict['Surf_Porosity'] = aux_perf_csv.iloc[i, 8]
#                 aux_perf_dict['Surf_Tex_RMS_Roughness'] = aux_perf_csv.iloc[i, 9]
#                 aux_perf_dict['Surf_Tex_Contrast'] = aux_perf_csv.iloc[i, 10]
#                 aux_perf_dict['Surf_Tex_Homogeneity'] = aux_perf_csv.iloc[i, 11]
#                 aux_perf_dict['Surf_Tex_Correlation'] = aux_perf_csv.iloc[i, 12]
#                 aux_perf_dict['Surf_Alignment'] = aux_perf_csv.iloc[i, 13]
#                 xml_path = os.path.join(target_dataset_path, materials_name, 'performance_parameters')
#                 xml_name = aux_perf_csv.iloc[i, 1]
#                 tree = dict_to_xml(aux_perf_dict, root_name="annotation")
#                 ET.indent(tree, space="  ", level=0)
#                 tree.write(os.path.join(xml_path, '%s.xml' % (xml_name)), encoding="utf-8", xml_declaration=False)
#                 print('已处理图像: %s' % (xml_name))
#                 break
#             else:
#                 cs_img_name = aux_perf_csv.iloc[i, 3]
#                 aux_perf_dict['CS_Image_File'] = aux_perf_csv.iloc[i, 3]
#                 aux_perf_dict['CS_Tex_RMS_Roughness'] = aux_perf_csv.iloc[i, 14]
#                 aux_perf_dict['CS_Tex_Contrast'] = aux_perf_csv.iloc[i, 15]
#                 aux_perf_dict['CS_Tex_Homogeneity'] = aux_perf_csv.iloc[i, 16]
#                 aux_perf_dict['CS_Tex_Correlation'] = aux_perf_csv.iloc[i, 17]
#                 aux_perf_dict['CS_Alignment'] = aux_perf_csv.iloc[i, 18]
#                 aux_perf_dict['CS_Horizontal'] = aux_perf_csv.iloc[i, 19]
#                 xml_path = os.path.join(target_dataset_path, materials_name, 'performance_parameters')
#                 xml_name = aux_perf_csv.iloc[i, 1]
#                 tree = dict_to_xml(aux_perf_dict, root_name="annotation")
#                 ET.indent(tree, space="  ", level=0)
#                 tree.write(os.path.join(xml_path, '%s.xml' % (xml_name)), encoding="utf-8", xml_declaration=False)
#                 shutil.copy(os.path.join(img_folder, cs_img_name), os.path.join(target_dataset_path, materials_name, 'images', 'CS', cs_img_name))
#                 print('已处理图像: %s' % (xml_name))
#                 break

