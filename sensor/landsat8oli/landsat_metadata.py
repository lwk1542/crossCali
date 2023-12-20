from xml.etree import ElementTree as ET
import datetime


def mtl(file: str):
    tree = ET.parse(file)
    root = tree.getroot()
    PROCESSING_RECORD = root.findall('IMAGE_ATTRIBUTES')[0]
    dto_string = PROCESSING_RECORD.find('DATE_ACQUIRED').text +"T"+PROCESSING_RECORD.find('SCENE_CENTER_TIME').text
    dto_string = dto_string.split('.')[0]+"."+dto_string.split('.')[1][:6]  #  strptime 函数在处理微秒时有长度限制，无法直接解析超过6位数的微秒部分
    datetime_obj = datetime.datetime.strptime(dto_string, "%Y-%m-%dT%H:%M:%S.%f")
    RESCALING = root.findall("LEVEL1_RADIOMETRIC_RESCALING")[0]
    gains = []
    offsets = []
    for i in range(7):
        gains.append(float(RESCALING.find("RADIANCE_MULT_BAND_"+str(i+1)).text))
        offsets.append(float(RESCALING.find("RADIANCE_ADD_BAND_" + str(i + 1)).text))
    PROJECTION_ATTRIBUTES = root.findall("PROJECTION_ATTRIBUTES")[0]
    rows = PROJECTION_ATTRIBUTES.find("REFLECTIVE_LINES").text
    columns = PROJECTION_ATTRIBUTES.find("REFLECTIVE_SAMPLES").text
    return datetime_obj, gains, offsets, int(rows), int(columns)