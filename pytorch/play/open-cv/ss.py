


import cv2
import os
ksize = 10


import json
file_path = './dataset/json/test_box_2021_08_25_수정.json'
with open(file_path, 'r') as f:
    json_test = json.load(f)

new_json = json_test.copy()

new_json['assets'] = []

for asset in json_test['assets'][400:]:
    img = cv2.imread(asset['image']['path'])
    if type(img)==type(None):
        continue

    new_json['assets'].append(asset)

    for region in asset['region']:
        print(region['tags'][0])
        if region['tags'][0] == 13 or region['tags'][0] == 6:
            x , y, w, h = region['boundingBox']['left'], region['boundingBox']['top'], region['boundingBox']['width'], region['boundingBox']['height']
            x = max(0, x)
            y = max(0, y)
            img.shape
            print(x, y, x+w, y+h)
            if int(x + w) > img.shape[1]:
                w = img.shape[1] - x
            if int(y + h) > img.shape[0]:
                h = img.shape[0] - y
            roi = img[int(y): int(y + h), int(x): int(x + w)]
            roi = cv2.blur(roi, (ksize, ksize))
            img[int(y): int(y + h), int(x): int(x + w)] = roi

            a = asset['image']['path'].replace('dataset', 'dataset2')
            output_path = '/'.join(a.split('/')[:-1])
            os.makedirs(output_path, exist_ok=True)
            cv2.imwrite(a, img)


            # cv2.imshow('test', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


    print(asset['image']['path'])

with open('./dataset2/new_json_box.json', 'w') as f:
    json.dump(new_json, f)

# # img = cv2.imread(json_test['assets'][23]['image']['path'])
# # img
# #
# # json_test['assets'][23]['region'][12]['boundingBox']
#
# a = json_test['assets'][0]['image']['path'].replace('dataset','dataset2')
#
# output_path = '/'.join(a.split('/')[:-1])
# output_path
#
# asset['image']['path']
#
# import os
#
# os.makedirs(output_path,exist_ok=True)
#
# os.getcwd()
#
# img = cv2.imread(json_test['assets'][0]['image']['path'])
#
# img.shape