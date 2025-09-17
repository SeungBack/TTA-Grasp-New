import numpy as np
import random
import json

dict_name = 'test_similar_mini'
save_root = '/home/seung/Workspaces/Datasets/GraspNet-1Billion/splits'
scene_ids = list(range(130, 131))
img_ids = list(range(0, 256))


# sample_img_per_scene = 32
# # # sample random images per scene
# sampled_dict = {}
# for scene_id in scene_ids:
#     sampled_img_ids = random.sample(img_ids, sample_img_per_scene)
#     print(f'Scene {scene_id}: {sampled_img_ids}')
#     sampled_dict['scene_{}'.format(str(scene_id).zfill(4))] = sampled_img_ids

# # print(sampled_dict)

# # # save sampled_dict
# with open(f'{save_root}/{dict_name}.json', 'w') as f:
#     json.dump(sampled_dict, f, indent=4)

scene_id_img_id_pairs = [(scene_id, img_id) for scene_id in scene_ids for img_id in img_ids]
with open(f'{save_root}/{dict_name}.json', 'w') as f:
    json.dump(scene_id_img_id_pairs, f, indent=4)
exit()