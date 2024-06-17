from datasets.utils import Taxonomy

_FINE_LABEL_NAMES = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak",
    "orange",
    "orchid",
    "otter",
    "palm",
    "pear",
    "pickup_truck",
    "pine",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow",
    "wolf",
    "woman",
    "worm",
]

superclass2class = {
'aquatic_mammals':	['beaver', 'dolphin', 'otter', 'seal', 'whale'],
'fish':	['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
'household electrical devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
'large carnivores':	['bear', 'leopard', 'lion', 'tiger', 'wolf'],
'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
'people': ['baby', 'boy', 'girl', 'man', 'woman'],
'reptiles':	['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
'trees': ['maple', 'oak', 'palm', 'pine', 'willow'],
'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
'vehicles 2':	['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
}

superclass2class_v2 = {
'trees': ['maple', 'oak', 'palm', 'pine', 'willow'],
'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
'household electrical devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
'large carnivores':	['bear', 'leopard', 'lion', 'tiger', 'wolf'],
'invertebrates': ['bee', 'beetle', 'butterfly', 'caterpillar', 'worm'],
'hard_shell_animals': ['crab', 'lobster', 'snail', 'turtle', 'cockroach'],
'small_aquatic_animals': ['aquarium_fish', 'flatfish', 'ray', 'trout', 'otter'],
'large_aquatic_animals': ['beaver', 'dolphin', 'seal', 'crocodile', 'shark'],
'outdoor_scenes_2': ['cloud', 'sea', 'bridge', 'road', 'skyscraper'],
'outdoor_scenes_1': ['forest', 'mountain', 'plain', 'castle', 'house'],
'large animals': ['camel', 'cattle', 'elephant', 'whale', 'dinosaur'],
'small mammals': ['shrew', 'squirrel', 'mouse', 'baby', 'raccoon'],
'medium-sized mammals': ['fox', 'porcupine', 'possum', 'skunk', 'kangaroo'],
'pets': ['hamster', 'rabbit', 'lizard', 'snake', 'spider'],
'primates': ['chimpanzee', 'boy', 'girl', 'man', 'woman'],
'personal_vehicles': ['bicycle', 'motorcycle', 'lawn_mower', 'pickup_truck', 'streetcar'],
'transit_vehicles': ['rocket', 'tank', 'tractor', 'bus', 'train'],
}



supersuperclasses = {
    'aquatic_animals' : ['aquatic_mammals', 'fish'],
    'mammals' : ['medium-sized mammals', 'small mammals'],
    'large_animals' : ['large carnivores', 'large omnivores and herbivores'],
    'invertebrates' : ['insects', 'non-insect invertebrates'],
    'plants' : ['flowers', 'trees'],
    'food_related' : ['fruit and vegetables', 'food containers'],
    'devices' : ['household electrical devices', 'household furniture'],
    'outdoor_scenes' : ['large man-made outdoor things', 'large natural outdoor scenes'],
    'vehicles' : ['vehicles 1', 'vehicles 2'],
    'other' : ['people', 'reptiles']
}



def flatten(l):
    return [item for sublist in l for item in sublist]

CLASSES = [[c] for c in _FINE_LABEL_NAMES]

SUPERCLASSES = []
for s in superclass2class.values():
    SUPERCLASSES.append([[c] for c in s])

ALTERNATIVESUPERCLASSES = []
for s in superclass2class_v2.values():
    ALTERNATIVESUPERCLASSES.append([[c] for c in s])

SUPERSUPERCLASSES = []
for s in supersuperclasses.values():
    SUPERSUPERCLASSES.append(flatten([superclass2class[s_] for s_ in s]))


DEFAULT_TAXONOMY = Taxonomy(CLASSES)
SUPERCLASS_TAXONOMY = Taxonomy(SUPERCLASSES)
ALTERNATIVE_SUPERCLASS_TAXONOMY = Taxonomy(ALTERNATIVESUPERCLASSES)
SUPERSUPERCLAS_TAXONOMY = Taxonomy(SUPERSUPERCLASSES)
