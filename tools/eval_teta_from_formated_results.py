import teta
import os
import pickle
import numpy as np
import fcntl
import json
import time
from multiprocessing import Pool, set_start_method
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from filelock import FileLock


base_class_synset = {'pop_(soda)', 'oar', 'dining_table', 'wineglass', 'coffee_maker', 'thermostat', 'blinker', 'dirt_bike', 'stirrup', 'helmet', 'fire_alarm', 'handle', 'jersey', 'onion', 'canister', 'fire_engine', 'salami', 'chocolate_bar', 'ram_(animal)', 'clip', 'dress_hat', 'shield', 'tractor_(farm_equipment)', 'pet', 'bunk_bed', 'polo_shirt', 'cowboy_hat', 'sweatshirt', 'boiled_egg', 'blouse', 'hook', 'pickup_truck', 'bandanna', 'bamboo', 'railcar_(part_of_a_train)', 'dartboard', 'giant_panda', 'radio_receiver', 'swimsuit', 'handcart', 'flap', 'clothespin', 'bottle_opener', 'walking_stick', 'crumb', 'ring', 'wooden_spoon', 'earphone', 'deadbolt', 'bowl', 'wheelchair', 'volleyball', 'bracelet', 'brake_light', 'cub_(animal)', 'hose', 'starfish', 'pencil', 'avocado', 'cape', 'log', 'egg_yolk', 'microwave_oven', 'faucet', 'chandelier', 'pumpkin', 'fighter_jet', 'timer', 'sweatband', 'eggplant', 'giraffe', 'food_processor', 'pipe', 'pew_(church_bench)', 'radish', 'identity_card', 'sofa', 'vent', 'toothbrush', 'windmill', 'folding_chair', 'ladybug', 'soap', 'step_stool', 'birdbath', 'mouse_(computer_equipment)', 'fish', 'camera_lens', 'brassiere', 'cellular_telephone', 'strainer', 'lampshade', 'easel', 'tinfoil', 'propeller', 'cigarette_case', 'pen', 'highchair', 'toothpick', 'orange_juice', 'water_bottle', 'ham', 'pie', 'hand_towel', 'cruise_ship', 'toilet_tissue', 'eagle', 'shower_head', 'cube', 'life_buoy', 'fishing_rod', 'eraser', 'booklet', 'coaster', 'cap_(headwear)', 'streetlight', 'jacket', 'bridal_gown', 'soup', 'map', 'beret', 'sleeping_bag', 'bandage', 'briefcase', 'bear', 'eggbeater', 'tablecloth', 'clothes_hamper', 'squirrel', 'tartan', 'belt_buckle', 'calendar', 'bow_(decorative_ribbons)', 'nest', 'cappuccino', 'paintbrush', 'bedspread', 'remote_control', 'wheel', 'sail', 'birdcage', 'blackberry', 'elephant', 'crock_pot', 'cow', 'beanie', 'suitcase', 'knife', 'truck', 'ashtray', 'cover', 'crib', 'barrel', 'mound_(baseball)', 'palette', 'saddle_(on_an_animal)', 'calf', 'sculpture', 'ladder', 'fire_hose', 'lamppost', 'watermelon', 'soupspoon', 'newspaper', 'forklift', 'painting', 'rocking_chair', 'tote_bag', 'Lego', 'stove', 'chicken_(animal)', 'freight_car', 'apple', 'bath_towel', 'cleansing_agent', 'bulletin_board', 'seahorse', 'heart', 'air_conditioner', 'fork', 'wedding_cake', 'rubber_band', 'headband', 'tennis_ball', 'urinal', 'jean', 'solar_array', 'cupboard', 'school_bus', 'bath_mat', 'lion', 'bread-bin', 'hairbrush', 'snowboard', 'toilet', 'trailer_truck', 'hair_dryer', 'screwdriver', 'teapot', 'skateboard', 'awning', 'packet', 'hammer', 'bowler_hat', 'figurine', 'spice_rack', 'iPod', 'golfcart', 'bookcase', 'salad', 'teacup', 'robe', 'police_cruiser', 'webcam', 'steak_(food)', 'teddy_bear', 'suit_(clothing)', 'cab_(taxi)', 'router_(computer_equipment)', 'trousers', 'vacuum_cleaner', 'pot', 'ginger', 'mailbox_(at_home)', 'traffic_light', 'vest', 'scale_(measuring_instrument)', 'coffee_table', 'coconut', 'aerosol_can', 'tiara', 'piano', 'tomato', 'fire_extinguisher', 'visor', 'shovel', 'spatula', 'necklace', 'teakettle', 'hummingbird', 'crayon', 'jar', 'bolt', 'grocery_bag', 'kayak', 'coleslaw', 'flannel', 'bouquet', 'gravestone', 'garden_hose', 'knee_pad', 'bow-tie', 'racket', 'pepper', 'wine_bottle', 'ironing_board', 'bullhorn', 'bucket', 'doughnut', 'magnet', 'cart', 'stereo_(sound_system)', 'turkey_(food)', 'cooler_(for_food)', 'gazelle', 'postcard', 'bib', 'bathrobe', 'paddle', 'peeler_(tool_for_fruit_and_vegetables)', 'sour_cream', 'hat', 'sunhat', 'glass_(drink_container)', 'person', 'calculator', 'tricycle', 'barrette', 'wristlet', 'doorknob', 'cabinet', 'cast', 'cigarette', 'lightbulb', 'blinder_(for_horses)', 'ski_boot', 'corkscrew', 'trunk', 'underwear', 'coffeepot', 'telephone', 'doll', 'candy_cane', 'gun', 'shampoo', 'lettuce', 'thermos_bottle', 'kitten', 'bat_(animal)', 'ski_pole', 'medicine', 'pole', 'wall_socket', 'scoreboard', 'Dixie_cup', 'shoe', 'goose', 'binder', 'carrot', 'lanyard', 'lemon', 'soccer_ball', 'television_set', 'shaker', 'wet_suit', 'coatrack', 'water_cooler', 'Christmas_tree', 'domestic_ass', 'television_camera', 'duck', 'can_opener', 'windshield_wiper', 'almond', 'vending_machine', 'money', 'napkin', 'puppy', 'flower_arrangement', 'surfboard', 'tiger', 'dolphin', 'window_box_(for_plants)', 'skewer', 'duffel_bag', 'jeep', 'artichoke', 'anklet', 'bell', 'beer_bottle', 'bread', 'condiment', 'antenna', 'sandwich', 'monitor_(computer_equipment) computer_monitor', 'meatball', 'rabbit', 'cornice', 'kitchen_sink', 'parachute', 'kilt', 'recliner', 'table_lamp', 'banner', 'apron', 'corset', 'towel_rack', 'raspberry', 'car_(automobile)', 'sled', 'gargle', 'yacht', 'business_card', 'cayenne_(spice)', 'camcorder', 'dishtowel', 'pear', 'pizza', 'duct_tape', 'scissors', 'passport', 'turtle', 'underdrawers', 'flag', 'ladle', 'saddlebag', 'camera', 'peanut_butter', 'strap', 'flamingo', 'mat_(gym_equipment)', 'spectacles', 'toast_(food)', 'hammock', 'bullet_train', 'tapestry', 'potholder', 'iron_(for_clothing)', 'suspenders', 'tank_top_(clothing)', 'bathtub', 'grater', 'stapler_(stapling_machine)', 'crutch', 'hairnet', 'water_faucet', 'footstool', 'seabird', 'buoy', 'manger', 'basketball', 'hot_sauce', 'alligator', 'statue_(sculpture)', 'hamburger', 'tarp', 'aquarium', 'saucer', 'grill', 'mop', 'place_mat', 'baseball_bat', 'hog', 'orange_(fruit)', 'spotlight', 'headscarf', 'ski_parka', 'videotape', 'weathervane', 'newsstand', 'quilt', 'potato', 'butter', 'tape_measure', 'snowman', 'parasail_(sports)', 'projector', 'card', 'can', 'costume', 'blazer', 'cufflink', 'shaving_cream', 'computer_keyboard', 'dish', 'mashed_potato', 'measuring_cup', 'pony', 'armband', 'seashell', 'tongs', 'chocolate_cake', 'tinsel', 'pliers', 'tag', 'bed', 'wind_chime', 'clipboard', 'bus_(vehicle)', 'pillow', 'atomizer', 'table', 'bobbin', 'shopping_cart', 'zucchini', 'black_sheep', 'wagon', 'stop_sign', 'loveseat', 'flipper_(footwear)', 'padlock', 'windsock', 'rifle', 'green_onion', 'CD_player', 'gift_wrap', 'sunglasses', 'flowerpot', 'canoe', 'dresser', 'mirror', 'ferry', 'overalls_(clothing)', 'straw_(for_drinking)', 'pinecone', 'cantaloup', 'whipped_cream', 'desk', 'water_ski', 'fireplug', 'spoon', 'airplane', 'dishwasher', 'gelatin', 'dog', 'olive_oil', 'pastry', 'cat', 'jam', 'shower_curtain', 'edible_corn', 'blender', 'ski', 'ostrich', 'green_bean', 'drawer', 'noseband_(for_animals)', 'sweat_pants', 'monkey', 'jet_plane', 'wok', 'control', 'polar_bear', 'foal', 'tassel', 'marker', 'broom', 'birdhouse', 'veil', 'bottle_cap', 'coin', 'icecream', 'goggles', 'necktie', 'pea_(food)', 'cincture', 'award', 'Band_Aid', 'raincoat', 'sushi', 'plate', 'reamer_(juicer)', 'short_pants', 'sweater', 'milk', 'baby_buggy', 'boat', 'kite', 'microphone', 'bulldog', 'motor_scooter', 'envelope', 'toolbox', 'beanbag', 'hairpin', 'shark', 'baseball', 'notebook', 'birdfeeder', 'thumbtack', 'rolling_pin', 'shirt', 'saltshaker', 'raft', 'wreath', 'football_(American)', 'tennis_racket', 'sock', 'beer_can', 'parka', 'doormat', 'glove', 'slipper_(footwear)', 'bagel', 'tripod', 'typewriter', 'speaker_(stero_equipment)', 'poster', 'tank_(storage_vessel)', 'parrot', 'flute_glass', 'egg', 'handbag', 'flashlight', 'alarm_clock', 'backpack', 'headstall_(for_horses)', 'basket', 'musical_instrument', 'rhinoceros', 'saddle_blanket', 'waffle', 'yogurt', 'lollipop', 'garbage_truck', 'key', 'water_jug', 'celery', 'box', 'carton', 'cookie', 'magazine', 'urn', 'ball', 'walking_cane', 'pancake', 'kiwi_fruit', 'tape_(sticky_cloth_or_paper)', 'street_sign', 'sunflower', 'toaster_oven', 'cherry', 'stool', 'wallet', 'wristband', 'file_cabinet', 'rearview_mirror', 'platter', 'pottery', 'beef_(food)', 'crossbar', 'grizzly', 'nut', 'headboard', 'measuring_stick', 'duckling', 'dental_floss', 'chair', 'bun', 'bicycle', 'blueberry', 'plastic_bag', 'horse', 'shoulder_bag', 'prawn', 'water_tower', 'birthday_cake', 'sweet_potato', 'alcohol', 'gull', 'slide', 'latch', 'pita_(bread)', 'pan_(for_cooking)', 'cushion', 'wall_clock', 'battery', 'camel', 'minivan', 'receipt', 'igniter', 'mask', 'mandarin_orange', 'pigeon', 'lamp', 'choker', 'toaster', 'drum_(musical_instrument)', 'towel', 'mixer_(kitchen_tool)', 'salmon_(fish)', 'yoke_(animal_equipment)', 'parasol', 'sausage', 'curtain', 'runner_(carpet)', 'needle', 'baseball_base', 'cabin_car', 'umbrella', 'wrench', 'sheep', 'dress_suit', 'pacifier', 'nightshirt', 'coat', 'paper_plate', 'blanket', 'sponge', 'broccoli', 'parakeet', 'pickle', 'pineapple', 'armchair', 'home_plate_(baseball)', 'pajamas', 'cup', 'amplifier', 'diaper', 'crucifix', 'button', 'chickpea', 'crate', 'frog', 'bobby_pin', 'peach', 'grape', 'bean_curd', 'cash_register', 'scrubbing_brush', 'pitcher_(vessel_for_liquid)', 'vase', 'pad', 'honey', 'manhole', 'chili_(vegetable)', 'sandal_(type_of_shoe)', 'sink', 'toy', 'oil_lamp', 'lime', 'crisp_(potato_chip)', 'printer', 'globe', 'fish_(food)', 'skirt', 'deer', 'bull', 'cracker', 'kimono', 'watering_can', 'fume_hood', 'perfume', 'earring', 'garlic', 'tights_(clothing)', 'lizard', 'boot', 'camper_(vehicle)', 'toothpaste', 'license_plate', 'Ferris_wheel', 'mast', 'projectile_(weapon)', 'mattress', 'power_shovel', 'telephone_booth', 'muffin', 'wedding_ring', 'motorcycle', 'balloon', 'cardigan', 'ice_maker', 'pelican', 'blackboard', 'elk', 'fruit_juice', 'refrigerator', 'crab_(animal)', 'guitar', 'legging_(clothing)', 'postbox_(public)', 'colander', 'dog_collar', 'tea_bag', 'ambulance', 'spider', 'tortilla', 'handkerchief', 'chopping_board', 'deck_chair', 'coat_hanger', 'basketball_backboard', 'lamb_(animal)', 'French_toast', 'automatic_washer', 'cowbell', 'paper_towel', 'record_player', 'butterfly', 'clasp', 'bead', 'helicopter', 'notepad', 'bottle', 'hinge', 'cucumber', 'reflector', 'dress', 'mousepad', 'bench', 'melon', 'tow_truck', 'wagon_wheel', 'billboard', 'trash_can', 'drill', 'oven', 'cistern', 'mitten', 'dumpster', 'cock', 'cornet', 'trophy_cup', 'knob', 'lip_balm', 'headlight', 'flagpole', 'baseball_glove', 'bird', 'frying_pan', 'crescent_roll', 'tray', 'owl', 'clock_tower', 'taillight', 'cone', 'signboard', 'banana', 'brussels_sprouts', 'pocketknife', 'thermometer', 'crown', 'baseball_cap', 'water_scooter', 'goat', 'turtleneck_(clothing)', 'pepper_mill', 'penguin', 'pretzel', 'strawberry', 'parking_meter', 'belt', 'cake', 'dishrag', 'tissue_paper', 'asparagus', 'golf_club', 'mug', 'wine_bucket', 'sword', 'watch', 'wig', 'mushroom', 'turban', 'pouch', 'dish_antenna', 'life_jacket', 'horse_carriage', 'passenger_car_(part_of_a_train)', 'book', 'sportswear', 'silo', 'jewelry', 'salsa', 'hamster', 'sewing_machine', 'zebra', 'cupcake', 'thread', 'bell_pepper', 'dispenser', 'cauliflower', 'cork_(bottle_plug)', 'fan', 'fireplace', 'jumpsuit', 'phonograph_record', 'candle', 'shopping_bag', 'snowmobile', 'telephone_pole', 'scarf', 'whistle', 'motor', 'chopstick', 'binoculars', 'candle_holder', 'laptop_computer', 'kettle', 'brownie', 'freshener', 'heater', 'poker_(fire_stirring_tool)', 'frisbee', 'lantern', 'barrow', 'crow', 'clock', 'train_(railroad_vehicle)', 'mail_slot', 'flip-flop_(sandal)', 'razorblade', 'steering_wheel', 'radiator', 'ottoman'}
base_class_id = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29, 32, 33, 34, 35, 36, 37, 41, 43, 44, 45, 46, 47, 48, 50, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 65, 66, 67, 68, 70, 72, 73, 74, 75, 76, 77, 79, 80, 81, 83, 84, 86, 87, 88, 89, 90, 91, 92, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 107, 108, 109, 110, 111, 112, 114, 115, 116, 118, 120, 121, 122, 124, 125, 127, 128, 129, 132, 133, 134, 135, 137, 138, 139, 141, 143, 145, 146, 148, 149, 150, 152, 153, 154, 156, 157, 158, 160, 162, 163, 165, 166, 168, 169, 170, 171, 173, 174, 175, 176, 177, 178, 180, 181, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 197, 198, 199, 200, 201, 203, 204, 205, 206, 207, 208, 211, 212, 213, 216, 217, 218, 219, 220, 221, 224, 225, 226, 227, 228, 229, 230, 232, 235, 239, 241, 242, 243, 246, 248, 249, 252, 253, 254, 255, 256, 259, 260, 261, 263, 264, 267, 268, 271, 272, 273, 274, 276, 277, 278, 279, 280, 283, 284, 285, 286, 288, 289, 290, 293, 296, 297, 298, 299, 303, 305, 306, 308, 309, 311, 312, 314, 315, 318, 319, 320, 322, 324, 325, 327, 328, 329, 330, 332, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 350, 351, 356, 358, 359, 360, 361, 363, 367, 369, 370, 371, 372, 373, 375, 377, 378, 379, 380, 383, 384, 385, 386, 387, 390, 391, 392, 393, 394, 395, 396, 399, 401, 402, 403, 404, 406, 408, 409, 411, 412, 415, 417, 418, 419, 421, 422, 423, 424, 425, 429, 430, 433, 434, 436, 437, 438, 440, 441, 442, 443, 444, 445, 447, 448, 450, 451, 452, 453, 454, 455, 457, 459, 460, 461, 462, 463, 464, 465, 466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 483, 484, 485, 487, 489, 490, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 504, 505, 507, 510, 511, 512, 514, 515, 517, 519, 520, 521, 522, 523, 524, 525, 526, 528, 529, 530, 531, 533, 534, 536, 537, 539, 540, 544, 546, 547, 548, 549, 550, 552, 553, 554, 555, 556, 558, 559, 562, 563, 564, 565, 566, 569, 570, 573, 576, 578, 579, 581, 584, 586, 587, 588, 589, 590, 591, 592, 593, 595, 596, 598, 600, 601, 604, 605, 607, 608, 609, 611, 612, 613, 614, 615, 617, 621, 622, 623, 624, 626, 627, 628, 629, 630, 631, 633, 636, 637, 639, 641, 642, 643, 644, 645, 647, 649, 650, 652, 653, 654, 655, 656, 658, 659, 660, 661, 666, 667, 668, 669, 670, 673, 675, 676, 677, 679, 680, 681, 682, 683, 684, 685, 687, 689, 692, 694, 695, 696, 697, 698, 699, 700, 701, 703, 704, 705, 706, 707, 708, 709, 711, 713, 715, 716, 717, 718, 719, 720, 721, 723, 724, 725, 726, 728, 731, 732, 734, 735, 736, 737, 738, 739, 740, 741, 742, 744, 745, 746, 747, 748, 749, 750, 751, 753, 756, 757, 760, 761, 762, 763, 765, 766, 767, 768, 770, 771, 773, 774, 775, 776, 777, 780, 781, 782, 786, 789, 790, 791, 793, 794, 795, 797, 798, 799, 800, 801, 802, 804, 806, 807, 811, 813, 814, 816, 817, 818, 819, 821, 825, 826, 827, 828, 830, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 854, 857, 860, 861, 863, 865, 866, 867, 868, 870, 871, 872, 874, 875, 876, 877, 878, 879, 880, 881, 882, 884, 885, 888, 889, 893, 895, 896, 897, 898, 899, 900, 901, 903, 904, 906, 907, 909, 910, 911, 912, 915, 916, 919, 921, 922, 923, 924, 926, 927, 928, 929, 930, 932, 933, 934, 935, 936, 940, 943, 946, 947, 948, 949, 950, 951, 953, 954, 955, 957, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 970, 971, 973, 976, 977, 978, 979, 980, 981, 982, 984, 986, 988, 989, 993, 995, 996, 997, 999, 1000, 1001, 1002, 1004, 1006, 1007, 1008, 1009, 1011, 1013, 1014, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1050, 1051, 1052, 1055, 1056, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1076, 1077, 1078, 1079, 1081, 1082, 1083, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1117, 1120, 1121, 1122, 1123, 1125, 1127, 1128, 1130, 1131, 1132, 1133, 1134, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1147, 1149, 1151, 1152, 1153, 1154, 1155, 1156, 1160, 1161, 1162, 1163, 1164, 1166, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203}
novel_class_synset = {'gag', 'taco', 'sherbert', 'barbell', 'chocolate_mousse', 'ice_pack', 'burrito', 'shepherd_dog', 'handcuff', 'penny_(coin)', 'lamb-chop', 'earplug', 'ferret', 'batter_(food)', 'lab_coat', 'baboon', 'knitting_needle', 'date_(fruit)', 'gargoyle', 'puncher', 'ballet_skirt', 'detergent', 'roller_skate', 'cooker', 'harmonium', 'pipe_bowl', 'crawfish', 'cockroach', 'puffer_(fish)', 'sling_(bandage)', 'lasagna', 'fig_(fruit)', 'eclair', 'tobacco_pipe', 'seaplane', 'race_car', 'neckerchief', 'curling_iron', 'patty_(food)', 'cider', 'microscope', 'bass_horn', 'masher', 'crowbar', 'telephoto_lens', 'prune', 'fedora', 'armor', 'canteen', 'stylus', 'ax', 'bonnet', 'drumstick', 'gasmask', 'boom_microphone', 'cigar_box', 'car_battery', 'bow_(weapon)', 'dove', 'carnation', 'milk_can', 'dragonfly', 'cylinder', 'inhaler', 'liquor', 'machine_gun', 'hummus', 'wooden_leg', 'squid_(food)', 'gemstone', 'die', 'chaise_longue', 'bubble_gum', 'sharpener', 'banjo', 'tambourine', 'smoothie', 'coverall', 'root_beer', 'milestone', 'mallard', 'Tabasco_sauce', 'keg', 'thimble', 'fishbowl', 'locker', 'houseboat', 'brass_plaque', 'compass', 'quiche', 'lightning_rod', 'water_gun', 'Bible', 'tux', 'violin', 'steak_knife', 'cream_pitcher', 'mammoth', 'checkerboard', 'generator', 'pool_table', 'rat', 'subwoofer', 'flash', 'puppet', 'beachball', 'bowling_ball', 'pennant', 'salad_plate', 'coil', 'ice_skate', 'chalice', 'poker_chip', 'clarinet', 'legume', 'vat', 'goldfish', 'bookmark', 'road_map', 'plow_(farm_equipment)', 'cloak', 'shredder_(for_paper)', 'joystick', 'mint_candy', 'river_boat', 'electric_chair', 'jewel', 'army_tank', 'cymbal', 'blimp', 'sawhorse', 'pinwheel', 'crouton', 'gondola_(boat)', 'barge', 'football_helmet', 'paperback_book', 'cleat_(for_securing_rope)', 'grits', 'sugarcane_(plant)', 'saucepan', 'garbage', 'pitchfork', 'sombrero', 'string_cheese', 'wolf', 'bob', 'cornmeal', 'pencil_box', 'corkboard', 'vinegar', 'dumbbell', 'hookah', 'vulture', 'cabana', 'kitchen_table', 'nailfile', 'spear', 'clutch_bag', 'stagecoach', 'drone', 'pirate_flag', 'water_heater', 'fleece', 'hotplate', 'file_(tool)', 'sugar_bowl', 'eyepatch', 'octopus_(animal)', 'satchel', 'chain_mail', 'hot-air_balloon', 'halter_top', 'clementine', 'keycard', 'Sharpie', 'milkshake', 'skullcap', 'funnel', 'hamper', 'scarecrow', 'gorilla', 'headset', 'wardrobe', 'phonebook', 'popsicle', 'tachometer', 'combination_lock', 'armoire', 'chessboard', 'escargot', 'crabmeat', 'waffle_iron', 'diary', 'hand_glass', 'piggy_bank', 'motor_vehicle', 'cougar', 'beeper', 'lemonade', 'passenger_ship', 'vodka', 'hardback_book', 'knocker_(on_a_door)', 'applesauce', 'clippers_(for_plants)', 'cassette', 'quesadilla', 'first-aid_kit', 'space_shuttle', 'paperweight', 'griddle', 'horse_buggy', 'baguet', 'coloring_material', 'diving_board', 'truffle_(chocolate)', 'salmon_(food)', 'unicycle', 'syringe', 'stew', 'hair_curler', 'heron', 'bedpan', 'octopus_(food)', 'handsaw', 'nutcracker', 'crape', 'leather', 'hatbox', 'egg_roll', 'turnip', 'plume', 'falcon', 'manatee', 'bagpipe', 'broach', 'chap', 'pocket_watch', 'pendulum', 'stepladder', 'omelet', 'rib_(food)', 'shears', 'koala', 'persimmon', 'cornbread', 'pudding', 'jelly_bean', 'sofa_bed', 'fudge', 'trench_coat', 'hippopotamus', 'softball', 'parchment', 'pegboard', 'pantyhose', 'bulldozer', 'trampoline', 'playpen', 'puffin', 'Rollerblade', 'ping-pong_ball', 'apricot', 'gameboard', 'casserole', 'lawn_mower', 'dagger', 'convertible_(automobile)', 'bolo_tie', 'boxing_glove', 'shot_glass', 'pencil_sharpener', 'dollar', 'cargo_ship', 'washbasin', 'rag_doll', 'pin_(non_jewelry)', 'soup_bowl', 'radar', 'dinghy', 'music_stool', 'pistol', 'checkbook', 'stirrer', 'bait', 'candy_bar', 'birthday_card', 'cocoa_(beverage)', 'gourd', 'mascot', 'chocolate_milk', 'dropper', 'cooking_utensil', 'dalmatian', 'nosebag_(for_animals)', 'martini', 'sparkler_(fireworks)', 'scraper', 'breechcloth', 'kennel', 'triangle_(musical_instrument)', 'limousine', 'elevator_car', 'soya_milk', 'tequila', 'beetle', 'comic_book', 'saxophone', 'futon', 'hockey_stick', 'papaya', 'poncho', 'matchbox', 'walrus', 'safety_pin', 'pan_(metal_container)', 'snake', 'shower_cap', 'eel', 'dishwasher_detergent', 'hornet', 'dollhouse', 'bulletproof_vest', 'dustpan', 'gravy_boat', 'hourglass', 'inkpad', 'table-tennis_table', 'shaver_(electric)', 'chinaware', 'rodent', 'shawl', 'chime', 'arctic_(type_of_shoe)', 'pug-dog', 'mallet'}
novel_class_id = {13, 14, 17, 20, 21, 30, 31, 38, 39, 40, 42, 49, 51, 52, 63, 69, 71, 78, 82, 85, 93, 105, 106, 113, 117, 119, 123, 126, 130, 131, 136, 140, 142, 144, 147, 151, 155, 159, 161, 164, 167, 172, 179, 182, 196, 202, 209, 210, 214, 215, 222, 223, 231, 233, 234, 236, 237, 238, 240, 244, 245, 247, 250, 251, 257, 258, 262, 265, 266, 269, 270, 275, 281, 282, 287, 291, 292, 294, 295, 300, 301, 302, 304, 307, 310, 313, 316, 317, 321, 323, 326, 331, 333, 348, 349, 352, 353, 354, 355, 357, 362, 364, 365, 366, 368, 374, 376, 381, 382, 388, 389, 397, 398, 400, 405, 407, 410, 413, 414, 416, 420, 426, 427, 428, 431, 432, 435, 439, 446, 449, 456, 458, 467, 478, 479, 480, 481, 482, 486, 488, 491, 492, 503, 506, 508, 509, 513, 516, 518, 527, 532, 535, 538, 541, 542, 543, 545, 551, 557, 560, 561, 567, 568, 571, 572, 574, 575, 577, 580, 582, 583, 585, 594, 597, 599, 602, 603, 606, 610, 616, 618, 619, 620, 625, 632, 634, 635, 638, 640, 646, 648, 651, 657, 662, 663, 664, 665, 671, 672, 674, 678, 686, 688, 690, 691, 693, 702, 710, 712, 714, 722, 727, 729, 730, 733, 743, 752, 754, 755, 758, 759, 764, 769, 772, 778, 779, 783, 784, 785, 787, 788, 792, 796, 803, 805, 808, 809, 810, 812, 815, 820, 822, 823, 824, 829, 831, 849, 850, 851, 852, 853, 855, 856, 858, 859, 862, 864, 869, 873, 883, 886, 887, 890, 891, 892, 894, 902, 905, 908, 913, 914, 917, 918, 920, 925, 931, 937, 938, 939, 941, 942, 944, 945, 952, 956, 958, 969, 972, 974, 975, 983, 985, 987, 990, 991, 992, 994, 998, 1003, 1005, 1010, 1012, 1015, 1016, 1028, 1029, 1030, 1031, 1032, 1047, 1048, 1049, 1053, 1054, 1057, 1058, 1075, 1080, 1084, 1116, 1118, 1119, 1124, 1126, 1129, 1135, 1144, 1145, 1146, 1148, 1150, 1157, 1158, 1159, 1165, 1167, 1193}

def format_teta_result(result_list, type = 'Combined', ignore_title=False):
    """
    :param result_list: teta result like[TETA, LocA, AssoA ..., ClsPr]
    :param type: ['combined', 'base', 'Novel']
    :return:
    """
    result_str = ""
    title_str = "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}\n".format(
        "TETA50:",
        "TETA",
        "LocS",
        "AssocS",
        "ClsS",
        "LocRe",
        "LocPr",
        "AssocRe",
        "AssocPr",
        "ClsRe",
        "ClsPr",
    )
    if not ignore_title:
        result_str += title_str
    first_col = "{:<10} ".format(type)
    result_str += first_col
    formatted_strings = ["{:<10.3f}".format(float(num)) for num in result_list]
    result_str += ' '.join(formatted_strings) + '\n'
    return result_str



def write_to_file(filepath, content):
    max_attempts = 100
    attempt = 0

    while attempt < max_attempts:
        try:
            with open(filepath, 'a') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                f.write(f"{content}\n")
                f.flush()
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return True

        except IOError as e:
            print(f"Process {os.getpid()} waiting... Attempt {attempt + 1}/{max_attempts}")
            time.sleep(1)  # 减少等待时间，使进程更频繁地尝试获取锁
            attempt += 1

    print(f"Process {os.getpid()} failed to write after maximum attempts")
    return False
def compute_teta_on_ovsetup(teta_res, base_class_names, novel_class_names):
    if "COMBINED_SEQ" in teta_res:
        teta_res = teta_res["COMBINED_SEQ"]

    frequent_teta = []
    rare_teta = []
    for key in teta_res:
        if key in base_class_names:
            frequent_teta.append(np.array(teta_res[key]["TETA"][50]).astype(float))
        elif key in novel_class_names:
            rare_teta.append(np.array(teta_res[key]["TETA"][50]).astype(float))

    print("Base and Novel classes performance")

    # print the header
    print(
        "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
            "TETA50:",
            "TETA",
            "LocS",
            "AssocS",
            "ClsS",
            "LocRe",
            "LocPr",
            "AssocRe",
            "AssocPr",
            "ClsRe",
            "ClsPr",
        )
    )

    if frequent_teta:
        freq_teta_mean = np.mean(np.stack(frequent_teta), axis=0)

        # print the frequent teta mean
        print("{:<10} ".format("Base"), end="")
        print(*["{:<10.3f}".format(num) for num in freq_teta_mean])

    else:
        print("No Base classes to evaluate!")
        freq_teta_mean = None
    if rare_teta:
        rare_teta_mean = np.mean(np.stack(rare_teta), axis=0)

        # print the rare teta mean
        print("{:<10} ".format("Novel"), end="")
        print(*["{:<10.3f}".format(num) for num in rare_teta_mean])
    else:
        print("No Novel classes to evaluate!")
        rare_teta_mean = None

    return freq_teta_mean, rare_teta_mean

def evaulate_teta_from_formated_results(resfile_path='results/debug3/epoch_8', ann_file='/data1/clark/dataset/openDomain/TAO/tao/annotations/validation_ours_v1.json'):
    eval_results = dict()

    default_eval_config = teta.config.get_default_eval_config()
    # print only combined since TrackMAP is undefined for per sequence breakdowns
    default_eval_config["PRINT_ONLY_COMBINED"] = True
    default_eval_config["DISPLAY_LESS_PROGRESS"] = True
    default_eval_config["OUTPUT_TEM_RAW_DATA"] = True
    default_eval_config["NUM_PARALLEL_CORES"] = 16  # 16
    default_dataset_config = teta.config.get_default_dataset_config()
    default_dataset_config["TRACKERS_TO_EVAL"] = ["OVTrack"]
    default_dataset_config["GT_FOLDER"] = ann_file
    default_dataset_config["OUTPUT_FOLDER"] = resfile_path
    default_dataset_config["TRACKER_SUB_FOLDER"] = os.path.join(
        resfile_path, "tao_track.json"
    )
    evaluator = teta.Evaluator(default_eval_config)
    dataset_list = [teta.datasets.TAO(default_dataset_config)]

    evaluator.evaluate(dataset_list, [teta.metrics.TETA()])

    eval_results_path = os.path.join(
        resfile_path, "OVTrack", "teta_summary_results.pth"
    )
    eval_res = pickle.load(open(eval_results_path, "rb"))
    combined_result = format_teta_result(eval_res['COMBINED_SEQ']['average']['TETA'][50], 'Combined',
                                         ignore_title=False)

    freq_teta_mean, rare_teta_mean = compute_teta_on_ovsetup(eval_res, base_class_synset, novel_class_synset)
    base_result = format_teta_result(freq_teta_mean.tolist(), "Base", ignore_title=True)
    novel_result = format_teta_result(rare_teta_mean.tolist(), "Novel", ignore_title=True)
    print('\n' + combined_result + base_result + novel_result)
    eval_results['combined_result'] = combined_result
    eval_results['base_result'] = base_result
    eval_results['novel_result'] = novel_result

    return eval_results


def evaluate_and_write(args):
    resfile_path, checkpoint_dir, epoch, val_ann_path, new_head, save_log_name = args
    start_time = time.time()

    # 评估函数
    eval_results = evaulate_teta_from_formated_results(resfile_path=resfile_path, ann_file=val_ann_path)
    combined_result = eval_results['combined_result']
    base_result = eval_results['base_result']
    novel_result = eval_results['novel_result']
    if new_head:
        epoch_line = new_head + '\n' + combined_result + base_result + novel_result + '\n'
    else:
        epoch_line = f'epoch[{epoch}]:' + '\n' + combined_result + base_result + novel_result + '\n'

    # 使用文件锁确保并发写入安全
    # result_file = os.path.join(checkpoint_dir, 'eval_result.txt')
    result_file = os.path.join(checkpoint_dir, save_log_name)
    lock_file = result_file + '.lock'

    with FileLock(lock_file):
        with open(result_file, 'a') as f:
            f.write(epoch_line)

    end_time = time.time()
    print(f"Finished processing {resfile_path} in {end_time - start_time} seconds")
    return epoch_line


def filter_tao_track_json(tao_track_json, max_category_instances, only_novel=0):
    """
    :param tao_track_json:
    :param max_category_instances:
    :param only_novel:  1 means: only novel, -1 means only base, others both
    :return:
    """
    # Step 1: 创建一个字典来按照 category_id 分组
    category_groups = defaultdict(list)

    # Step 2: 将每个检测结果放入相应的 category_id 组
    for detection in tao_track_json:
        category_groups[detection['category_id']].append(detection)

        # Step 3: 对每个类别内的检测结果按 score 降序排列
    for category_id in category_groups:
        category_groups[category_id].sort(key=lambda x: x['score'], reverse=True)

        # Step 4: 仅保留每个类别前 max_category_instances 个检测结果
    filtered_detections = []
    for category_id, detections in category_groups.items():
        if only_novel == 1:
            if category_id in novel_class_id:
                filtered_detections.extend(detections[:max_category_instances])
            else:
                filtered_detections.extend(detections)
        elif only_novel == -1:
            if category_id in base_class_id:
                filtered_detections.extend(detections[:max_category_instances])
            else:
                filtered_detections.extend(detections)
        else:
            filtered_detections.extend(detections[:max_category_instances])

    return filtered_detections


def evaluate_filter_each_category_and_write(args):
    resfile_path, checkpoint_dir, epoch, val_ann_path, new_head, save_log_name, max_instance_per_cate, only_novel = args
    start_time = time.time()

    # filter the tao_track.json here
    ori_tao_track_json_path = os.path.join(resfile_path, 'tao_track.json')
    with open(ori_tao_track_json_path, 'r') as f:
        tao_track_json = json.load(f)
    tao_track_json_filter = filter_tao_track_json(tao_track_json, max_instance_per_cate, only_novel)
    # update the tao_track.json path
    saved_new_path = os.path.join(resfile_path, 'filter_category', 'tao_track.json')
    os.makedirs(os.path.dirname(saved_new_path), exist_ok=True)
    with open(saved_new_path, 'w') as f:
        json.dump(tao_track_json_filter, f)
    resfile_path = os.path.dirname(saved_new_path)

    # 评估函数
    eval_results = evaulate_teta_from_formated_results(resfile_path=resfile_path, ann_file=val_ann_path)
    combined_result = eval_results['combined_result']
    base_result = eval_results['base_result']
    novel_result = eval_results['novel_result']
    if new_head:
        epoch_line = new_head + '\n' + combined_result + base_result + novel_result + '\n'
    else:
        epoch_line = f'epoch[{epoch}]:' + '\n' + combined_result + base_result + novel_result + '\n'

    # 使用文件锁确保并发写入安全
    # result_file = os.path.join(checkpoint_dir, 'eval_result.txt')
    result_file = os.path.join(checkpoint_dir, save_log_name)
    lock_file = result_file + '.lock'

    with FileLock(lock_file):
        with open(result_file, 'a') as f:
            f.write(epoch_line)

    end_time = time.time()
    print(f"Finished processing {resfile_path} in {end_time - start_time} seconds")
    return epoch_line



def off_eval_by_multi_thread(resfile_paths, checkpoint_dir, val_ann_path, new_head_list=None, save_log_name = 'eval_result.txt',batch_size=3):
    start_time = time.time()

    # 按照epoch数字排序文件路径
    # resfile_paths = sorted(resfile_paths, key=lambda x: int(os.path.split(x)[-1].split('_')[-1]))

    # 分批处理，每批5个文件
    # batch_size = 5
    total_files = len(resfile_paths)

    for start_idx in range(0, total_files, batch_size):
        batch_start_time = time.time()

        # 获取当前批次的文件路径
        end_idx = min(start_idx + batch_size, total_files)
        current_batch_paths = resfile_paths[start_idx:end_idx]
        current_new_head_list = new_head_list[start_idx:end_idx] if new_head_list is not None else None

        print(f"\nProcessing batch {start_idx // batch_size + 1}, files {start_idx + 1} to {end_idx}")

        # 准备当前批次的参数
        args_list = [
            (resfile_path, checkpoint_dir, os.path.split(resfile_path)[-1].split('_')[-1], val_ann_path, current_new_head_list[i] if new_head_list else None, save_log_name)
            for i, resfile_path in enumerate(current_batch_paths)
        ]
        print()
        for arg in args_list:
            print(arg)
        print()

        # 设置进程数
        num_processes = len(current_batch_paths)  # 每个文件一个进程

        # 处理当前批次
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(evaluate_and_write, args) for args in args_list]
            batch_results = [future.result() for future in futures]

        batch_end_time = time.time()
        print(f"Batch {start_idx // batch_size + 1} completed in {batch_end_time - batch_start_time:.2f} seconds")

        # 可以在批次之间添加短暂暂停
        if end_idx < total_files:
            print("Waiting for next batch...")
            time.sleep(1)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nAll batches completed. Total processing time: {total_time:.2f} seconds")




def off_eval_by_multi_thread_filter_category(resfile_paths, checkpoint_dir, val_ann_path, new_head_list=None, save_log_name = 'eval_result.txt', batch_size=3, max_instace_per_cate=10000, only_novel=False):
    start_time = time.time()

    # 按照epoch数字排序文件路径
    # resfile_paths = sorted(resfile_paths, key=lambda x: int(os.path.split(x)[-1].split('_')[-1]))

    # 分批处理，每批5个文件
    # batch_size = 5
    total_files = len(resfile_paths)

    for start_idx in range(0, total_files, batch_size):
        batch_start_time = time.time()

        # 获取当前批次的文件路径
        end_idx = min(start_idx + batch_size, total_files)
        current_batch_paths = resfile_paths[start_idx:end_idx]
        current_new_head_list = new_head_list[start_idx:end_idx] if new_head_list is not None else None

        print(f"\nProcessing batch {start_idx // batch_size + 1}, files {start_idx + 1} to {end_idx}")

        # 准备当前批次的参数
        args_list = [
            (resfile_path, checkpoint_dir, os.path.split(resfile_path)[-1].split('_')[-1], val_ann_path, current_new_head_list[i] if new_head_list else None, save_log_name, max_instace_per_cate, only_novel)
            for i, resfile_path in enumerate(current_batch_paths)
        ]
        print()
        for arg in args_list:
            print(arg)
        print()

        # 设置进程数
        num_processes = len(current_batch_paths)  # 每个文件一个进程

        # 处理当前批次
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # futures = [executor.submit(evaluate_and_write, args) for args in args_list]
            # todo only used to debug
            futures = [executor.submit(evaluate_filter_each_category_and_write, args) for args in args_list]
            batch_results = [future.result() for future in futures]

        batch_end_time = time.time()
        print(f"Batch {start_idx // batch_size + 1} completed in {batch_end_time - batch_start_time:.2f} seconds")

        # 可以在批次之间添加短暂暂停
        if end_idx < total_files:
            print("Waiting for next batch...")
            time.sleep(1)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nAll batches completed. Total processing time: {total_time:.2f} seconds")

# def off_eval_by_multi_thread(resfile_paths, checkpoint_dir, val_ann_path):
#     start_time = time.time()
#     # checkpoint_dir = '/data/clark/models/ovtrack/tao_train_dataset/debug_seq_tao_base/debug'
#     # resfile_dir = '/data1/clark/models/ovtrack/resutls/results/results/debug3'
#     #
#     # # 获取并排序文件列表
#     resfile_paths = sorted(resfile_paths, key=lambda x: int(os.path.split(x)[-1].split('_')[-1]))
#     # resfile_dir_list = [os.path.join(resfile_dir, file) for file in resfile_dir_list]
#     # resfile_paths = resfile_dir_list[:5]
#
#     # 准备参数
#     args_list = [
#         (resfile_path, checkpoint_dir, os.path.split(resfile_path)[-1].split('_')[-1], val_ann_path)
#         for resfile_path in resfile_paths
#     ]
#
#     # 设置进程数
#     num_processes = min(len(resfile_paths), os.cpu_count())
#
#     # 使用ProcessPoolExecutor进行并行处理
#     with ProcessPoolExecutor(max_workers=num_processes) as executor:
#         futures = [executor.submit(evaluate_and_write, args) for args in args_list]
#         results = [future.result() for future in futures]
#
#     end_time = time.time()
#     total_time = end_time - start_time
#     print(f"Total processing time: {total_time} seconds")

# def main():
#     start_time = time.time()
#     checkpoint_dir = '/data/clark/models/ovtrack/tao_train_dataset/debug_seq_tao_base/debug'
#     resfile_dir = '/data1/clark/models/ovtrack/resutls/results/results/debug3'
#
#     # 获取并排序文件列表
#     resfile_dir_list = sorted(
#         [file for file in os.listdir(resfile_dir) if file.startswith('epoch_')],
#         key=lambda x: int(x.split('_')[-1])
#     )
#     resfile_dir_list = [os.path.join(resfile_dir, file) for file in resfile_dir_list]
#     resfile_paths = resfile_dir_list[:5]
#
#     # 准备参数
#     args_list = [
#         (resfile_path, checkpoint_dir, os.path.split(resfile_path)[-1].split('_')[-1])
#         for resfile_path in resfile_paths
#     ]
#
#     # 设置进程数
#     num_processes = min(len(resfile_paths), os.cpu_count())
#
#     # 使用ProcessPoolExecutor进行并行处理
#     with ProcessPoolExecutor(max_workers=num_processes) as executor:
#         futures = [executor.submit(evaluate_and_write, args) for args in args_list]
#         results = [future.result() for future in futures]
#
#     end_time = time.time()
#     total_time = end_time - start_time
#     print(f"Total processing time: {total_time} seconds")

if __name__ == '__main__':
    # main()
    """
    # search tracker_thres
    base_dir = '/data1/clark/models/ovtrack/resutls/results/results/debug3/1'
    resfile_paths = [os.path.join(base_dir, path) for path in os.listdir('/data1/clark/models/ovtrack/resutls/results/results/debug3/1') if path.startswith('trackersearch')]
    base_dir = '/data1/clark/models/ovtrack/resutls/results/results/debug3/0'
    resfile_paths2 = [os.path.join(base_dir, path) for path in os.listdir('/data1/clark/models/ovtrack/resutls/results/results/debug3/0') if path.startswith('trackersearch')]
    resfile_paths.extend(resfile_paths2)
    new_head_list = []
    for resfile_path in resfile_paths:
        resfile_name = os.path.split(resfile_path)[-1]
        _, match_score_thr, _, max_bbox, _, epoch = resfile_name.split('_')
        match_score_thr = float(match_score_thr)
        max_bbox = int(max_bbox)
        new_head_list.append(f'epoch[{epoch}], match score thres[{round(match_score_thr, 2)} max bbox [{max_bbox}]]:')
    checpoint_dir = '/data/clark/models/ovtrack/tao_train_dataset/debug_seq_tao_base/cvpr2025/association_256_no_dynamic_motion_cls_fusion_no_gt_cycloss_100bbox_035thres_maxratio0_5xdata_30_frame_range_load_from_detpro_ori_all_aug'
    val_ann_path = '/data1/clark/dataset/openDomain/TAO/tao/annotations/validation_ours_v1.json'
    """

    # eval the checkpoint dir
    base_dir = '/data1/clark/models/ovtrack/resutls/results/results/debug3'
    epoch_list = [f'epoch_{i}' for i in range(1, 11)]
    resfile_paths = [os.path.join(base_dir, path) for path in os.listdir('/data1/clark/models/ovtrack/resutls/results/results/debug3') if path in epoch_list]
    base_dir = '/data1/clark/models/ovtrack/resutls/results/results/debug3/0'
    new_head_list = None
    # for resfile_path in resfile_paths:
    #     resfile_name = os.path.split(resfile_path)[-1]
    #     _, match_score_thr, _, max_bbox, _, epoch = resfile_name.split('_')
    #     match_score_thr = float(match_score_thr)
    #     max_bbox = int(max_bbox)
    #     new_head_list.append(f'epoch[{epoch}], match score thres[{round(match_score_thr, 2)} max bbox [{max_bbox}]]:')
    # checpoint_dir = '/data/clark/models/ovtrack/tao_train_dataset/debug_seq_tao_base/cvpr2025/association_256_no_dynamic_motion_cls_fusion_no_gt_cycloss_100bbox_035thres_maxratio0_3xdata_30_frame_range_load_from_detpro_ori_all_aug'
    # checpoint_dir = '/data/clark/models/ovtrack/tao_train_dataset/debug_seq_tao_base/debug'
    checpoint_dir = '/data/clark/models/ovtrack/tao_train_dataset/seq_tao_base_with_modify/cvpr2025/association_256_no_dynamic_motion_cls_fusion_no_gt_cycloss_100bbox_035thres_maxratio0_5xdata_30_frame_range_load_from_detpro_ori_all_aug'
    val_ann_path = '/data1/clark/dataset/openDomain/TAO/tao/annotations/validation_ours_v1.json'
    # only used to debug
    # resfile_paths[0] = '/data1/clark/models/ovtrack/resutls/results/results/debug3/epoch_5_bak_new'
    resfile_paths[0] = '/data1/clark/models/ovtrack/resutls/results/results/test_new_modify_res'
    # off_eval_by_multi_thread(resfile_paths[:1], checpoint_dir, val_ann_path, new_head_list=new_head_list, batch_size=5)
    # only_novel = False
    # for max_instance_per_cate in range(35000, 100000, 5000):
    # resfile_paths[0] = '/data1/clark/models/ovtrack/resutls/results/results/debug3/epoch_5_bak'

    # 1: only novel -1: only base
    only_novel = -1
    save_log_name = 'only_base_filter.txt'

    for max_instance_per_cate in range(18000, 32000, 2000):
        new_head_list = [f'max_instance_per_cate: {max_instance_per_cate} only in novel']
        print(f"max_instance_per_cate: {max_instance_per_cate}")
        print(f"max_instance_per_cate: {max_instance_per_cate}")
        print(f"max_instance_per_cate: {max_instance_per_cate}")
        print(f"max_instance_per_cate: {max_instance_per_cate}")
        print(f"max_instance_per_cate: {max_instance_per_cate}")
        print(f"max_instance_per_cate: {max_instance_per_cate}")
        off_eval_by_multi_thread_filter_category(resfile_paths[:1], checpoint_dir, val_ann_path, new_head_list=new_head_list, batch_size=5, max_instace_per_cate=max_instance_per_cate, only_novel=only_novel, save_log_name=save_log_name)
    pass