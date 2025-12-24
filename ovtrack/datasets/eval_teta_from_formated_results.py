import teta
import os
import pickle
import numpy as np
import fcntl
import time
from multiprocessing import Pool, set_start_method
from concurrent.futures import ProcessPoolExecutor
from filelock import FileLock


base_class_synset = {'pop_(soda)', 'oar', 'dining_table', 'wineglass', 'coffee_maker', 'thermostat', 'blinker', 'dirt_bike', 'stirrup', 'helmet', 'fire_alarm', 'handle', 'jersey', 'onion', 'canister', 'fire_engine', 'salami', 'chocolate_bar', 'ram_(animal)', 'clip', 'dress_hat', 'shield', 'tractor_(farm_equipment)', 'pet', 'bunk_bed', 'polo_shirt', 'cowboy_hat', 'sweatshirt', 'boiled_egg', 'blouse', 'hook', 'pickup_truck', 'bandanna', 'bamboo', 'railcar_(part_of_a_train)', 'dartboard', 'giant_panda', 'radio_receiver', 'swimsuit', 'handcart', 'flap', 'clothespin', 'bottle_opener', 'walking_stick', 'crumb', 'ring', 'wooden_spoon', 'earphone', 'deadbolt', 'bowl', 'wheelchair', 'volleyball', 'bracelet', 'brake_light', 'cub_(animal)', 'hose', 'starfish', 'pencil', 'avocado', 'cape', 'log', 'egg_yolk', 'microwave_oven', 'faucet', 'chandelier', 'pumpkin', 'fighter_jet', 'timer', 'sweatband', 'eggplant', 'giraffe', 'food_processor', 'pipe', 'pew_(church_bench)', 'radish', 'identity_card', 'sofa', 'vent', 'toothbrush', 'windmill', 'folding_chair', 'ladybug', 'soap', 'step_stool', 'birdbath', 'mouse_(computer_equipment)', 'fish', 'camera_lens', 'brassiere', 'cellular_telephone', 'strainer', 'lampshade', 'easel', 'tinfoil', 'propeller', 'cigarette_case', 'pen', 'highchair', 'toothpick', 'orange_juice', 'water_bottle', 'ham', 'pie', 'hand_towel', 'cruise_ship', 'toilet_tissue', 'eagle', 'shower_head', 'cube', 'life_buoy', 'fishing_rod', 'eraser', 'booklet', 'coaster', 'cap_(headwear)', 'streetlight', 'jacket', 'bridal_gown', 'soup', 'map', 'beret', 'sleeping_bag', 'bandage', 'briefcase', 'bear', 'eggbeater', 'tablecloth', 'clothes_hamper', 'squirrel', 'tartan', 'belt_buckle', 'calendar', 'bow_(decorative_ribbons)', 'nest', 'cappuccino', 'paintbrush', 'bedspread', 'remote_control', 'wheel', 'sail', 'birdcage', 'blackberry', 'elephant', 'crock_pot', 'cow', 'beanie', 'suitcase', 'knife', 'truck', 'ashtray', 'cover', 'crib', 'barrel', 'mound_(baseball)', 'palette', 'saddle_(on_an_animal)', 'calf', 'sculpture', 'ladder', 'fire_hose', 'lamppost', 'watermelon', 'soupspoon', 'newspaper', 'forklift', 'painting', 'rocking_chair', 'tote_bag', 'Lego', 'stove', 'chicken_(animal)', 'freight_car', 'apple', 'bath_towel', 'cleansing_agent', 'bulletin_board', 'seahorse', 'heart', 'air_conditioner', 'fork', 'wedding_cake', 'rubber_band', 'headband', 'tennis_ball', 'urinal', 'jean', 'solar_array', 'cupboard', 'school_bus', 'bath_mat', 'lion', 'bread-bin', 'hairbrush', 'snowboard', 'toilet', 'trailer_truck', 'hair_dryer', 'screwdriver', 'teapot', 'skateboard', 'awning', 'packet', 'hammer', 'bowler_hat', 'figurine', 'spice_rack', 'iPod', 'golfcart', 'bookcase', 'salad', 'teacup', 'robe', 'police_cruiser', 'webcam', 'steak_(food)', 'teddy_bear', 'suit_(clothing)', 'cab_(taxi)', 'router_(computer_equipment)', 'trousers', 'vacuum_cleaner', 'pot', 'ginger', 'mailbox_(at_home)', 'traffic_light', 'vest', 'scale_(measuring_instrument)', 'coffee_table', 'coconut', 'aerosol_can', 'tiara', 'piano', 'tomato', 'fire_extinguisher', 'visor', 'shovel', 'spatula', 'necklace', 'teakettle', 'hummingbird', 'crayon', 'jar', 'bolt', 'grocery_bag', 'kayak', 'coleslaw', 'flannel', 'bouquet', 'gravestone', 'garden_hose', 'knee_pad', 'bow-tie', 'racket', 'pepper', 'wine_bottle', 'ironing_board', 'bullhorn', 'bucket', 'doughnut', 'magnet', 'cart', 'stereo_(sound_system)', 'turkey_(food)', 'cooler_(for_food)', 'gazelle', 'postcard', 'bib', 'bathrobe', 'paddle', 'peeler_(tool_for_fruit_and_vegetables)', 'sour_cream', 'hat', 'sunhat', 'glass_(drink_container)', 'person', 'calculator', 'tricycle', 'barrette', 'wristlet', 'doorknob', 'cabinet', 'cast', 'cigarette', 'lightbulb', 'blinder_(for_horses)', 'ski_boot', 'corkscrew', 'trunk', 'underwear', 'coffeepot', 'telephone', 'doll', 'candy_cane', 'gun', 'shampoo', 'lettuce', 'thermos_bottle', 'kitten', 'bat_(animal)', 'ski_pole', 'medicine', 'pole', 'wall_socket', 'scoreboard', 'Dixie_cup', 'shoe', 'goose', 'binder', 'carrot', 'lanyard', 'lemon', 'soccer_ball', 'television_set', 'shaker', 'wet_suit', 'coatrack', 'water_cooler', 'Christmas_tree', 'domestic_ass', 'television_camera', 'duck', 'can_opener', 'windshield_wiper', 'almond', 'vending_machine', 'money', 'napkin', 'puppy', 'flower_arrangement', 'surfboard', 'tiger', 'dolphin', 'window_box_(for_plants)', 'skewer', 'duffel_bag', 'jeep', 'artichoke', 'anklet', 'bell', 'beer_bottle', 'bread', 'condiment', 'antenna', 'sandwich', 'monitor_(computer_equipment) computer_monitor', 'meatball', 'rabbit', 'cornice', 'kitchen_sink', 'parachute', 'kilt', 'recliner', 'table_lamp', 'banner', 'apron', 'corset', 'towel_rack', 'raspberry', 'car_(automobile)', 'sled', 'gargle', 'yacht', 'business_card', 'cayenne_(spice)', 'camcorder', 'dishtowel', 'pear', 'pizza', 'duct_tape', 'scissors', 'passport', 'turtle', 'underdrawers', 'flag', 'ladle', 'saddlebag', 'camera', 'peanut_butter', 'strap', 'flamingo', 'mat_(gym_equipment)', 'spectacles', 'toast_(food)', 'hammock', 'bullet_train', 'tapestry', 'potholder', 'iron_(for_clothing)', 'suspenders', 'tank_top_(clothing)', 'bathtub', 'grater', 'stapler_(stapling_machine)', 'crutch', 'hairnet', 'water_faucet', 'footstool', 'seabird', 'buoy', 'manger', 'basketball', 'hot_sauce', 'alligator', 'statue_(sculpture)', 'hamburger', 'tarp', 'aquarium', 'saucer', 'grill', 'mop', 'place_mat', 'baseball_bat', 'hog', 'orange_(fruit)', 'spotlight', 'headscarf', 'ski_parka', 'videotape', 'weathervane', 'newsstand', 'quilt', 'potato', 'butter', 'tape_measure', 'snowman', 'parasail_(sports)', 'projector', 'card', 'can', 'costume', 'blazer', 'cufflink', 'shaving_cream', 'computer_keyboard', 'dish', 'mashed_potato', 'measuring_cup', 'pony', 'armband', 'seashell', 'tongs', 'chocolate_cake', 'tinsel', 'pliers', 'tag', 'bed', 'wind_chime', 'clipboard', 'bus_(vehicle)', 'pillow', 'atomizer', 'table', 'bobbin', 'shopping_cart', 'zucchini', 'black_sheep', 'wagon', 'stop_sign', 'loveseat', 'flipper_(footwear)', 'padlock', 'windsock', 'rifle', 'green_onion', 'CD_player', 'gift_wrap', 'sunglasses', 'flowerpot', 'canoe', 'dresser', 'mirror', 'ferry', 'overalls_(clothing)', 'straw_(for_drinking)', 'pinecone', 'cantaloup', 'whipped_cream', 'desk', 'water_ski', 'fireplug', 'spoon', 'airplane', 'dishwasher', 'gelatin', 'dog', 'olive_oil', 'pastry', 'cat', 'jam', 'shower_curtain', 'edible_corn', 'blender', 'ski', 'ostrich', 'green_bean', 'drawer', 'noseband_(for_animals)', 'sweat_pants', 'monkey', 'jet_plane', 'wok', 'control', 'polar_bear', 'foal', 'tassel', 'marker', 'broom', 'birdhouse', 'veil', 'bottle_cap', 'coin', 'icecream', 'goggles', 'necktie', 'pea_(food)', 'cincture', 'award', 'Band_Aid', 'raincoat', 'sushi', 'plate', 'reamer_(juicer)', 'short_pants', 'sweater', 'milk', 'baby_buggy', 'boat', 'kite', 'microphone', 'bulldog', 'motor_scooter', 'envelope', 'toolbox', 'beanbag', 'hairpin', 'shark', 'baseball', 'notebook', 'birdfeeder', 'thumbtack', 'rolling_pin', 'shirt', 'saltshaker', 'raft', 'wreath', 'football_(American)', 'tennis_racket', 'sock', 'beer_can', 'parka', 'doormat', 'glove', 'slipper_(footwear)', 'bagel', 'tripod', 'typewriter', 'speaker_(stero_equipment)', 'poster', 'tank_(storage_vessel)', 'parrot', 'flute_glass', 'egg', 'handbag', 'flashlight', 'alarm_clock', 'backpack', 'headstall_(for_horses)', 'basket', 'musical_instrument', 'rhinoceros', 'saddle_blanket', 'waffle', 'yogurt', 'lollipop', 'garbage_truck', 'key', 'water_jug', 'celery', 'box', 'carton', 'cookie', 'magazine', 'urn', 'ball', 'walking_cane', 'pancake', 'kiwi_fruit', 'tape_(sticky_cloth_or_paper)', 'street_sign', 'sunflower', 'toaster_oven', 'cherry', 'stool', 'wallet', 'wristband', 'file_cabinet', 'rearview_mirror', 'platter', 'pottery', 'beef_(food)', 'crossbar', 'grizzly', 'nut', 'headboard', 'measuring_stick', 'duckling', 'dental_floss', 'chair', 'bun', 'bicycle', 'blueberry', 'plastic_bag', 'horse', 'shoulder_bag', 'prawn', 'water_tower', 'birthday_cake', 'sweet_potato', 'alcohol', 'gull', 'slide', 'latch', 'pita_(bread)', 'pan_(for_cooking)', 'cushion', 'wall_clock', 'battery', 'camel', 'minivan', 'receipt', 'igniter', 'mask', 'mandarin_orange', 'pigeon', 'lamp', 'choker', 'toaster', 'drum_(musical_instrument)', 'towel', 'mixer_(kitchen_tool)', 'salmon_(fish)', 'yoke_(animal_equipment)', 'parasol', 'sausage', 'curtain', 'runner_(carpet)', 'needle', 'baseball_base', 'cabin_car', 'umbrella', 'wrench', 'sheep', 'dress_suit', 'pacifier', 'nightshirt', 'coat', 'paper_plate', 'blanket', 'sponge', 'broccoli', 'parakeet', 'pickle', 'pineapple', 'armchair', 'home_plate_(baseball)', 'pajamas', 'cup', 'amplifier', 'diaper', 'crucifix', 'button', 'chickpea', 'crate', 'frog', 'bobby_pin', 'peach', 'grape', 'bean_curd', 'cash_register', 'scrubbing_brush', 'pitcher_(vessel_for_liquid)', 'vase', 'pad', 'honey', 'manhole', 'chili_(vegetable)', 'sandal_(type_of_shoe)', 'sink', 'toy', 'oil_lamp', 'lime', 'crisp_(potato_chip)', 'printer', 'globe', 'fish_(food)', 'skirt', 'deer', 'bull', 'cracker', 'kimono', 'watering_can', 'fume_hood', 'perfume', 'earring', 'garlic', 'tights_(clothing)', 'lizard', 'boot', 'camper_(vehicle)', 'toothpaste', 'license_plate', 'Ferris_wheel', 'mast', 'projectile_(weapon)', 'mattress', 'power_shovel', 'telephone_booth', 'muffin', 'wedding_ring', 'motorcycle', 'balloon', 'cardigan', 'ice_maker', 'pelican', 'blackboard', 'elk', 'fruit_juice', 'refrigerator', 'crab_(animal)', 'guitar', 'legging_(clothing)', 'postbox_(public)', 'colander', 'dog_collar', 'tea_bag', 'ambulance', 'spider', 'tortilla', 'handkerchief', 'chopping_board', 'deck_chair', 'coat_hanger', 'basketball_backboard', 'lamb_(animal)', 'French_toast', 'automatic_washer', 'cowbell', 'paper_towel', 'record_player', 'butterfly', 'clasp', 'bead', 'helicopter', 'notepad', 'bottle', 'hinge', 'cucumber', 'reflector', 'dress', 'mousepad', 'bench', 'melon', 'tow_truck', 'wagon_wheel', 'billboard', 'trash_can', 'drill', 'oven', 'cistern', 'mitten', 'dumpster', 'cock', 'cornet', 'trophy_cup', 'knob', 'lip_balm', 'headlight', 'flagpole', 'baseball_glove', 'bird', 'frying_pan', 'crescent_roll', 'tray', 'owl', 'clock_tower', 'taillight', 'cone', 'signboard', 'banana', 'brussels_sprouts', 'pocketknife', 'thermometer', 'crown', 'baseball_cap', 'water_scooter', 'goat', 'turtleneck_(clothing)', 'pepper_mill', 'penguin', 'pretzel', 'strawberry', 'parking_meter', 'belt', 'cake', 'dishrag', 'tissue_paper', 'asparagus', 'golf_club', 'mug', 'wine_bucket', 'sword', 'watch', 'wig', 'mushroom', 'turban', 'pouch', 'dish_antenna', 'life_jacket', 'horse_carriage', 'passenger_car_(part_of_a_train)', 'book', 'sportswear', 'silo', 'jewelry', 'salsa', 'hamster', 'sewing_machine', 'zebra', 'cupcake', 'thread', 'bell_pepper', 'dispenser', 'cauliflower', 'cork_(bottle_plug)', 'fan', 'fireplace', 'jumpsuit', 'phonograph_record', 'candle', 'shopping_bag', 'snowmobile', 'telephone_pole', 'scarf', 'whistle', 'motor', 'chopstick', 'binoculars', 'candle_holder', 'laptop_computer', 'kettle', 'brownie', 'freshener', 'heater', 'poker_(fire_stirring_tool)', 'frisbee', 'lantern', 'barrow', 'crow', 'clock', 'train_(railroad_vehicle)', 'mail_slot', 'flip-flop_(sandal)', 'razorblade', 'steering_wheel', 'radiator', 'ottoman'}
novel_class_synset = {'gag', 'taco', 'sherbert', 'barbell', 'chocolate_mousse', 'ice_pack', 'burrito', 'shepherd_dog', 'handcuff', 'penny_(coin)', 'lamb-chop', 'earplug', 'ferret', 'batter_(food)', 'lab_coat', 'baboon', 'knitting_needle', 'date_(fruit)', 'gargoyle', 'puncher', 'ballet_skirt', 'detergent', 'roller_skate', 'cooker', 'harmonium', 'pipe_bowl', 'crawfish', 'cockroach', 'puffer_(fish)', 'sling_(bandage)', 'lasagna', 'fig_(fruit)', 'eclair', 'tobacco_pipe', 'seaplane', 'race_car', 'neckerchief', 'curling_iron', 'patty_(food)', 'cider', 'microscope', 'bass_horn', 'masher', 'crowbar', 'telephoto_lens', 'prune', 'fedora', 'armor', 'canteen', 'stylus', 'ax', 'bonnet', 'drumstick', 'gasmask', 'boom_microphone', 'cigar_box', 'car_battery', 'bow_(weapon)', 'dove', 'carnation', 'milk_can', 'dragonfly', 'cylinder', 'inhaler', 'liquor', 'machine_gun', 'hummus', 'wooden_leg', 'squid_(food)', 'gemstone', 'die', 'chaise_longue', 'bubble_gum', 'sharpener', 'banjo', 'tambourine', 'smoothie', 'coverall', 'root_beer', 'milestone', 'mallard', 'Tabasco_sauce', 'keg', 'thimble', 'fishbowl', 'locker', 'houseboat', 'brass_plaque', 'compass', 'quiche', 'lightning_rod', 'water_gun', 'Bible', 'tux', 'violin', 'steak_knife', 'cream_pitcher', 'mammoth', 'checkerboard', 'generator', 'pool_table', 'rat', 'subwoofer', 'flash', 'puppet', 'beachball', 'bowling_ball', 'pennant', 'salad_plate', 'coil', 'ice_skate', 'chalice', 'poker_chip', 'clarinet', 'legume', 'vat', 'goldfish', 'bookmark', 'road_map', 'plow_(farm_equipment)', 'cloak', 'shredder_(for_paper)', 'joystick', 'mint_candy', 'river_boat', 'electric_chair', 'jewel', 'army_tank', 'cymbal', 'blimp', 'sawhorse', 'pinwheel', 'crouton', 'gondola_(boat)', 'barge', 'football_helmet', 'paperback_book', 'cleat_(for_securing_rope)', 'grits', 'sugarcane_(plant)', 'saucepan', 'garbage', 'pitchfork', 'sombrero', 'string_cheese', 'wolf', 'bob', 'cornmeal', 'pencil_box', 'corkboard', 'vinegar', 'dumbbell', 'hookah', 'vulture', 'cabana', 'kitchen_table', 'nailfile', 'spear', 'clutch_bag', 'stagecoach', 'drone', 'pirate_flag', 'water_heater', 'fleece', 'hotplate', 'file_(tool)', 'sugar_bowl', 'eyepatch', 'octopus_(animal)', 'satchel', 'chain_mail', 'hot-air_balloon', 'halter_top', 'clementine', 'keycard', 'Sharpie', 'milkshake', 'skullcap', 'funnel', 'hamper', 'scarecrow', 'gorilla', 'headset', 'wardrobe', 'phonebook', 'popsicle', 'tachometer', 'combination_lock', 'armoire', 'chessboard', 'escargot', 'crabmeat', 'waffle_iron', 'diary', 'hand_glass', 'piggy_bank', 'motor_vehicle', 'cougar', 'beeper', 'lemonade', 'passenger_ship', 'vodka', 'hardback_book', 'knocker_(on_a_door)', 'applesauce', 'clippers_(for_plants)', 'cassette', 'quesadilla', 'first-aid_kit', 'space_shuttle', 'paperweight', 'griddle', 'horse_buggy', 'baguet', 'coloring_material', 'diving_board', 'truffle_(chocolate)', 'salmon_(food)', 'unicycle', 'syringe', 'stew', 'hair_curler', 'heron', 'bedpan', 'octopus_(food)', 'handsaw', 'nutcracker', 'crape', 'leather', 'hatbox', 'egg_roll', 'turnip', 'plume', 'falcon', 'manatee', 'bagpipe', 'broach', 'chap', 'pocket_watch', 'pendulum', 'stepladder', 'omelet', 'rib_(food)', 'shears', 'koala', 'persimmon', 'cornbread', 'pudding', 'jelly_bean', 'sofa_bed', 'fudge', 'trench_coat', 'hippopotamus', 'softball', 'parchment', 'pegboard', 'pantyhose', 'bulldozer', 'trampoline', 'playpen', 'puffin', 'Rollerblade', 'ping-pong_ball', 'apricot', 'gameboard', 'casserole', 'lawn_mower', 'dagger', 'convertible_(automobile)', 'bolo_tie', 'boxing_glove', 'shot_glass', 'pencil_sharpener', 'dollar', 'cargo_ship', 'washbasin', 'rag_doll', 'pin_(non_jewelry)', 'soup_bowl', 'radar', 'dinghy', 'music_stool', 'pistol', 'checkbook', 'stirrer', 'bait', 'candy_bar', 'birthday_card', 'cocoa_(beverage)', 'gourd', 'mascot', 'chocolate_milk', 'dropper', 'cooking_utensil', 'dalmatian', 'nosebag_(for_animals)', 'martini', 'sparkler_(fireworks)', 'scraper', 'breechcloth', 'kennel', 'triangle_(musical_instrument)', 'limousine', 'elevator_car', 'soya_milk', 'tequila', 'beetle', 'comic_book', 'saxophone', 'futon', 'hockey_stick', 'papaya', 'poncho', 'matchbox', 'walrus', 'safety_pin', 'pan_(metal_container)', 'snake', 'shower_cap', 'eel', 'dishwasher_detergent', 'hornet', 'dollhouse', 'bulletproof_vest', 'dustpan', 'gravy_boat', 'hourglass', 'inkpad', 'table-tennis_table', 'shaver_(electric)', 'chinaware', 'rodent', 'shawl', 'chime', 'arctic_(type_of_shoe)', 'pug-dog', 'mallet'}

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
    default_eval_config["NUM_PARALLEL_CORES"] = 4  # 16
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


def off_eval_by_multi_thread(resfile_paths, checkpoint_dir, val_ann_path, new_head_list=None, save_log_name = 'eval_result.txt',batch_size=2):
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
    pass