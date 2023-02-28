Python==3.9


Helpful Links:

https://www.etedal.net/2020/04/pybullet-panda_2.html

Tasks:
Investigate Path Planner Towards Grasp
Investigate Grasp Metrics
Investigate Grasp Sampling

https://github.com/ChenEating716/pybullet-URDF-models


Task: Data collection for grasping task

To run the code: 

python -u demo.py --object <object_name> --num_sim <no.of simulations to run> --folder_path <path where the json results need to be stored>

Eg:

python -u demo.py --object soap --num_sim 5 --folder_path grasp_soap

To view the json in a more readable format, if soap_data.json is the json output, run this command to turn the json into a readable json file

cat output_data/soap_data.json | python -mjson.tool > output_data/aligned_soap_data.json

The aligned output is stored in aligned_soap_data.json

To refer the robotic information accessed such as velocity, positions and forces etc, kindly refer the pybullet documentation to see the return types of the respective functions

Pybullet documentation: http://dirkmittler.homeip.net/blend4web_ce/uranium/bullet/docs/pybullet_quickstartguide.pdf

Functions used in code:
getJointState
getLinkState
getcontactpoints

['flat_screwdriver', 'potato_chip_1', 'two_color_hammer', 'suger_1', 'pudding_box', 'toothpaste_1', 'plate', 'potato_chip_3', 'soap', 'green_bowl', 'plastic_strawberry', 'lipton_tea', 'round_plate_4', 'blue_cup', 'correction_fluid', 'plastic_pear', 'square_plate_2', 'gelatin_box', 'mug', 'red_marker', 'plate_holder', 'round_plate_1', 'doraemon_plate', 'shampoo', 'plastic_plum', 'orion_pie', 'fork', 'magic_clean', 'square_plate_4', 'spoon', 'potato_chip_2', 'book_4', 'blue_tea_box', 'green_cup', 'poker_1', 'book_5', 'book_6', 'phillips_screwdriver', 'plastic_apple', 'square_plate_3', 'blue_moon', 'clear_box_1', 'suger_3', 'bowl', 'book_1', 'large_clamp', 'yellow_bowl', 'orange_cup', 'bleach_cleanser', 'square_plate_1', 'pink_tea_box', 'repellent', 'plastic_lemon', 'pen_container_1', 'book_holder_1', 'remote_controller_1', 'mini_claw_hammer_1', 'scissors', 'cleanser', 'round_plate_3', 'book_holder_2', 'power_drill', 'suger_2', 'medium_clamp', 'small_clamp', 'extra_large_clamp', 'stapler_2', 'remote_controller_2', 'glue_1', 'plastic_banana', 'plastic_orange', 'large_marker', 'knife', 'pitcher', 'blue_plate', 'conditioner', 'cracker_box', 'doraemon_bowl', 'stapler_1', 'black_marker', 'grey_plate', 'sugar_box', 'blue_marker', 'book_holder_3', 'round_plate_2', 'yellow_cup', 'soap_dish', 'small_marker', 'book_3', 'clear_box_2', 'glue_2', 'plastic_peach', 'clear_box', 'book_2']