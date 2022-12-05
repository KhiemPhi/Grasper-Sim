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

