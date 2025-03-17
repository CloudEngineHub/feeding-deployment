# import urdf into pybullet
import pybullet as p
import pybullet_data

# setup pybullet

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# load urdf
robotId = p.loadURDF("/home/rkjenamani/utensil_experiments/feeding-deployment/src/feeding_deployment/assets/flair_feeding_utensil/feeding_utensil.urdf", [0, 0, 0], useFixedBase=True)

while True:
    p.stepSimulation()