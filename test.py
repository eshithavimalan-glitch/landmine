import pybullet as p
import pybullet_data
import time

print("Starting PyBullet...")

physicsClient = p.connect(p.GUI)

print("Connection:", physicsClient)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.81)

plane = p.loadURDF("plane.urdf")

print("Plane loaded!")

while True:
    p.stepSimulation()
    time.sleep(1 / 240)