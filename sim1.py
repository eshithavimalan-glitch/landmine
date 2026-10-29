import math
import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data

# Set up PyBullet environment
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

plane = p.loadURDF("plane.urdf")
p.changeVisualShape(plane, -1, rgbaColor=[0.76, 0.68, 0.48, 1.0])

START_X, END_X = 0, 10
p.resetDebugVisualizerCamera(
    cameraDistance=10.0,
    cameraYaw=45,
    cameraPitch=-40,
    cameraTargetPosition=[5, 0, 0],
)

# Load single quadruped robot
robot_id = p.loadURDF("quadruped/quadruped.urdf", [START_X, 0, 0.5])

# Joint mapping & gait parameters
MOTORS = {
    "front_rightR": 0,
    "front_rightL": 3,
    "front_leftR": 6,
    "front_leftL": 9,
    "back_rightR": 12,
    "back_rightL": 15,
    "back_leftR": 18,
    "back_leftL": 21,
}
DIAGONAL_PHASE = {
    "front_right": 0.0,
    "back_left": 0.0,
    "front_left": math.pi,
    "back_right": math.pi,
}

AMPLITUDE, FREQUENCY = 0.2, 1.5
NORMAL_SPEED, BOOST_SPEED = 1.4, 1.3
STOP_DURATION, BOOST_DURATION = 4.0, 2.0
DETECTION_RANGE = 0.8

# Load reference images for ORB detection
IMG_FOLDER = r"C:\Users\Eshitha Vimalan\Desktop\landmine"
image_files = ["landmine.jpg", "landmine2.jpg", "landmine3.jpg"]

orb = cv2.ORB_create(nfeatures=800)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
reference_descriptors = []

for fname in image_files:
    path = os.path.join(IMG_FOLDER, fname)
    if os.path.exists(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            _, des = orb.detectAndCompute(img, None)
            if des is not None:
                reference_descriptors.append((fname, des))

# Minimal Minefield (3 landmines along or near the center path)
MINES = [
    {"id": "mine_1", "image": image_files[0], "pos": [2.5, 0.0, 0.03]},
    {"id": "mine_2", "image": image_files[1], "pos": [5.5, 0.2, 0.03]},
    {"id": "mine_3", "image": image_files[2], "pos": [8.0, -0.2, 0.03]},
]

for mine in MINES:
    texture_path = os.path.join(IMG_FOLDER, mine["image"])
    shape = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER, radius=0.15, length=0.05, rgbaColor=[1, 1, 1, 1]
    )
    body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=shape,
        basePosition=mine["pos"],
    )
    if os.path.exists(texture_path):
        p.changeVisualShape(body, -1, textureUniqueId=p.loadTexture(texture_path))

# Minimal Obstacles (1 tree, 1 stone)
OBSTACLES = [
    {"type": "tree", "pos": [4.0, 0.05, 0]},
    {"type": "stone", "pos": [7.0, -0.05, 0]},
]

for obs in OBSTACLES:
    ox, oy, _ = obs["pos"]
    if obs["type"] == "tree":
        trunk = p.createVisualShape(
            p.GEOM_CYLINDER, radius=0.08, length=0.6, rgbaColor=[0.45, 0.28, 0.1, 1]
        )
        canopy = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.35, rgbaColor=[0.13, 0.5, 0.13, 1]
        )
        p.createMultiBody(
            0, -1, trunk, basePosition=[ox, oy, 0.3]
        )
        p.createMultiBody(
            0, -1, canopy, basePosition=[ox, oy, 0.7]
        )
    else:
        stone = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.22, rgbaColor=[0.5, 0.5, 0.5, 1]
        )
        p.createMultiBody(
            0, -1, stone, basePosition=[ox, oy, 0.18]
        )

# Helper functions
def get_camera_image(body_id, w=128, h=96):
    pos, _ = p.getBasePositionAndOrientation(body_id)
    view = p.computeViewMatrix(
        [pos[0], pos[1], pos[2] + 0.3], [pos[0] + 1.0, pos[1], pos[2]], [0, 0, 1]
    )
    proj = p.computeProjectionMatrixFOV(
        fov=70, aspect=1.0, nearVal=0.05, farVal=5.0
    )
    _, _, rgb, _, _ = p.getCameraImage(
        w, h, viewMatrix=view, projectionMatrix=proj, renderer=p.ER_TINY_RENDERER
    )
    return cv2.cvtColor(
        np.reshape(rgb, (h, w, 4))[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR
    )


def check_for_landmine(frame, filename):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, des = orb.detectAndCompute(gray, None)
    if des is None:
        return False, 0

    best_count = 0
    for name, ref_des in reference_descriptors:
        if name == filename:
            matches = bf.match(des, ref_des)
            good_matches = [m for m in matches if m.distance < 70]
            best_count = max(best_count, len(good_matches))
    return best_count >= 8, best_count


# Execution variables
cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera Feed", 512, 384)

state = "MOVING"
stop_end_time = 0.0
boost_end_time = 0.0
reported_mines = set()
start_time = time.time()
step = 0

# Main loop
while True:
    t = time.time() - start_time
    now = time.time()

    # Leg joint movement
    for leg_side, j_idx in MOTORS.items():
        phase = DIAGONAL_PHASE[leg_side[:-1]]
        offset = 0.0 if leg_side.endswith("R") else (math.pi / 4)
        angle = AMPLITUDE * math.sin(2 * math.pi * FREQUENCY * t + phase + offset)
        p.setJointMotorControl2(
            robot_id, j_idx, p.POSITION_CONTROL, targetPosition=angle, force=35
        )

    pos, _ = p.getBasePositionAndOrientation(robot_id)

    # Destination check
    if pos[0] >= END_X:
        print("Destination reached!")
        break

    # State transitions
    if state == "STOPPED" and now >= stop_end_time:
        state = "MOVING"
        boost_end_time = now + BOOST_DURATION

    speed = (
        0.0
        if state == "STOPPED"
        else (BOOST_SPEED if now < boost_end_time else NORMAL_SPEED)
    )

    # Obstacle avoidance (lateral steering)
    target_y = 0.0
    for obs in OBSTACLES:
        ox, oy, _ = obs["pos"]
        if (ox - 1.2) <= pos[0] <= (ox + 1.0):
            target_y = 0.5 if oy <= 0 else -0.5
            break

    vy = max(-0.6, min(0.6, 1.5 * (target_y - pos[1])))
    cur_vel, cur_ang = p.getBaseVelocity(robot_id)
    p.resetBaseVelocity(
        robot_id, linearVelocity=[speed, vy, cur_vel[2]], angularVelocity=cur_ang
    )

    # Camera & detection processing
    if step % 4 == 0:
        cam_frame = get_camera_image(robot_id)
        cv2.imshow("Camera Feed", cv2.resize(cam_frame, (512, 384)))

        if state == "MOVING":
            for mine in MINES:
                if mine["id"] in reported_mines:
                    continue
                mx, my, mz = mine["pos"]
                if abs(pos[0] - mx) < DETECTION_RANGE and abs(pos[1] - my) < DETECTION_RANGE:
                    det_frame = get_camera_image(robot_id, 320, 240)
                    found, matches = check_for_landmine(det_frame, mine["image"])
                    if found:
                        reported_mines.add(mine["id"])
                        print(f"LANDMINE DETECTED at ({mx:.2f}, {my:.2f}) with {matches} matches!")
                        
                        # Add debug text & visual indicator ring
                        p.addUserDebugText(
                            f"({mx:.2f}, {my:.2f})",
                            [mx, my, mz + 0.5],
                            textColorRGB=[1, 0, 0],
                            textSize=0.75,
                        )
                        marker = p.createVisualShape(
                            p.GEOM_CYLINDER, radius=0.35, length=0.01, rgbaColor=[1, 1, 0, 1]
                        )
                        p.createMultiBody(0, -1, marker, basePosition=[mx, my, 0.005])

                        state = "STOPPED"
                        stop_end_time = now + STOP_DURATION
                        break

    cv2.waitKey(1)
    step += 1
    p.stepSimulation()
    time.sleep(1 / 240)

cv2.destroyAllWindows()

# Final summary map
plt.figure(figsize=(8, 4))
if reported_mines:
    m_xs = [m["pos"][0] for m in MINES if m["id"] in reported_mines]
    m_ys = [m["pos"][1] for m in MINES if m["id"] in reported_mines]
    plt.scatter(m_xs, m_ys, marker="^", color="red", s=150, label="Detected Landmine")

plt.scatter(
    [o["pos"][0] for o in OBSTACLES if o["type"] == "tree"],
    [o["pos"][1] for o in OBSTACLES if o["type"] == "tree"],
    marker="o", color="green", s=120, label="Tree",
)
plt.scatter(
    [o["pos"][0] for o in OBSTACLES if o["type"] == "stone"],
    [o["pos"][1] for o in OBSTACLES if o["type"] == "stone"],
    marker="s", color="gray", s=100, label="Stone",
)

plt.axhline(0, color="blue", linestyle="--", alpha=0.4, label="Path")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Mission Summary Map")
plt.legend()
plt.grid(True)
plt.xlim(-1, END_X + 1)
plt.ylim(-2, 2)
plt.tight_layout()
plt.show()