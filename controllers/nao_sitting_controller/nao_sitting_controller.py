"""NAO Robot Sitting Pose Controller - Makes NAO sit down"""

from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())

print("NAO Sitting Controller Started")

# Get NAO motors
motor_names = [
    'LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll',
    'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll',
    'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll',
    'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll',
    'HeadYaw', 'HeadPitch'
]

motors = {}
for name in motor_names:
    motor = robot.getDevice(name)
    if motor:
        motors[name] = motor

print(f"Found {len(motors)} motors")

# Sitting pose angles (radians)
sitting_pose = {
    # Legs - bent for sitting
    'LHipYawPitch': 0.0,
    'LHipRoll': 0.0,
    'LHipPitch': -1.4,       # Hip bend ~80 degrees
    'LKneePitch': 2.1,       # Knee bend ~120 degrees
    'LAnklePitch': -0.7,     # Ankle adjust
    'LAnkleRoll': 0.0,

    'RHipRoll': 0.0,
    'RHipPitch': -1.4,
    'RKneePitch': 2.1,
    'RAnklePitch': -0.7,
    'RAnkleRoll': 0.0,

    # Arms - relaxed on legs
    'LShoulderPitch': 1.4,
    'LShoulderRoll': 0.15,
    'LElbowYaw': -1.2,
    'LElbowRoll': -0.5,

    'RShoulderPitch': 1.4,
    'RShoulderRoll': -0.15,
    'RElbowYaw': 1.2,
    'RElbowRoll': 0.5,

    # Head - looking forward
    'HeadYaw': 0.0,
    'HeadPitch': 0.0
}

# Apply sitting pose
print("Applying sitting pose...")
for motor_name, angle in sitting_pose.items():
    if motor_name in motors:
        motors[motor_name].setPosition(angle)

print("✓ NAO is sitting!")

# Main loop
while robot.step(timestep) != -1:
    pass
