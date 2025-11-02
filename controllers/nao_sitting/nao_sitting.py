"""
NAO Robot Sitting Pose Controller
Makes NAO sit on the sofa
"""

from controller import Robot
import math

# Create robot instance
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Get all motors
motors = {}
motor_names = [
    # Legs
    'LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll',
    'RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll',
    # Arms
    'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll',
    'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll',
    # Head
    'HeadYaw', 'HeadPitch'
]

for name in motor_names:
    motor = robot.getDevice(name)
    if motor:
        motors[name] = motor
    else:
        print(f"Warning: Motor {name} not found")

print("NAO Sitting Pose Controller Started")
print(f"Found {len(motors)} motors")

# Define sitting pose angles (in radians)
sitting_pose = {
    # Left Leg - sitting position
    'LHipYawPitch': 0.0,
    'LHipRoll': 0.0,
    'LHipPitch': -1.5,      # Bend hip forward ~85 degrees
    'LKneePitch': 2.0,      # Bend knee ~115 degrees
    'LAnklePitch': 0.0,
    'LAnkleRoll': 0.0,

    # Right Leg - sitting position
    'RHipYawPitch': 0.0,
    'RHipRoll': 0.0,
    'RHipPitch': -1.5,      # Bend hip forward ~85 degrees
    'RKneePitch': 2.0,      # Bend knee ~115 degrees
    'RAnklePitch': 0.0,
    'RAnkleRoll': 0.0,

    # Left Arm - relaxed on leg
    'LShoulderPitch': 1.2,
    'LShoulderRoll': 0.2,
    'LElbowYaw': -0.5,
    'LElbowRoll': -0.3,

    # Right Arm - relaxed on leg
    'RShoulderPitch': 1.2,
    'RShoulderRoll': -0.2,
    'RElbowYaw': 0.5,
    'RElbowRoll': 0.3,

    # Head - looking forward
    'HeadYaw': 0.0,
    'HeadPitch': 0.0
}

# Apply sitting pose
print("\nApplying sitting pose...")
for motor_name, angle in sitting_pose.items():
    if motor_name in motors:
        motors[motor_name].setPosition(angle)
        print(f"  {motor_name}: {angle:.2f} rad ({math.degrees(angle):.1f}°)")
    else:
        print(f"  {motor_name}: NOT FOUND")

print("\n✓ NAO is now sitting!")
print("Controller will keep NAO in sitting position.")

# Main loop - just maintain the pose
while robot.step(timestep) != -1:
    pass  # NAO stays in sitting position
