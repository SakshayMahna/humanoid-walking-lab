import mujoco
from pathlib import Path
from PIL import Image

# Load Robot Model
script_dir = Path(__file__).resolve().parent
robot_xml = script_dir.parent / "robot" / "g1.xml"

model = mujoco.MjModel.from_xml_path(str(robot_xml))
data = mujoco.MjData(model)

# Model Sizes
print("\n=== Model sizes ===")
print(f"nq           = {model.nq}")              # qpos length: free joint 7 (pos + quat) + 29 hinges = 36
print(f"nv           = {model.nv}")              # qvel length / DoFs: free joint 6 + 29 hinges = 35
print(f"nu           = {model.nu}")              # actuators (ctrl length): 0, the XML has no <actuator> block
print(f"njnt         = {model.njnt}")            # joints: 1 free + 29 hinge = 30
print(f"nbody        = {model.nbody}")           # 30 XML bodies + world body (id 0) = 31
print(f"nsensor      = {model.nsensor}")         # gyro, velocimeter, accelerometer, subtreeangmom = 4
print(f"nsensordata  = {model.nsensordata}")     # sensordata length: 4 sensors × 3 = 12
print(f"opt.timestep = {model.opt.timestep}")    # physics step (s): MuJoCo default 0.002, no <option> in XML

# Joint Table
print("\n=== Joints ===")
header = f"{'id':>3} | {'name':<28} | {'type':<6} | {'qposadr':>7} | {'dofadr':>6} | range"
print(header)
print("-" * len(header))
for i in range(model.njnt):
    joint = model.joint(i)
    print(f"{i:>3} | {joint.name:<28} | {mujoco.mjtJoint(joint.type[0]).name.removeprefix('mjJNT_'):<6} | {joint.qposadr[0]:>7} | {joint.dofadr[0]:>6} | {joint.range}")

# Sensor Table
print("\n=== Sensors ===")
header = f"{'id':>3} | {'name':<14} | {'type':<14} | {'adr':>3} | {'dim':>3}"
print(header)
print("-" * len(header))
for i in range(model.nsensor):
    sensor = model.sensor(i)
    print(f"{i:>3} | {sensor.name:<14} | {mujoco.mjtSensor(sensor.type[0]).name.removeprefix('mjSENS_'):<14} | {sensor.adr[0]:>3} | {sensor.dim[0]:>3}")

# Free Joint State
print("\n=== Free joint state ===")
mujoco.mj_forward(model, data)
print(f"position (m)       = {data.qpos[0:3]}")
print(f"quaternion (wxyz)  = {data.qpos[3:7]}")
print(f"lin vel (m/s)      = {data.qvel[0:3]}")
print(f"ang vel (rad/s)    = {data.qvel[3:6]}")

# Render Simulation
with mujoco.Renderer(model) as renderer:
    renderer.update_scene(data)
    image = Image.fromarray(renderer.render())
    image.save(script_dir / "g1_initial.png")
    image.show()

# Free-Fall Test
print("\n=== Free-fall test ===")

duration = 0.5
g = model.opt.gravity[2]
dt = model.opt.timestep

mujoco.mj_resetData(model, data)
z0 = data.qpos[2]

while data.time < duration:
    mujoco.mj_step(model, data)

t = data.time
z_expected = z0 + 0.5 * g * (t**2)
z_actual = data.qpos[2]
error = abs(z_actual - z_expected)

print(f"t = {t:.4f} s | z0 = {z0:.4f} m | expected = {z_expected:.4f} m | actual = {z_actual:.4f} m")
print(f"error = {error * 1000:.3f} mm (dt = {dt} s)")

# The error is not zero? Why?
# No floor or contact
assert data.ncon == 0, f"unexpected contacts: {data.ncon}"

# MuJoCo uses semi-implicit Euler, which drifts by 0.5 * abs(g) * dt * t for constant acceleration (~ 4.9 mm here)
# Tolerance = 2x that
tol = abs(g) * dt * t
assert error < tol, f"free-fall error {error:.5f} m exceeds tolerance {tol:.5f} m"

# The 0.5 * g * (t**2) equation is continuous solution. MuJoCo moves in discrete steps of dt.
# Its default Euler integrator is semi-implicit: each step it updates velocity first, then uses that new velocity to update position.
# Can be observed by calculating each step. Each step moves the body using the velocity at the end of the step.
# The error is proportional to dt, which is why Euler is a first-order method.

# Semi-implicit Euler is cheap and stable. 
# Unlike explicit Euler, which uses the old velocity, it doesn't steadily gain energy in oscillating systems.
# A few millimeters of drift doesn't matter in control. What matters for the policy is that it uses the same dt and integrator it was trained with.

# Free fall video
print("\n=== Free-fall video ===")
video_duration = 0.5
capture_fps = 120
playback_fps = 30

cam = mujoco.MjvCamera()
cam.lookat[:] = [0, 0, 0.2]
cam.distance = 4.0
cam.azimuth = 135
cam.elevation = 0

mujoco.mj_resetData(model, data)
frames = []
with mujoco.Renderer(model, height=480, width=640) as renderer:
    while data.time < video_duration:
        mujoco.mj_step(model, data)
        if len(frames) < data.time * capture_fps:
            renderer.update_scene(data, camera=cam)
            frames.append(Image.fromarray(renderer.render()))

frames[0].save(
    script_dir / "free_fall.gif",
    save_all=True,
    append_images=frames[1:],
    duration=1000 // playback_fps,
    loop=0,
)
print(f"saved {len(frames)} frames to {script_dir / 'free_fall.gif'}")