import pybullet as p
import pybullet_data
import time


def create_long_table_with_edges():
    """
    Create a long table with 4 vertical walls. Completely frictionless surface.
    """

    table_length = 4.0
    table_width  = 1.0
    table_thickness = 0.05
    wall_height = 0.3
    wall_thickness = 0.05

    # --- Table surface ---
    table_col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[table_length/2, table_width/2, table_thickness/2]
    )
    table_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[table_length/2, table_width/2, table_thickness/2],
        rgbaColor=[0.85, 0.85, 0.85, 1]
    )

    table = p.createMultiBody(
        baseCollisionShapeIndex=table_col,
        baseVisualShapeIndex=table_vis,
        basePosition=[0, 0, -table_thickness/2]
    )

    # --- End walls (+X, -X) ---
    wall_col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[wall_thickness/2, table_width/2, wall_height/2]
    )
    wall_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[wall_thickness/2, table_width/2, wall_height/2],
        rgbaColor=[0.1, 0.1, 0.1, 1]
    )

    for x in [table_length/2, -table_length/2]:
        w = p.createMultiBody(
            baseCollisionShapeIndex=wall_col,
            baseVisualShapeIndex=wall_vis,
            basePosition=[x, 0, wall_height/2]
        )
        # No friction, perfect restitution
        p.changeDynamics(w, -1, lateralFriction=0.0, restitution=1.0)

    # --- Side walls (+Y, -Y) ---
    side_wall_col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[table_length/2, wall_thickness/2, wall_height/2]
    )
    side_wall_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[table_length/2, wall_thickness/2, wall_height/2],
        rgbaColor=[0.1, 0.1, 0.1, 1]
    )

    for y in [table_width/2, -table_width/2]:
        w = p.createMultiBody(
            baseCollisionShapeIndex=side_wall_col,
            baseVisualShapeIndex=side_wall_vis,
            basePosition=[0, y, wall_height/2]
        )
        p.changeDynamics(w, -1, lateralFriction=0.0, restitution=1.0)

    return table


def create_puck():
    """
    Create a frictionless, perfectly elastic hockey puck.
    """
    radius = 0.06
    height = 0.02
    mass = 0.17

    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height,
                              rgbaColor=[0, 0, 0, 1])

    puck = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[-1.5, 0.0, height/2 + 0.01]
    )

    # Remove all friction + perfect restitution
    p.changeDynamics(
        puck, -1,
        lateralFriction=0.01,
        spinningFriction=0.01,
        rollingFriction=0.01,
        restitution=1.0,
        linearDamping=0.0,
        angularDamping=0.0
    )

    return puck


def run_env():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setRealTimeSimulation(0)

    create_long_table_with_edges()
    puck = create_puck()

    # Launch puck fully elastically toward +X
    p.resetBaseVelocity(puck, linearVelocity=[1.0, 3.0, 0.0])

    # Optional: make contacts more "perfect"
    p.setPhysicsEngineParameter(contactSlop=0.0)
    p.setPhysicsEngineParameter(numSolverIterations=200)

    while True:
        p.stepSimulation()
        time.sleep(1/240)


if __name__ == "__main__":
    run_env()
