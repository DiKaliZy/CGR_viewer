import utility

def PD_control(current_rot, current_vel, desired_rot, desired_vel, kp=500, kd=1):
    torque = kp*(utility.log(current_rot.T@desired_rot)) + kd*(desired_vel - current_vel)
    return torque
