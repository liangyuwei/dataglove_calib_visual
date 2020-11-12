#!/usr/bin/env python

import sys
import copy
import rospy
import math
import pdb
import numpy as np
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import trajectory_msgs.msg
from math import pi
from std_msgs.msg import String

import getopt # process the terminal arguments

import h5py


# Linear mapping: (x - x_min) / (x_max - x_min) == (x_hat - x_min_hat) / (x_max_hat - x_min_hat)
def linear_map(x_min, x_max, x, x_min_hat, x_max_hat):
  return (x - x_min) / (x_max - x_min) * (x_max_hat - x_min_hat) + x_min_hat


### Process one path point 
def ir_linear_map(human_joint_angles, left_or_right):
  
  ## initialization
  ir_hand_dof = 12
  irhand_joint_angles = np.zeros(ir_hand_dof)

  ## preparation (from start to final: flexion and abduction!!! note the direction of motion!!!)
  # abduction = 15 * math.pi / 180.0 # 0.35 max, approximately +-20 deg
  # different values for left and right hands...
  ir_start = np.array([0.0,  0.0,  \
                        0.0,  0.0,  \
                        0.0,  0.0,  \
                        0.0,  0.0,  \
                        0.3,  0.4,  0.0,  0.0])
  if (left_or_right):
    ir_final = np.array([-1.6, -1.6, \
                         -1.6, -1.6, \
                         -1.6, -1.6, \
                         -1.6, -1.6, \
                         -1.0, 0.0, -0.2, -0.15]) 
  else:
    ir_final = np.array([-1.6, -1.6, \
                         -1.6, -1.6, \
                         -1.6, -1.6, \
                         -1.6, -1.6, \
                         -1.0, 0.0, -0.2, -0.15]) 


  hm_start = np.array([0,    0, 53,  0,   0, 30,  0,   0, 22,  0,   0, 35,  0,   0,  0]) # modify index-middle abduction angle range to allow finger crossing..
            # np.array([0,    0, 53,  0,   0, 22,  0,   0, 22,  0,   0, 35,  0,   0,  0])
  hm_final = np.array([45, 100,  0, 90, 120,  0, 90, 120,  0, 90, 120,  0, 90, 120, 90]) #58]) # in dataglove 's sequence

  ## one-to-one joint matching
  # index
  irhand_joint_angles[0] = linear_map(hm_start[3], hm_final[3], human_joint_angles[3], ir_start[0], ir_final[0])
  irhand_joint_angles[1] = linear_map(hm_start[4], hm_final[4], human_joint_angles[4], ir_start[1], ir_final[1])
  # middle
  irhand_joint_angles[2] = linear_map(hm_start[6], hm_final[6], human_joint_angles[6], ir_start[2], ir_final[2])
  irhand_joint_angles[3] = linear_map(hm_start[7], hm_final[7], human_joint_angles[7], ir_start[3], ir_final[3])
  # ring
  irhand_joint_angles[4] = linear_map(hm_start[9], hm_final[9], human_joint_angles[9], ir_start[4], ir_final[4]) # stick to neutral position
  irhand_joint_angles[5] = linear_map(hm_start[10], hm_final[10], human_joint_angles[10], ir_start[5], ir_final[5])
  # little
  irhand_joint_angles[6] = linear_map(hm_start[12], hm_final[12], human_joint_angles[12], ir_start[6], ir_final[6])
  irhand_joint_angles[7] = linear_map(hm_start[13], hm_final[13], human_joint_angles[13], ir_start[7], ir_final[7])
  # thumb
  irhand_joint_angles[8] = linear_map(hm_start[14], hm_final[14], human_joint_angles[14], ir_start[8], ir_final[8])
  irhand_joint_angles[9] = linear_map(hm_start[2], hm_final[2], human_joint_angles[2], ir_start[9], ir_final[9])
  irhand_joint_angles[10] = linear_map(hm_start[0], hm_final[0], human_joint_angles[0], ir_start[10], ir_final[10])
  irhand_joint_angles[11] = linear_map(hm_start[1], hm_final[1], human_joint_angles[1], ir_start[11], ir_final[11])
  
  return irhand_joint_angles


### Arrange mapping for a whole joint path
def map_glove_to_irhand(human_hand_path):
  # input is of the size (N x 30), containing data of two hands

  # prep
  ir_hand_dof = 12
  ir_hand_path = np.zeros((human_hand_path.shape[0], 2*ir_hand_dof))

  import pdb
  pdb.set_trace()
  print(">>>> Iterate to process the input dataglove data...")
  # iterate to process
  for n in range(human_hand_path.shape[0]):
    print("== processing point {}/{}".format((n+1), human_hand_path.shape[0]))
    # left 
    ir_hand_path[n, :ir_hand_dof] = ir_linear_map(human_hand_path[n, :15], True)
    # right 
    ir_hand_path[n, ir_hand_dof:] = ir_linear_map(human_hand_path[n, 15:], False)

  return ir_hand_path


def main():

  file_name = "glove-calib-2020-11-02.h5"
  test_seq_name = 'test_finger_1/glove_angle' # 'test_finger_1_calibrated'
  use_old_version = False

  try:
    options, args = getopt.getopt(sys.argv[1:], "hf:t:o", ["help", "file-name=", "test-seq-name=", "old-version"])
  except getopt.GetoptError:
    sys.exit()

  for option, value in options:
    if option in ("-h", "--help"):
      print("Help:\n")
      print("   This script executes the IK results.\n")
      print("Arguments:\n")
      print("   -f, --file-name=, specify the name of the h5 file to read joint trajectory from, suffix is required.\n")
      print("   -t, --test-seq-name=, specify the name of the test sequence to visualize.\n")
      print("   -o, --old-version, specify to use the data collected by old version of dataglove (S14+).\n")
      exit(0)
    if option in ("-f", "--file-name"):
      print("Name of the h5 file to read joint trajectory from: {0}\n".format(value))
      file_name = value
    if option in ("-t", "--test-seq-name"):
      print("Name of the test sequence data: {0}\n".format(value))
      test_seq_name = value
    if option in ("-o", "--old-version"):
      print("Using the old version of dataglove data...")
      use_old_version = True


  try:

    ### Set up move group
    print("============ Set up Move Group ...")
    dual_hands_group = moveit_commander.MoveGroupCommander("dual_hands")


    ### Read h5 file for dataglove data
    f = h5py.File(file_name, "r")
    if use_old_version:
      l_tmp = f[test_seq_name + '/l_glove_angle'][:]
      r_tmp = f[test_seq_name + '/r_glove_angle'][:]
      len_path = l_tmp.shape[0]
      human_hand_path = np.concatenate((l_tmp, \
                                        np.zeros((len_path,1)), \
                                        r_tmp, \
                                        np.zeros((len_path,1)), \
                                        ), axis=1)
    else:
      human_hand_path = f[test_seq_name][:]  # (N x 30) for both hands
    f.close()
    print('human_hand_path shape is ({} x {})'.format(human_hand_path.shape[0], human_hand_path.shape[1]))
    

    ### Perform one-to-one linear mapping
    ir_hand_path = map_glove_to_irhand(human_hand_path)


    ### Hands: Go to start positions
    print("============ Both hands go to a fixed feasible initial positions...")
    q_init = np.zeros(12*2)
    dual_hands_start = q_init.tolist() #sr_hand_path[0, :].tolist()
    dual_hands_group.allow_replanning(True)
    dual_hands_group.go(dual_hands_start, wait=True)
    dual_hands_group.stop()


    ### Construct a plan
    print("============ Construct a plan of two arms' motion...")
    cartesian_plan = moveit_msgs.msg.RobotTrajectory()
    cartesian_plan.joint_trajectory.header.frame_id = '/world'
    cartesian_plan.joint_trajectory.joint_names = ['link1', 'link11', 'link2', 'link22', 'link3', 'link33', 'link4', 'link44', 'link5', 'link51', 'link52', 'link53'] \
    + ['Link1', 'Link11', 'Link2', 'Link22', 'Link3', 'Link33', 'Link4', 'Link44', 'Link5', 'Link51', 'Link52', 'Link53']

    # add a non-colliding initial state to the trajectory for it to be able to execute via MoveIt
    path_point = trajectory_msgs.msg.JointTrajectoryPoint()
    path_point.positions = q_init.tolist()
    t = rospy.Time(0) 
    path_point.time_from_start.secs = t.secs
    path_point.time_from_start.nsecs = t.nsecs        
    cartesian_plan.joint_trajectory.points.append(copy.deepcopy(path_point))
    
    # add the original initial point for delay demonstration
    path_point = trajectory_msgs.msg.JointTrajectoryPoint()
    path_point.positions = ir_hand_path[0, :].tolist()
    t = rospy.Time(0.1) # jump to the actual initial state immediately
    path_point.time_from_start.secs = t.secs
    path_point.time_from_start.nsecs = t.nsecs        
    cartesian_plan.joint_trajectory.points.append(copy.deepcopy(path_point))
    
    t_delay = 2.0  # delay for 2 seconds before executing the motion 

    # add the whole path
    if use_old_version:
      freq = 15.0
    else:
      freq = 30.0
    for i in range(ir_hand_path.shape[0]):
        path_point = trajectory_msgs.msg.JointTrajectoryPoint()
        path_point.positions = ir_hand_path[i, :]
        t = rospy.Time(t_delay + i*1.0/freq) #rospy.Time(i*1.0/15.0) # rospy.Time(timestamp_array[i]) # 15 Hz # rospy.Time(0.5*i) #
        path_point.time_from_start.secs = t.secs
        path_point.time_from_start.nsecs = t.nsecs        
        cartesian_plan.joint_trajectory.points.append(copy.deepcopy(path_point))


    import pdb
    pdb.set_trace()

    ### Execute the plan
    print("============ Execute the plan...")
    # execute the plan
    print("============ Execute the planned path...")
    dual_hands_group.execute(cartesian_plan, wait=True)


  except rospy.ROSInterruptException:
    return

  except KeyboardInterrupt:
    return




if __name__ == '__main__':

  main()



