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
def sr_linear_map(human_joint_angles):
  
  ## initialization
  sr_hand_dof = 22
  srhand_joint_angles = np.zeros(sr_hand_dof)

  ## preparation (from start to final: flexion and abduction!!! note the direction of motion!!!)
  abduction = 15 * math.pi / 180.0 # 0.35 max, approximately +-20 deg
  # SR hand - # FF, MF, RF, LF, TH
  # actually, one-to-one is not suitable for Shadowhand's thumb joints;
  # structural difference from the dataglove model... measured data of dataglove cannot be simply one-to-one mapped 
  sr_start = np.array([abduction, 0, 0, 0, \
                       abduction/2, 0, 0, 0, \
                       0, 0, 0, 0, \
                       0, 0, 0, 0, 0, \
                       0, 0, 0, 0, 0])
  sr_final = np.array([-abduction, 1.56, 1.56, 1.56, \
                       -abduction/2, 1.56, 1.56, 1.56, \
                       -abduction, 1.56, 1.56, 1.56, \
                        0, -abduction, 1.56, 1.56, 1.56, \
                        0, 1.21, 0, 0.69, 1.56]) 
  hm_start = np.array([0,    0, 53,  0,   0, 30,  0,   0, 22,  0,   0, 35,  0,   0,  0]) # modify index-middle abduction angle range to allow finger crossing..
            # np.array([0,    0, 53,  0,   0, 22,  0,   0, 22,  0,   0, 35,  0,   0,  0])
  hm_final = np.array([45, 100,  0, 90, 120,  0, 90, 120,  0, 90, 120,  0, 90, 120, 58]) # in dataglove 's sequence

  ## one-to-one joint matching: 2 joint(wrist) + 22 joint(hand) = 24 joint; 2 DOFs(wrist) + 18 DOFs(hand) = 20 DOFs
  # forefinger (abd/add -> flex/ext)
  srhand_joint_angles[0] = linear_map(hm_start[5], hm_final[5], human_joint_angles[5], sr_start[0], sr_final[0])
  srhand_joint_angles[1] = linear_map(hm_start[3], hm_final[3], human_joint_angles[3], sr_start[1], sr_final[1])
  srhand_joint_angles[2] = linear_map(hm_start[4], hm_final[4], human_joint_angles[4], sr_start[2], sr_final[2])
  srhand_joint_angles[3] = srhand_joint_angles[2] * 2.0 / 3.0 # two-thirds rule
  # middle finger
  srhand_joint_angles[4] = linear_map(hm_start[5], hm_final[5], human_joint_angles[5], 0.0, 0.0) # stick to neutral position
  srhand_joint_angles[5] = linear_map(hm_start[6], hm_final[6], human_joint_angles[6], sr_start[5], sr_final[5])
  srhand_joint_angles[6] = linear_map(hm_start[7], hm_final[7], human_joint_angles[7], sr_start[6], sr_final[6])
  srhand_joint_angles[7] = srhand_joint_angles[6] * 2.0 / 3.0 # two-thirds rule
  # ring finger
  srhand_joint_angles[8] = linear_map(hm_start[8], hm_final[8], human_joint_angles[8], sr_start[8], sr_final[8])
  srhand_joint_angles[9] = linear_map(hm_start[9], hm_final[9], human_joint_angles[9], sr_start[9], sr_final[9])
  srhand_joint_angles[10] = linear_map(hm_start[10], hm_final[10], human_joint_angles[10], sr_start[10], sr_final[10])
  srhand_joint_angles[11] = srhand_joint_angles[10] * 2.0 / 3.0 # two-thirds rule
  # little finger
  srhand_joint_angles[12] = 0.0 # joint between little finger and palm
  srhand_joint_angles[13] = linear_map(hm_start[11], hm_final[11], human_joint_angles[11], sr_start[13], sr_final[13])
  srhand_joint_angles[14] = linear_map(hm_start[12], hm_final[12], human_joint_angles[12], sr_start[14], sr_final[14])
  srhand_joint_angles[15] = linear_map(hm_start[13], hm_final[13], human_joint_angles[13], sr_start[15], sr_final[15])
  srhand_joint_angles[16] = srhand_joint_angles[15] * 2.0 / 3.0 # two-thirds rule
  # forefinger
  srhand_joint_angles[17] = 0.0 # fixed at 0, but there should be coupling...
  srhand_joint_angles[18] = linear_map(hm_start[14], hm_final[14], human_joint_angles[14], sr_start[18], sr_final[18])
  srhand_joint_angles[19] = 0.0 # fixed at 0, but there should be coupling...
  srhand_joint_angles[20] = linear_map(hm_start[1], hm_final[1], human_joint_angles[1], sr_start[20], sr_final[20])
  srhand_joint_angles[21] = linear_map(hm_start[0], hm_final[0], human_joint_angles[0], sr_start[21], sr_final[21])

  return srhand_joint_angles


### Arrange mapping for a whole joint path
def map_glove_to_srhand(human_hand_path):
  # input is of the size (N x 30), containing data of two hands

  # prep
  sr_hand_dof = 22
  sr_hand_path = np.zeros((human_hand_path.shape[0], 2*sr_hand_dof))

  import pdb
  pdb.set_trace()
  print(">>>> Iterate to process the input dataglove data...")
  # iterate to process
  for n in range(human_hand_path.shape[0]):
    print("== processing point {}/{}".format((n+1), human_hand_path.shape[0]))
    # left 
    sr_hand_path[n, :sr_hand_dof] = sr_linear_map(human_hand_path[n, :15])
    # right 
    sr_hand_path[n, sr_hand_dof:] = sr_linear_map(human_hand_path[n, 15:])

  return sr_hand_path


def main():

  file_name = "glove-calib-2020-11-02.h5"
  test_seq_name = 'test_finger_1/glove_angle' # 'test_finger_1_calibrated'

  try:
    options, args = getopt.getopt(sys.argv[1:], "hf:t:", ["help", "file-name=", "test-seq-name"])
  except getopt.GetoptError:
    sys.exit()

  for option, value in options:
    if option in ("-h", "--help"):
      print("Help:\n")
      print("   This script executes the IK results.\n")
      print("Arguments:\n")
      print("   -f, --file-name=, specify the name of the h5 file to read joint trajectory from, suffix is required.\n")
      print("   -t, --test-seq-name=, specify the name of the test sequence to visualize.\n")
      exit(0)
    if option in ("-f", "--file-name"):
      print("Name of the h5 file to read joint trajectory from: {0}\n".format(value))
      file_name = value
    if option in ("-t", "--test-seq-name"):
      print("Name of the test sequence data: {0}\n".format(value))
      test_seq_name = value


  try:

    ### Set up move group
    print("============ Set up Move Group ...")
    dual_hands_group = moveit_commander.MoveGroupCommander("dual_hands")


    ### Read h5 file for dataglove data
    f = h5py.File(file_name, "r")
    human_hand_path = f[test_seq_name][:]  # (N x 30) for both hands
    f.close()
    print('human_hand_path shape is ({} x {})'.format(human_hand_path.shape[0], human_hand_path.shape[1]))
    

    ### Perform one-to-one linear mapping
    sr_hand_path = map_glove_to_srhand(human_hand_path)


    ### Hands: Go to start positions
    print "============ Both hands go to a fixed feasible initial positions..."
    q_init = np.zeros(44)
    dual_hands_start = q_init.tolist() #sr_hand_path[0, :].tolist()
    dual_hands_group.allow_replanning(True)
    dual_hands_group.go(dual_hands_start, wait=True)
    dual_hands_group.stop()


    ### Construct a plan
    print("============ Construct a plan of two arms' motion...")
    cartesian_plan = moveit_msgs.msg.RobotTrajectory()
    cartesian_plan.joint_trajectory.header.frame_id = '/world'
    cartesian_plan.joint_trajectory.joint_names = ['lh_FFJ4', 'lh_FFJ3', 'lh_FFJ2', 'lh_FFJ1'] \
                  + ['lh_MFJ4', 'lh_MFJ3', 'lh_MFJ2', 'lh_MFJ1'] \
                  + ['lh_RFJ4', 'lh_RFJ3', 'lh_RFJ2', 'lh_RFJ1'] \
                  + ['lh_LFJ5', 'lh_LFJ4', 'lh_LFJ3', 'lh_LFJ2', 'lh_LFJ1'] \
                  + ['lh_THJ5', 'lh_THJ4', 'lh_THJ3', 'lh_THJ2', 'lh_THJ1'] \
                  + ['rh_FFJ4', 'rh_FFJ3', 'rh_FFJ2', 'rh_FFJ1'] \
                  + ['rh_MFJ4', 'rh_MFJ3', 'rh_MFJ2', 'rh_MFJ1'] \
                  + ['rh_RFJ4', 'rh_RFJ3', 'rh_RFJ2', 'rh_RFJ1'] \
                  + ['rh_LFJ5', 'rh_LFJ4', 'rh_LFJ3', 'rh_LFJ2', 'rh_LFJ1'] \
                  + ['rh_THJ5', 'rh_THJ4', 'rh_THJ3', 'rh_THJ2', 'rh_THJ1'] \

    # add a non-colliding initial state to the trajectory for it to be able to execute via MoveIt
    path_point = trajectory_msgs.msg.JointTrajectoryPoint()
    path_point.positions = q_init.tolist()
    t = rospy.Time(0) 
    path_point.time_from_start.secs = t.secs
    path_point.time_from_start.nsecs = t.nsecs        
    cartesian_plan.joint_trajectory.points.append(copy.deepcopy(path_point))
    
    # add the original initial point for delay demonstration
    path_point = trajectory_msgs.msg.JointTrajectoryPoint()
    path_point.positions = sr_hand_path[0, :].tolist()
    t = rospy.Time(0.1) # jump to the actual initial state immediately
    path_point.time_from_start.secs = t.secs
    path_point.time_from_start.nsecs = t.nsecs        
    cartesian_plan.joint_trajectory.points.append(copy.deepcopy(path_point))
    
    t_delay = 2.0  # delay for 2 seconds before executing the motion 

    # add the whole path
    for i in range(sr_hand_path.shape[0]):
        path_point = trajectory_msgs.msg.JointTrajectoryPoint()
        path_point.positions = sr_hand_path[i, :]
        t = rospy.Time(t_delay + i*1.0/30.0) #rospy.Time(i*1.0/15.0) # rospy.Time(timestamp_array[i]) # 15 Hz # rospy.Time(0.5*i) #
        path_point.time_from_start.secs = t.secs
        path_point.time_from_start.nsecs = t.nsecs        
        cartesian_plan.joint_trajectory.points.append(copy.deepcopy(path_point))


    import pdb
    pdb.set_trace()

    ### Execute the plan
    print "============ Execute the plan..."
    # execute the plan
    print "============ Execute the planned path..."        
    dual_hands_group.execute(cartesian_plan, wait=True)


  except rospy.ROSInterruptException:
    return

  except KeyboardInterrupt:
    return




if __name__ == '__main__':

  main()



