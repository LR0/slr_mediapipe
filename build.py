# -*- coding: utf-8 -*- 
import os
import sys
import argparse

def main(input_data_path,output_data_path):
    cmd='GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/multi_hand_tracking/multi_hand_tracking_cpu_text \--calculator_graph_config_file=mediapipe/graphs/hand_tracking/multi_hand_tracking_desktop_live_text.pbtxt'
    
    listfile=os.listdir(input_data_path)
    
    for file in listfile:
        fullfilename=os.listdir(input_data_path+file)
        for mp4list in fullfilename:
            inputfilen='   --input_video_path='+fullfilename
            cmdret=cmd+inputfilen
            os.system(cmdret)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='operating Mediapipe')
    parser.add_argument("--input_data_path",help=" ")
    args=parser.parse_args()
    input_data_path=args.input_data_path
    print(input_data_path)
    main(input_data_path,output_data_path)
