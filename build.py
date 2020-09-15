# -*- coding: utf-8 -*- 
import os
import sys
import argparse

def main(input_data_path):
    cmd='GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/multi_hand_tracking/multi_hand_tracking_cpu_text \--calculator_graph_config_file=mediapipe/graphs/hand_tracking/multi_hand_tracking_desktop_live_text.pbtxt'
    
    listfile=os.listdir(input_data_path)
    
    for file in listfile:
    
        if ".DS_Store" in file:
           continue
           
        fullfilename=os.listdir(input_data_path+file)
        for mp4list in fullfilename:
            if ".DS_Store" in mp4list or "txt" in mp4list:
                    continue
            inputfilen='   --input_video_path='+input_data_path+file+'/'+mp4list
            print(mp4list)
            cmdret=cmd+inputfilen
            os.system(cmdret)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='operating Mediapipe')
    parser.add_argument("--input_data_path",help=" ")
    args=parser.parse_args()
    input_data_path=args.input_data_path
    print(input_data_path)
    main(input_data_path)
