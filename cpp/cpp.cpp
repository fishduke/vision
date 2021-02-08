#include <opencv2/opencv_modules.hpp>
#include <stdio.h>
#include <iostream> 

using namespace cv; 
using namespace std; 

int main() {
    Mat frame; 
    VideoCapture cap; // OpenCV에서 제공되는 VideoCapture Class
    
    // check if file exists. if none program ends
    
    if (cap.open("movie2.mp4") == 0) {
        cout << "no such file!" << endl;
        waitKey(0); 
        }
        
    while(1) {
        cap >> frame; // cap 동영상은 다수의 영상의 집합으로 구성되어 있음. 하나의 영상이 하나의 Matrix로 이동된다. 첫번째 영상부터
        if (frame.empty()) {
            cout << "end of video" << endl;
            break;
            }
            
            imshow("video", frame);
            waitKey(33); // 사용자의 키를 33ms만큼 기다림 → 영상간 간격을 33ms로 설정. 30fps
        }
    }
