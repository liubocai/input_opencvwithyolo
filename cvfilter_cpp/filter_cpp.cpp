/**
    Example C++ OpenCV filter plugin that doesn't do anything. Copy/paste this
    to create your own awesome filter plugins for mjpg-streamer.
    
    At the moment, only the input_opencv.so plugin supports filter plugins.
*/

#include "opencv2/opencv.hpp"

#include "modules/class_detector.h"
#include <memory>
#include <thread>
#include <vector>

using namespace cv;
using namespace std;

unique_ptr<Detector> detector(new Detector());

// exports for the filter
extern "C" {
    bool filter_init(const char * args, void** filter_ctx);
    void filter_process(void* filter_ctx, Mat &src, Mat &dst);
    void filter_free(void* filter_ctx);
}


/**
    Initializes the filter. If you return something, it will be passed to the
    filter_process function, and should be freed by the filter_free function
*/
bool filter_init(const char * args, void** filter_ctx) {
    Config config_v5;
    config_v5.net_type = YOLOV5;
    config_v5.detect_thresh = 0.5;
    config_v5.file_model_cfg = "/home/nvidia/mjpg-streamer/mjpg-streamer-experimental/plugins/input_opencv/filters/cvfilter_cpp/config/yolov5n.cfg";
    config_v5.file_model_weights = "/home/nvidia/mjpg-streamer/mjpg-streamer-experimental/plugins/input_opencv/filters/cvfilter_cpp/config/yolov5n.weights";
    config_v5.calibration_image_list_file_txt = "/home/nvidia/mjpg-streamer/mjpg-streamer-experimental/plugins/input_opencv/filters/cvfilter_cpp/config/calibration_images.txt";
    config_v5.inference_precison = FP32;    
    detector->init(config_v5);
    return true;
}

/**
    Called by the OpenCV plugin upon each frame
*/
void filter_process(void* filter_ctx, Mat &src, Mat &dst) {
    // TODO insert your filter code here	
    Mat image;
    src.copyTo(image);
	vector<BatchResult> batch_res;
    vector<Mat> batch_img;
	batch_img.push_back(image);
    if (!batch_img[0].empty()) {       
        std::cout << "before detection batch_img is not empty" << std::endl;        
    }
    //detect
    detector->detect(batch_img, batch_res);
    if (!batch_img[0].empty()) {
        std::cout << "after detection batch_img is not empty" << std::endl;
    }

    //disp
    for (int i=0;i<batch_img.size();++i)
    {
        if (batch_img[i].empty()) {
            // if batch_img[i] is empty
            dst = src;
            std::cout << "empty" << std::endl;
            return;
        }
        for (const auto &r : batch_res[i])
        {
            std::cout << "batch " << i << " id:" << r.id << " prob:" << r.prob << " rect:" << r.rect << std::endl;
            cv::rectangle(batch_img[i], r.rect, cv::Scalar(255, 0, 0), 2);
            std::stringstream stream;
            stream << std::fixed << std::setprecision(2) << "id:" << r.id << "  score:" << r.prob;
            cv::putText(batch_img[i], stream.str(), cv::Point(r.rect.x, r.rect.y - 5), 0, 0.5, cv::Scalar(0, 0, 255), 2);      
        }
        dst = batch_img[i];
    }

    //dst = src;
    
}

/**
    Called when the input plugin is cleaning up
*/
void filter_free(void* filter_ctx) {
    // empty
}

