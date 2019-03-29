#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/video/background_segm.hpp>
//#include <opencv2/core/mat.hpp>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <vector>
#include <time.h>

using namespace cv;
using namespace std;

struct GaussianDistribution
{
	float w;
	float u;
	float sigma;
};

struct PixelGMM
{
    int num;// 高斯模型最大数量
    int index;// 当前使用的高斯模型数量
    GaussianDistribution* gd;// 高斯模型数组指针
};
bool modelInit = false;
const int GAUSSIAN_MODULE_NUMS = 5;// 模型数量
const float ALPHA = 0.005;// 学习速率
const float SIGMA = 30;// 标准差初始值
const float WEIGHT = 0.05;// 高斯模型权重初始值
const float T = 0.7;// 有效高斯分布阈值
int rows, cols;
PixelGMM* ppgmm;

int main(int argc, char *argv[])
{
	clock_t startTime,endTime;
	startTime = clock();
	int num_device = cuda::getCudaEnabledDeviceCount();
	if(num_device <= 0)
	{	
		cerr << "There is no device." << endl;
		return -1;
	}
	int enable_device_id = -1;
	for(int i = 0; i < num_device; i++)
	{
		cuda::DeviceInfo dev_info(i);
		if(dev_info.isCompatible())
		{
			enable_device_id = i;
		}
	}
	cout << "enable_device_id = " << enable_device_id << endl;
	if(enable_device_id < 0)
	{
		cerr << "GPU module is not compatible." << endl;
		return -1;
	}

	//cuda init
	cuda::setDevice(enable_device_id);
	
	//choose which GPU to run on, change this on a multi-GPU system.
	cudaSetDevice(0);
	#if 0
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if(cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed! Haven't a CUDA-capable GPU installed.");
		return -1;
	}
	#endif 
	cudaFree(0);
	
	VideoCapture capture("video.avi");
	Ptr<BackgroundSubtractorMOG2> bg_model = createBackgroundSubtractorMOG2(100,SIGMA,false);
	Mat rate, image, fgimage, fgmask;
	bg_model->setVarThreshold(20);
	cv::Size s;
	s.width = 860; 
	s.height = 540;

	while(1)
	{
		int niters = 1;
            	vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		
		capture >> rate;
		if(!rate.data)
		{
			cout << "picture error!" << endl;
			//return -1;
			break;
		}
		resize(rate, image, s, INTER_CUBIC);

		if (fgimage.empty())
                fgimage.create(image.size(), image.type());

		//cuda::GpuMat gpu_image(image);
		//cuda::GpuMat gpu_fgmask;
            	//bg_model->apply(image, fgmask, ALPHA);
            	//bg_model->apply(gpu_image, gpu_fgmask, ALPHA);
		cv::_InputArray inArray(image);
		cv::_OutputArray outArray(fgmask);
            	bg_model->apply(inArray, outArray, ALPHA);
		//Mat dst;
		//gpu_fgmask.download(outArray);
		fgmask = outArray.getMat();
		//gpu_fgmask.download(fgmask);

            	fgimage = Scalar::all(0);
            	image.copyTo(fgimage, fgmask);
            Mat bgimage;
            	bg_model->getBackgroundImage(bgimage);
		
		Mat temp;
		
		dilate(fgmask, temp, Mat(), Point(-1,-1), niters);//膨胀，3*3的element，迭代次数为niters
            	erode(temp, temp, Mat(), Point(-1,-1), niters*2);//腐蚀
            	dilate(temp, temp, Mat(), Point(-1,-1), niters);
		
		findContours( temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );//找轮廓
            	fgimage = Mat::zeros(image.size(), CV_8UC3);
		
		if(contours.size() > 0)
		{
			float Area;
			Rect rect;
			vector<Point> m;
			for(int i = contours.size() - 1; i >= 0; i--)
			{
				vector<Point> c = contours[i];
                    		//获取面积
                    		Area = contourArea(c);
                   	 	if (Area < 50)//50这个值根据需求设定，这里指的是目标的大小
                    		{
                        		continue;
                    		}
                    		else
                    		{
                        		m = c;
                    		}
                    		rect = boundingRect(m);
				//if(rect.y > 350 && (rect.width < 500 || rect.height < 500))
				if(rect.y > 100 && Area < 100)
				{
					//cout << "rect.x = " << rect.x << ", rect.y = " << rect.y << ", rect.width = " << rect.width << ", rect.height = " << rect.height << endl;
					continue;
				}
				if(rect.y > 200 && Area < 250)
				{
					//cout << "rect.x = " << rect.x << ", rect.y = " << rect.y << ", rect.width = " << rect.width << ", rect.height = " << rect.height << endl;
					continue;
				}
				cout << "rect.x = " << rect.x << ", rect.y = " << rect.y << ", rect.width = " << rect.width << ", rect.height = " << rect.height << endl;
                    		rectangle(image, rect, Scalar(0, 255, 0), 2);
			}
			Scalar color(0, 255, 0);
			drawContours(fgimage, contours, -1, color, CV_FILLED, 8, hierarchy);
		}

		//imshow("image", image);
            	//imshow("fgimage", fgimage);
            	//imshow("fgmask", fgmask);
            	//if (!bgimage.empty())
               		//imshow("bgimage", bgimage);

            	//waitKey(30);
	}
	endTime = clock();
	double duration = (double)(endTime - startTime) / CLOCKS_PER_SEC;
	cout << "Ending duration: " << duration << endl;

	return 0;
}

