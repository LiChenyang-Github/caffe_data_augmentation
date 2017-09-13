#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif  // USE_OPENCV
#include <stdint.h>
#include <vector>
#include "caffe/data_transformer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include <iostream>
#include <ctime>  //### To use time(0) as a seed for srand()

using namespace std;
using namespace cv;
using namespace caffe;

void DatumToMat(const Datum* datum, Mat& cv_img);
void MatToDatum(const cv::Mat& cv_img, Datum* datum);

template <typename Dtype>
void MirrorImg(const Datum* datum, Dtype* top_label, int item_id, const cv::Mat& srcImg, cv::Mat& dstImg);

template <typename Dtype>
void ShrinkImg(const Datum* datum, Dtype* top_label, int item_id, const cv::Mat& srcImg, cv::Mat& dstImg,
 float shrinkFloor, float shrinkStride, int randNumForS);

template <typename Dtype>
bool TranslateImg(const Datum* datum, Dtype* top_label, int item_id, const cv::Mat& srcImg, cv::Mat& dstImg,
 bool x_dir, int x_pixel, bool y_dir, int y_pixel);

void AdjustImgBrightness(const cv::Mat& srcImg, cv::Mat& dstImg, float coefV);

namespace caffe {

template <typename Dtype>
DataLayer<Dtype>::DataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
DataLayer<Dtype>::~DataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  const int labelNum = this->layer_param_.data_param().label_num();
  // Read a data point, and use it to initialize the top blob.
  Datum& datum = *(reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  if (this->output_labels_) {

    vector<int> label_shape;
    label_shape.push_back(batch_size);
    label_shape.push_back(labelNum);
    label_shape.push_back(1);
    label_shape.push_back(1);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void DataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  const int labelNum = this->layer_param_.data_param().label_num();
  const int translationPixelNum = this->layer_param_.data_param().translationpixel_num();
  const float shrinkFloor = this->layer_param_.data_param().shrink_floor();
  const float shrinkStride = this->layer_param_.data_param().shrink_stride();
  Datum& datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  //### initial a seed function
  srand(time(0));
  bool randNumForF = 0;  //### Flip or not
  bool isOut = false;
  bool randDirForT_x = 0;
  int randNumForT_x = 0;
  bool randDirForT_y = 0;
  int randNumForT_y = 0;
  int randNumForS = 0;
  int shrinkIntervalNum = (1 - shrinkFloor) / shrinkStride;
  int randNumForV = 0;

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    Datum& datum = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    //###
    int imgH = datum.height();
    int imgW = datum.width();
    
    //### because this->data_transformer_->phase_ is a protected object,
    //### we add a public func GetPhase to get it.
    if (this->data_transformer_->GetPhase()==TRAIN)
    {
      vector<Mat> imgs(5);
      for (int i = 0; i < imgs.size(); ++i)
      {
        imgs[i] = Mat(imgH,imgW,CV_8UC3,Scalar(255,255,255));
      }
      // imshow("test0",imgs[0]);
      // imshow("test1",imgs[1]);
      // imshow("test2",imgs[2]);
      // imshow("test3",imgs[3]);
      // waitKey();
      // Mat imgTmp(imgH,imgW,CV_8UC1);
      // Mat imgDst(imgH,imgW,CV_8UC1);

      DatumToMat(&datum, imgs[0]);
      // for (int i = 0; i < datum.float_data_size()/2; ++i)
      // {
      //   circle(imgs[0],Point(int(datum.float_data(i*2)*96),int(datum.float_data(i*2+1)*96)),1,Scalar(0,0,255),2);
      // }
      // imshow("test0",imgs[0]);

      //### randomly flip the img
      randNumForF = rand() % 2;
      //### randomly translate im along x and y axis 
      randDirForT_x = rand() % 2;
      randNumForT_x = rand() % translationPixelNum;
      randDirForT_y = rand() % 2;
      randNumForT_y = rand() % translationPixelNum;
      randNumForS = rand() % (shrinkIntervalNum + 1);
      //### randomly change the value of V
      randNumForV = rand() % 3;


      for(int i=0;i<labelNum;i++){
        top_label[item_id*labelNum+i] = datum.float_data(i); //read float labels
      }

      if (randNumForV == 0){
        imgs[1] = imgs[0].clone();
      }
      else if (randNumForV == 1){
        AdjustImgBrightness(imgs[0], imgs[1], 0.8);
      }
      else{
        AdjustImgBrightness(imgs[0], imgs[1], 1.2);
      }
      // for (int i = 0; i < datum.float_data_size()/2; ++i)
      // {
      //   circle(imgs[1],Point(int(top_label[item_id*labelNum+i*2]*96),int(top_label[item_id*labelNum+i*2+1]*96)),1,Scalar(0,0,255),2);
      // }
      // imshow("test0",imgs[1]);

      if (randNumForF == false){
        imgs[2] = imgs[1].clone();
      }
      else{
        MirrorImg(&datum, top_label, item_id, imgs[1], imgs[2]);
      }
      // for (int i = 0; i < datum.float_data_size()/2; ++i)
      // {
      //   circle(imgs[2],Point(int(top_label[item_id*labelNum+i*2]*96),int(top_label[item_id*labelNum+i*2+1]*96)),1,Scalar(0,0,255),2);
      // }
      // imshow("test1",imgs[2]);

      ShrinkImg(&datum, top_label, item_id, imgs[2], imgs[3],
       shrinkFloor, shrinkStride, randNumForS);
      // for (int i = 0; i < datum.float_data_size()/2; ++i)
      // {
      //   circle(imgs[3],Point(int(top_label[item_id*labelNum+i*2]*96),int(top_label[item_id*labelNum+i*2+1]*96)),1,Scalar(0,0,255),2);
      // }
      // imshow("test2",imgs[3]);

      isOut = TranslateImg(&datum, top_label, item_id, imgs[3], imgs[4], randDirForT_x, randNumForT_x, randDirForT_y, randNumForT_y);
      // for (int i = 0; i < datum.float_data_size()/2; ++i)
      // {
      //   circle(imgs[4],Point(int(top_label[item_id*labelNum+i*2]*96),int(top_label[item_id*labelNum+i*2+1]*96)),1,Scalar(0,0,255),2);
      // }
      // imshow("test3",imgs[4]);
      // waitKey();
      
      //### to change Mat into Datum again
      MatToDatum(imgs[4], &datum);
    }
    else
    {
      for(int i=0;i<labelNum;i++){
        top_label[item_id*labelNum+i] = datum.float_data(i); //read float labels
      }
    }

    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_->Transform(datum, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();
    reader_.free().push(const_cast<Datum*>(&datum));
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(Data);

}  // namespace caffe

/*
To change Datum into Mat
*/
void DatumToMat(const Datum* datum, Mat& cv_img){
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;

  std::string buffer(datum_size, ' ');
  buffer = datum->data();

  for (int h = 0; h < datum_height; ++h) {
    uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        //buffer[datum_index] = static_cast<char>(ptr[img_index++]);
        ptr[img_index++] = static_cast<uchar>(buffer[datum_index]);
      }
    }
  }
}

/*
To change Mat into Datum
*/
void MatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  //datum->clear_float_data();  //To keep the original label
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}

/*
Horizontally mirror the src img
*/
template<typename Dtype> 
void MirrorImg(const Datum* datum, Dtype* top_label, int item_id, const cv::Mat& srcImg, cv::Mat& dstImg){
    flip(srcImg, dstImg, 1);  // flip by y axis

    int labelNum = datum->float_data_size();
    int keyPointNum = labelNum / 2;
    for (int i = 0; i < keyPointNum; ++i){
      top_label[item_id*labelNum+i*2] = 1 - datum->float_data(i*2);
      top_label[item_id*labelNum+i*2+1] = datum->float_data(i*2+1);
    }   
}


template <typename Dtype>
bool TranslateImg(const Datum* datum, Dtype* top_label, int item_id, const cv::Mat& srcImg, cv::Mat& dstImg,
 bool x_dir, int x_pixel, bool y_dir, int y_pixel){
  int labelNum = datum->float_data_size();
  int imgW = datum->width();
  int imgH = datum->height();
  float offset_x = float(x_pixel) / float(imgW);
  float offset_y = float(y_pixel) / float(imgH);
  if (x_dir == false)
  {
    if (y_dir == false) //### move to the topleft
    {
      for (int i = 0; i < labelNum/2; ++i)
      {
          if ((datum->float_data(i*2)<offset_x)||(datum->float_data(i*2+1)<offset_y))
          {
              return true;
          }
      }
      Point tl(x_pixel, y_pixel);
      Point br(imgW-1, imgH-1);
      int roiW = br.x - tl.x + 1;
      int roiH = br.y - tl.y + 1;
      Rect roiSrc(tl.x,tl.y,roiW,roiH);
      Rect roiDst(0,0,roiW,roiH);
      srcImg(roiSrc).copyTo(dstImg(roiDst));
      for (int i = 0; i < labelNum/2; ++i)
      {
          top_label[item_id*labelNum+i*2] = top_label[item_id*labelNum+i*2] - offset_x;
          top_label[item_id*labelNum+i*2+1] = top_label[item_id*labelNum+i*2+1] - offset_y;
      }
      return false;
    }
    else    //### move to the bottomleft
    {
      for (int i = 0; i < labelNum/2; ++i)
      {
          if ((datum->float_data(i*2)<offset_x)||(datum->float_data(i*2+1)>(1-offset_y)))
          {
              return true;
          }
      }
      Point tl(x_pixel, 0);
      Point br(imgW-1, imgH-1-y_pixel);
      int roiW = br.x - tl.x + 1;
      int roiH = br.y - tl.y + 1;
      Rect roiSrc(tl.x,tl.y,roiW,roiH);
      Rect roiDst(0,y_pixel,roiW,roiH);
      srcImg(roiSrc).copyTo(dstImg(roiDst));
      for (int i = 0; i < labelNum/2; ++i)
      {
          top_label[item_id*labelNum+i*2] = top_label[item_id*labelNum+i*2] - offset_x;
          top_label[item_id*labelNum+i*2+1] = top_label[item_id*labelNum+i*2+1] + offset_y;
      }
      return false;
    }
  }
  else
  {
    if (y_dir == false) //### move to the topright
    {
      for (int i = 0; i < labelNum/2; ++i)
      {
          if ((datum->float_data(i*2)>(1-offset_x))||(datum->float_data(i*2+1)<offset_y))
          {
              return true;
          }
      }
      Point tl(0, y_pixel);
      Point br(imgW-1-x_pixel, imgH-1);
      int roiW = br.x - tl.x + 1;
      int roiH = br.y - tl.y + 1;
      Rect roiSrc(tl.x,tl.y,roiW,roiH);
      Rect roiDst(x_pixel,0,roiW,roiH);
      srcImg(roiSrc).copyTo(dstImg(roiDst));
      for (int i = 0; i < labelNum/2; ++i)
      {
          top_label[item_id*labelNum+i*2] = top_label[item_id*labelNum+i*2] + offset_x;
          top_label[item_id*labelNum+i*2+1] = top_label[item_id*labelNum+i*2+1] - offset_y;
      }
      return false;
    }
    else    //### move to the bottomright
    {
      for (int i = 0; i < labelNum/2; ++i)
      {
          if ((datum->float_data(i*2)>(1-offset_x))||(datum->float_data(i*2+1)>(1-offset_y)))
          {
              return true;
          }
      }
      Point tl(0, 0);
      Point br(imgW-1-x_pixel, imgH-1-y_pixel);
      int roiW = br.x - tl.x + 1;
      int roiH = br.y - tl.y + 1;
      Rect roiSrc(tl.x,tl.y,roiW,roiH);
      Rect roiDst(x_pixel,y_pixel,roiW,roiH);
      srcImg(roiSrc).copyTo(dstImg(roiDst));
      for (int i = 0; i < labelNum/2; ++i)
      {
          top_label[item_id*labelNum+i*2] = top_label[item_id*labelNum+i*2] + offset_x;
          top_label[item_id*labelNum+i*2+1] = top_label[item_id*labelNum+i*2+1] + offset_y;
      }
      return false;
    }
  }
}


template <typename Dtype>
void ShrinkImg(const Datum* datum, Dtype* top_label, int item_id, const cv::Mat& srcImg, cv::Mat& dstImg,
 float shrinkFloor, float shrinkStride, int randNumForS){
  int imgW = datum->width();
  int imgH = datum->height();
  int labelNum = datum->float_data_size();
  float shrinkScale = shrinkFloor + shrinkStride * randNumForS;

  if (shrinkScale == 1) //### no shrinking
  {
    srcImg.copyTo(dstImg);
    return;
  }
  int shrinkImgW = imgW * shrinkScale;
  int shrinkImgH = imgH * shrinkScale;
  Mat srcImgTmp;
  resize(srcImg, srcImgTmp, Size(shrinkImgW,shrinkImgH));
  Point tl(imgW/2-shrinkImgW/2, imgH/2-shrinkImgH/2);
  Rect roiDst(tl.x,tl.y,shrinkImgW,shrinkImgH);
  srcImgTmp.copyTo(dstImg(roiDst));
  for (int i = 0; i < labelNum/2; ++i)
  {
      top_label[item_id*labelNum+i*2] = (1.0 - shrinkScale) / 2 + top_label[item_id*labelNum+i*2] * shrinkScale;
      top_label[item_id*labelNum+i*2+1] = (1.0 - shrinkScale) / 2 + top_label[item_id*labelNum+i*2+1] * shrinkScale;
  }
}

/*
To adjust the brightness of a img through multiply V channel with coefV
*/
void AdjustImgBrightness(const cv::Mat& srcImg, cv::Mat& dstImg, float coefV){
  Mat dstTmp;
  vector<Mat> channelsHSV;
  cvtColor(srcImg, dstTmp, COLOR_BGR2HSV);
  split(dstTmp, channelsHSV);
  channelsHSV.at(2) = channelsHSV.at(2) * coefV;
  merge(channelsHSV, dstTmp);
  cvtColor(dstTmp, dstImg, COLOR_HSV2BGR);
}
