#include <algorithm>
#include <iostream>
#include <mutex>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <set>
#include <thread>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

// Create groups of Lidar points whose projection into the camera falls into the
// same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes,
                         std::vector<LidarPoint> &lidarPoints,
                         float shrinkFactor, cv::Mat &P_rect_xx,
                         cv::Mat &R_rect_xx, cv::Mat &RT) {
  // loop over all Lidar points and associate them to a 2D bounding box
  cv::Mat X(4, 1, cv::DataType<double>::type);
  cv::Mat Y(3, 1, cv::DataType<double>::type);

  for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1) {
    // assemble vector for matrix-vector-multiplication
    X.at<double>(0, 0) = it1->x;
    X.at<double>(1, 0) = it1->y;
    X.at<double>(2, 0) = it1->z;
    X.at<double>(3, 0) = 1;

    // project Lidar point into camera
    Y = P_rect_xx * R_rect_xx * RT * X;
    cv::Point pt;
    pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);  // pixel coordinates
    pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

    // pointers to all bounding boxes which enclose the current Lidar point
    vector<vector<BoundingBox>::iterator> enclosingBoxes;
    for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin();
         it2 != boundingBoxes.end(); ++it2) {
      // shrink current bounding box slightly to avoid having too many outlier
      // points around the edges
      cv::Rect smallerBox;
      smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
      smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
      smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
      smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

      // check wether point is within current bounding box
      if (smallerBox.contains(pt)) {
        enclosingBoxes.push_back(it2);
      }

    }  // eof loop over all bounding boxes

    // check wether point has been enclosed by one or by multiple boxes
    if (enclosingBoxes.size() == 1) {
      // add Lidar point to bounding box
      enclosingBoxes[0]->lidarPoints.push_back(*it1);
    }

  }  // eof loop over all Lidar points
}

void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize,
                   cv::Size imageSize, bool bWait) {
  // create topview image
  cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

  for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1) {
    // create randomized color for current 3D object
    cv::RNG rng(it1->boxID);
    cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150),
                                      rng.uniform(0, 150));

    // plot Lidar points into top view image
    int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
    float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
    for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end();
         ++it2) {
      // world coordinates
      // world position in m with x facing forward from sensor
      float xw = (*it2).x;
      // world position in m with y facing left from sensor
      float yw = (*it2).y;
      xwmin = xwmin < xw ? xwmin : xw;
      ywmin = ywmin < yw ? ywmin : yw;
      ywmax = ywmax > yw ? ywmax : yw;

      // top-view coordinates
      int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
      int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

      // find enclosing rectangle
      top = top < y ? top : y;
      left = left < x ? left : x;
      bottom = bottom > y ? bottom : y;
      right = right > x ? right : x;

      // draw individual point
      cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
    }

    // draw enclosing rectangle
    cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),
                  cv::Scalar(0, 0, 0), 2);

    // augment object with some key data
    char str1[200], str2[200];
    sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
    putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50),
            cv::FONT_ITALIC, 2, currColor);
    sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
    putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125),
            cv::FONT_ITALIC, 2, currColor);
  }

  // plot distance markers
  float lineSpacing = 0.5;  // gap between distance markers
  int nMarkers = floor(worldSize.height / lineSpacing);
  for (size_t i = 0; i < nMarkers; ++i) {
    int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) +
            imageSize.height;
    cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y),
             cv::Scalar(255, 0, 0));
  }

  // display image
  string windowName = "3D Objects";
  cv::namedWindow(windowName, 1);
  cv::imshow(windowName, topviewImg);

  if (bWait) {
    cv::waitKey(0);  // wait for key to be pressed
  }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox,
                              std::vector<cv::KeyPoint> &kptsPrev,
                              std::vector<cv::KeyPoint> &kptsCurr,
                              std::vector<cv::DMatch> &kptMatches) {
  for (auto kpts_match : kptMatches) {
    if (boundingBox.roi.contains(kptsCurr[kpts_match.trainIdx].pt) &&
        boundingBox.roi.contains(kptsPrev[kpts_match.queryIdx].pt)) {
      boundingBox.kptMatches.push_back(kpts_match);
    }
  }
}

float calcSquaredDist(cv::Point2f p1, cv::Point2f p2) {
  return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}

// Compute time-to-collision (TTC) based on keypoint correspondences in
// successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev,
                      std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate,
                      double &TTC, cv::Mat *visImg) {
  vector<float> distRatios;
  for (int i = 0; i < kptMatches.size(); ++i) {
    for (int j = i + 1; j < kptMatches.size(); ++j) {
      float prevDist = calcSquaredDist(kptsPrev[kptMatches[i].queryIdx].pt,
                                       kptsPrev[kptMatches[j].queryIdx].pt);
      float currDist = calcSquaredDist(kptsCurr[kptMatches[i].trainIdx].pt,
                                       kptsCurr[kptMatches[j].trainIdx].pt);
      float distRatio = currDist / prevDist;
      if (!isnan(distRatio)) distRatios.emplace_back(distRatio);
    }
  }

  sort(distRatios.begin(), distRatios.end());

  int distRatios_Q1Idx = distRatios.size() / 4;
  int distRatios_Q4Idx = distRatios.size() * 3 / 4;

  float distRatiosIqr =
      1.5 * (distRatios[distRatios_Q4Idx] - distRatios[distRatios_Q1Idx]);

  float distRatiosMedian;
  int medianIdx = distRatios.size() / 2;

  if (distRatios.size() % 2 == 0) {
    distRatiosMedian = (distRatios[medianIdx] + distRatios[medianIdx + 1]) / 2;
  } else {
    distRatiosMedian = distRatios[medianIdx];
  }

  vector<float> distRatiosFiltered;
  for (auto distRatio : distRatios) {
    if (fabs(distRatio - distRatiosMedian) < distRatiosIqr)
      distRatiosFiltered.push_back(distRatio);
  }

  float sum = 0;
  for (auto distRatio : distRatiosFiltered) {
    sum += distRatio;
  }
  float distRatiosFilteredMean = sum / distRatiosFiltered.size();

  TTC = -1 / (1 - sqrt(distRatiosFilteredMean)) / frameRate;
}

float calcXMedianLidar(vector<LidarPoint> lidarPoints) {
  float median;
  int medianIdx = lidarPoints.size() / 2;
  if (lidarPoints.size() % 2 == 0) {
    median = (lidarPoints[medianIdx].x + lidarPoints[medianIdx + 1].x) / 2;
  } else {
    median = lidarPoints[medianIdx].x;
  }
  return median;
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate,
                     double &TTC) {
  sort(lidarPointsPrev.begin(), lidarPointsPrev.end(),
       [](LidarPoint &p1, LidarPoint &p2) { return p1.x < p2.x; });
  sort(lidarPointsCurr.begin(), lidarPointsCurr.end(),
       [](LidarPoint &p1, LidarPoint &p2) { return p1.x < p2.x; });

  int lidarPointsCurr_Q1Idx = lidarPointsCurr.size() / 4;
  int lidarPointsCurr_Q4Idx = lidarPointsCurr_Q1Idx * 3;
  int lidarPointsPrev_Q1Idx = lidarPointsPrev.size() / 4;
  int lidarPointsPRev_Q4Idx = lidarPointsPrev_Q1Idx * 3;

  float lidarPointsCurrIqr = 1.5 * (lidarPointsCurr[lidarPointsCurr_Q4Idx].x -
                                    lidarPointsCurr[lidarPointsCurr_Q1Idx].x);
  float lidarPointsPrevIqr = 1.5 * (lidarPointsPrev[lidarPointsCurr_Q4Idx].x -
                                    lidarPointsPrev[lidarPointsPrev_Q1Idx].x);

  float currXMean = calcXMedianLidar(lidarPointsCurr);
  float prevXMean = calcXMedianLidar(lidarPointsPrev);

  float currClosestDist;
  float prevClosestDist;
  for (int i = 0; i < lidarPointsCurr_Q1Idx; ++i) {
    if (currXMean - lidarPointsCurr[i].x < lidarPointsCurrIqr) {
      currClosestDist = lidarPointsCurr[i].x;
      break;
    }
  }
  for (int i = 0; i < lidarPointsPrev_Q1Idx; ++i) {
    if (prevXMean - lidarPointsPrev[i].x < lidarPointsPrevIqr) {
      prevClosestDist = lidarPointsPrev[i].x;
      break;
    }
  }
  // cout << prevClosestDist << " " << currClosestDist << " " << prevClosestDist - currClosestDist << endl;

  TTC = currClosestDist / frameRate / (prevClosestDist - currClosestDist);
}

void countMatchesBetweenFrames(vector<cv::DMatch> &matches,
                               DataFrame &prevFrame, DataFrame &currFrame,
                               int prevBbIdx, int currBbIdx,
                               map<int, int> &countMatchesInBB, mutex &mtx) {
  int countMatches = 0;

  // for each keypoint matches, check if it is inside both BB
  for (auto kpts_match : matches) {
    if (prevFrame.boundingBoxes[prevBbIdx].roi.contains(
            prevFrame.keypoints[kpts_match.queryIdx].pt) &&
        currFrame.boundingBoxes[currBbIdx].roi.contains(
            currFrame.keypoints[kpts_match.trainIdx].pt)) {
      countMatches++;
    }
  }
  const lock_guard<mutex> lck(mtx);
  countMatchesInBB.emplace(pair<int, int>(currBbIdx, countMatches));
}

void iterateOverPrevBB(std::vector<cv::DMatch> &matches,
                       std::map<int, int> &bbBestMatches, DataFrame &prevFrame,
                       DataFrame &currFrame, int prevBbIdx, set<int> &matchedBB,
                       mutex &mtx) {
  mutex countMatchesmtx;
  vector<thread> threads;

  // map of how many matches are found in candidate BB 
  // map<boxID, noOfKeypointsMatches>
  map<int, int> countMatchesInBB; 

  for (int currBbIdx = 0; currBbIdx < currFrame.boundingBoxes.size();
       ++currBbIdx) {
    // if a BB has been matched before, skip it
    if (matchedBB.find(currBbIdx) != matchedBB.end()) continue;
    threads.emplace_back(thread(&countMatchesBetweenFrames, ref(matches),
                                ref(prevFrame), ref(currFrame), prevBbIdx,
                                currBbIdx, ref(countMatchesInBB),
                                ref(countMatchesmtx)));
  }

  for (auto &thread : threads) {
    thread.join();
  }

  // find BB with the most matched keypoints
  auto bestBbMatch =
      max_element(countMatchesInBB.begin(), countMatchesInBB.end(),
                  [](pair<const int, int> &p1, pair<const int, int> &p2) {
                    return p1.second < p2.second;
                  });

  // check if roi between two tracked BB has similar area size
  float roi_similarity_threshold = 0.8;
  float prevFrameRoiArea = prevFrame.boundingBoxes[prevBbIdx].roi.area();
  float currFrameRoiArea =
      currFrame.boundingBoxes[bestBbMatch->first].roi.area();

  // if two BB are too different in size, skip it
  if (min(prevFrameRoiArea, currFrameRoiArea) /
          max(prevFrameRoiArea, currFrameRoiArea) <
      roi_similarity_threshold) {
    return;
  }

  // publish best matched BB
  const lock_guard<mutex> lck(mtx);
  bbBestMatches.emplace(pair<int, int>(prevBbIdx, bestBbMatch->first));
  matchedBB.insert(bestBbMatch->first);
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches,
                        std::map<int, int> &bbBestMatches, DataFrame &prevFrame,
                        DataFrame &currFrame) {
  mutex mtx;
  vector<thread> threads;

  // keep track of which BB in curr frame has been matched, to prevent looking
  // for mathces on this BB to reduce computation load
  set<int> matchedBB;

  for (int prevBbIdx = 0; prevBbIdx < prevFrame.boundingBoxes.size();
       ++prevBbIdx) {
    threads.emplace_back(thread(
        &iterateOverPrevBB, ref(matches), ref(bbBestMatches), ref(prevFrame),
        ref(currFrame), prevBbIdx, ref(matchedBB), ref(mtx)));
  }

  for (auto &thread : threads) {
    thread.join();
  }
}