#ifndef G2O_ODO_SLAM_H_
#define G2O_ODO_SLAM_H_

int process(unsigned char *NIOBUFFER, int size, int capacity, int *OUTPUT, int maxIterations, bool useGain, bool checkInput, bool verbose);

#endif
