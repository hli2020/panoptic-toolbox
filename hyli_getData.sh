#!/bin/bash

# This script downloads videos for a specific sequence:
# ./getData.sh [sequenceName] [numVGAViews] [numHDViews]
#
# e.g., to download 10 VGA camera views for the "sampleData" sequence:
# ./getData.sh sampleData 10 0
#

declare -a arr=(
"160422_ultimatum1"
"160422_haggling1"
"160224_haggling1"
"160226_haggling1"
"160906_band1"
"160906_band2"
"160906_band3"
"160906_band4"
"160906_ian1"
"160906_ian2"
"160906_ian3"
"160906_ian5"
"160906_pizza1"
"161202_haggling1"
)

for sampleName in "${arr[@]}"
do
   echo "downloading and extracting .. $sampleName"
   # or do whatever with individual element of the array
   bash ./scripts/getData.sh $sampleName 0 4 data
   bash ./scripts/extractAll.sh $sampleName data
   echo "done!"
done
