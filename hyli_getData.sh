#!/bin/bash

# This script downloads videos for a specific sequence:
# ./getData.sh [sequenceName] [numVGAViews] [numHDViews]
#
# e.g., to download 10 VGA camera views for the "sampleData" sequence:
# ./getData.sh sampleData 10 0
#

#sampleName=160422_ultimatum1
sampleName=160422_haggling1
#bash ./scripts/getData.sh $sampleName 0
bash ./scripts/extractAll.sh $sampleName
