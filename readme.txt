estiamte the scale explicitly

2015-8-13 update: update the GetMap function to change way in wich Mask is generated 

2015-8-14 update: clone from v2 and implement the scale estimation using CNN rather than HOG feature based.

2015-8-17 update: change appearance model update strategy to make it more adapteve to target appearance change (according to a random variable) 

2015-8-17 update: lower the learning rate for lnet and change the update probability from 0.5 to 0.3

2015-8-18 update: reset the learning rate for lnet to 8e-7 and the update probability to 0.5. The generated results are 8-2

2015-8-24 update: decrease the momentum for lnet from 0.6 to 0.4. 
The scale number is set to 9, but the scale window (hann_win) is kept unchanged (croped the central 9 values from the scale window generated using 33 scale number).
