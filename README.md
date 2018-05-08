# Query Net
## Setup
+ examples require Jon Berliner's pytorch helper library pyt at https://github.com/jonberliner/pyt
## Files
+ **deepset2d.py**: class for building location-augmented deep sets expecting inputs of size (batch\_size x in\_channels x height x width)
+ **utils.py**: most importantly contains functions needed for cartesian ops, essential to location-augmented sets
+ **query_net.py**: (NOT WORKING YET) going to be the larger system that not only does inference, but is also responsible for collecting samples for inference over an image
