# 598 Repo

All code should be able to be downloaded and run as is, the data may need to be updated in the future along with the filepath for the new data.

The outputs should be directly downloaded into the home folder.

The 3 Measures files should output the BC/ASPL/CC for the graph for the cases where it is scrambled, averaged or unaltered.

CDF Files give the cumulative distribution function for the in/out degrees for the left/right hemispheres.

The high/low dose plots will give you the BC/ASPL/CC for each conditon all on the same plot for either the high or low dose. So you will get 3 plots, one for each condition, with each plot having all 4 states for that dosage (3 low/high dose + 1 control).

The plotNetwork file just gives you the network. The bottom 50% of linkweights are filtered out to aid visibility.
It is normalized per hemisphere, so the sum of the weights in a hemisphere will add up to some value. There is a weight factor which is then applied.

Often in the code, the control case will be called control, whereas altered state will be refered to as Iso, even if the file is working with ketamine or pentobarbital.

