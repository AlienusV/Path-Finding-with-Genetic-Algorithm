# Path-Finding-with-Genetic-Algorithm

Dots try to find their path in a labyrinth using genetic algorithm. 

Populations:
There are different populations with different amount of members, brain sizes, decision and mutations rates and max speeds. Every aspect of these populations can be adjusted before starting the learn process or saving, tweaking the savefile and reloading. There are 16 different colors and 8 different shapes totalling to 128 differentiable populations.

Dots:
The members of these populations are called Dots. Every dot in a population are industinguishable except their brains. 

Brain:
A Dot brain is a 2D array consisting of 2D vectors. These 2D vectors are considered as an acceleration vector. These acceleration vectors are added one after another to create a velocity vector. This velocity vector is clipped according to their max speed and added to their position. The formula for the brain size of a population is FULL_PATH_LENGTH = BRAIN_SIZE X MAX_SPEED.  

Decision and Mutation Rates:
Decision Rate determines whether the Dot decides (the accelereation vector becomes the pointing unit vector from the dot to the next reward, this vector gets stored for the next generations) or remembers (remembers what their ancestor would do in this position). Mutation Rate affects every single brain cell and determines if the brain cell stays the same or gets mutated to a random vector.

Natural Selection:
The Dots get a fittness score according to their reached reward amount, their distance to their next reward and their brain steps. The Dots get ranked by their fittness scores and the top 20% becomes that generations best Dots. The next generation clones of these best 20% dots. After the cloning they go through a mutation phase.

Save Files:
The attributes of each population gets written in an Excel in a directory named by the starting time. Each population gets a different sheet. After all the attributes of the population gets written, every generations stats(Best Dots Name, Score, Brain Step Count, The amount of Dots that have reached the Goal(winners)). Every Populations has a binary file (.npy, about 500KiB for a brain with 5000 cells) that stores the best dots brain. The excel and brain files are backed up every minute.

Note: This is not the first version of this program. I've been working on this for a couple of months. Every version up until the implementation of the save files would be considered as an alpha version. This is the first stable version.

Inspired by: https://github.com/Code-Bullet/Smart-Dots-Genetic-Algorithm-Tutorial
