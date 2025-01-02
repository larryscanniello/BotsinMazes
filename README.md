# BotsinMazes

To run, use runme.py, with all other files in the same directory. This will open an interface to access all parts of the project.

This project is a semester-long project I completed in my artificial intelligence class at Rutgers.
The idea of the project was to use classical AI concepts, and to get an introduction to machine learning.
It gave me a lot of practice with coding in Python.
Soon, I will implement a GUI to enhance the visualizations.
For now, the simulations can be visualized by printing the grid/layout at each timestep to the file 'shipresults.txt'.

________________________________________________________________________________________________________________________________
Part 1: Fire-Extinguishing Bots

The goal is for the bots to reach the fire extinguisher square before they are consumed by fire.

Bot 1 - Plans the shortest path to the fire extinguisher, and does not deviate from this path
Bot 2 - At every step, replans the shortest path to the fire extinguisher
Bot 3 - At every step, replans the shortest path to the fire extinguisher and avoids fire adjacent squares if possible
Bot 4 - Runs 50 fire simulations at every time-step to assign every square a weight, then use Uniform Cost Search/Dijkstras to find lowest cost path. See write-up for explanation in painful detail.

Here is the legend for the visualization:

0 - open square
1 - closed square; bots cannot move to these
2 - fire square; if bot is in same square as fire, game over
5 - fire extinguisher
10 - bot 1
100 - bot 2
1000 - bot 3
10000 - bot 4

________________________________________________________________________________________________________________________________
Part 2: Mice-Catching Bots

The goal is for the bots to

