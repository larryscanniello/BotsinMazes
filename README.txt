# BotsinMazes

Use runme.py, with all other files in the same directory, to open an interface to access all parts of the project.

This is a semester-long project I completed in my artificial intelligence class at Rutgers.
The idea of the project was to use classical AI concepts, and to get an introduction to machine learning.
It gave me a lot of practice with Python.
Soon, I will implement a GUI and enhance the visualizations.
For now, the simulations can be visualized by printing the grid/layout at each timestep to the file 'shipresults.txt'.

________________________________________________________________________________________________________________________________

Part 1: Fire-Extinguishing Bots

The goal is for the bots to reach the fire extinguisher square before they are consumed by fire. 
The fire spreads to a cell with probability 1-(1-q)^K, where:
  K is the number of adjacent cells that are on fire
  q is a flammability parameter, that can be selected randomly, or inputted manually. Higher q = more flammable cells.

Bot 1 - Plans the shortest path to the fire extinguisher, and does not deviate from this path
Bot 2 - At every step, replans the shortest path to the fire extinguisher
Bot 3 - At every step, replans the shortest path to the fire extinguisher and avoids fire adjacent squares if possible
Bot 4 - Runs 50 fire simulations at every time-step to assign every square a weight, then use Uniform Cost Search/Dijkstras to find lowest cost path. See write-up for explanation in painful detail.

Legend for visualization:

0 - open square
1 - closed square; bots cannot move to these
2 - fire square; if bot is in same square as fire, then that bot is done
5 - fire extinguisher
10 - bot 1
100 - bot 2
1000 - bot 3
10000 - bot 4

________________________________________________________________________________________________________________________________

Part 2: Mice-Catching Bots

The goal is for the bots to catch one or two mice as soon as possible.
There are two modes: Either the mice stay in the same place at every time step, or can move at each time step.
At each time step, the bots must choose between two options: Move or sense.
If the bot senses, it receives a beep with probability exp(-a(d-1)), where:
  d is the Manhattan distance from the bot to a mouse (slightly different calculation in two mouse case - see write up)
  a is the sensitivity of the sensor, which is a parameter that can be selected randomly or inputted manually.

Bot 1 - Senses, moves to the highest probability square, senses again, repeat
Bot 2 - Alternates moving and sensing
Bot 3 - Calculates prediction of future states, uses Uniform Cost Search / Dijkstras to select path with highest probability of a mouse, plans 5 steps at a time

Legend for visualization:

0 - open square
1 - closed square; bots and mice cannot move to these
2 - mice
10 - bot 1
100 - bot 2
1000 - bot 3
The following you will only see in the two mice case:
12 - mouse, bot 1 has caught this mouse
102 - mouse, bot 2 has caught this mouse
1002 - mouse, bot 3 has caught this mouse
112 - mouse, bot 1 and bot 2 have caught this mouse
1012 - mouse, bot 1 and bot 3 have caught this mouse
1102 - mouse, bot 2 and bot 3 have caught this mouse

________________________________________________________________________________________________________________________________

Part 3: Neural nets catch bots

This is the same setup as part 2, except we have added a fourth bot, a simple neural net.
This neural net was trained on hundreds of thousands of training examples using part 2 of the project.
Bot 3 remains the same as part 2.
See write-up to see the actual code for all of the neural nets I trained and full analysis.
The neural networks are classifiers. 
They take in data about the current state, and outputs probabilities of one of five classes/bot moves: Up, down, left, right, or sense.
The highest probability move is selected. If that is invalid move, the next highest probability move is selected, and so on.
A "bump" is when the highest probability move is invalid. My code will output the number of bumps as part of the summary data. This is an important error measure.
I decided to only include the option of running the stochastic neural net with one mouse with my project here.
The bot also has backtrack protection turned on.
This is like playing Mario Kart with the setting turned on so you can't drive off the edge.
The neural net is still doing the driving, but with some help.



